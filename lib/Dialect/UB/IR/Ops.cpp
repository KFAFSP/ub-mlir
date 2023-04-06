/// Implements the UB dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Dialect/UB/IR/Ops.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "ub-mlir/Dialect/UB/IR/UB.h"

using namespace mlir;
using namespace mlir::ub;

/// Prints @p type if it is not NeverType.
static void printMaybeNever(OpAsmPrinter &p, Operation*, Type type)
{
    if (llvm::isa<NeverType>(type)) return;

    p << ": " << type;
}

/// Parses a @p type or NeverType.
static ParseResult parseMaybeNever(OpAsmParser &p, Type &type)
{
    if (p.parseOptionalColon()) {
        type = NeverType::get(p.getContext());
        return success();
    }

    return p.parseType(type);
}

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "ub-mlir/Dialect/UB/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PoisonOp
//===----------------------------------------------------------------------===//

void PoisonOp::print(OpAsmPrinter &p)
{
    if (getValue().isPoison()) {
        p << " : " << getType();
        return;
    }

    p << " " << getValue();
}

ParseResult PoisonOp::parse(OpAsmParser &p, OperationState &result)
{
    PoisonAttr value;
    if (!p.parseOptionalColon()) {
        Type type;
        if (p.parseType(type)) return failure();
        value = PoisonAttr::get(type);
    } else {
        if (p.parseAttribute(value)) return failure();
    }

    result.addAttribute(getAttributeNames()[0], value);
    result.addTypes(value.getType());
    return success();
}

LogicalResult PoisonOp::inferReturnTypes(
    MLIRContext*,
    Optional<Location>,
    ValueRange,
    DictionaryAttr attributes,
    RegionRange,
    SmallVectorImpl<Type> &result)
{
    const auto value = attributes.getAs<PoisonAttr>(getAttributeNames()[0]);
    if (!value) return failure();

    result.push_back(value.getType());
    return success();
}

namespace {

struct Unpoison : OpRewritePattern<PoisonOp> {
    using OpRewritePattern::OpRewritePattern;

    virtual LogicalResult
    matchAndRewrite(PoisonOp op, PatternRewriter &rewriter) const override
    {
        // Only applies to unpoisoned values.
        const auto valueAttr = op.getValue();
        if (valueAttr.isPoisoned()) return failure();

        // Delegate to the source dialect constant materializer.
        auto srcOp = valueAttr.getSourceDialect()->materializeConstant(
            rewriter,
            valueAttr.getSourceAttr(),
            valueAttr.getType(),
            op.getLoc());
        if (!srcOp) return failure();

        // Replace with source materialization.
        rewriter.replaceOp(op, srcOp->getResults());
        return success();
    }
};

} // namespace

void PoisonOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* ctx)
{
    patterns.add<Unpoison>(ctx);
}

OpFoldResult PoisonOp::fold(PoisonOp::FoldAdaptor adaptor)
{
    return adaptor.getValue();
}

//===----------------------------------------------------------------------===//
// FreezeOp
//===----------------------------------------------------------------------===//

OpFoldResult FreezeOp::fold(FreezeOp::FoldAdaptor adaptor)
{
    // Fold double freeze.
    if (getOperand().getDefiningOp<FreezeOp>()) return getOperand();

    // Do not fold if the input is not constant.
    if (!adaptor.getOperand()) return {};

    // Fold non-poisoned values, but without materializing the constant!
    const auto poisonAttr = llvm::dyn_cast<PoisonAttr>(adaptor.getOperand());
    if (!poisonAttr) return getOperand();

    // Do not fold poisoned values.
    return {};
}

//===----------------------------------------------------------------------===//
// NeverOp
//===----------------------------------------------------------------------===//

OpFoldResult NeverOp::fold(NeverOp::FoldAdaptor)
{
    return NeverAttr::get(getType());
}

namespace {

struct PropagateNever : OpRewritePattern<NeverOp> {
    using OpRewritePattern::OpRewritePattern;

    virtual LogicalResult
    matchAndRewrite(NeverOp op, PatternRewriter &rewriter) const override
    {
        auto propagated = false;
        for (auto user : op->getUsers()) {
            // Do not erase block terminators.
            if (user->getBlock()->getTerminator() == user) {
                // But do mark them as unreachable.
                propagated |= markAsUnreachable(user);
                continue;
            }

            // Replace all results of the user with never values, deleting the
            // op if it has no results at all.
            rewriter.replaceOp(
                user,
                llvm::to_vector(
                    llvm::map_range(user->getResultTypes(), [&](Type type) {
                        return rewriter.create<NeverOp>(user->getLoc(), type)
                            .getResult();
                    })));
            propagated = true;
        }

        return success(propagated);
    }
};

} // namespace

void NeverOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* ctx)
{
    patterns.add<PropagateNever>(ctx);
}

//===----------------------------------------------------------------------===//
// UBDialect
//===----------------------------------------------------------------------===//

void UBDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "ub-mlir/Dialect/UB/IR/Ops.cpp.inc"
        >();
}
