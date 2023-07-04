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

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "ub-mlir/Dialect/UB/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PoisonOp
//===----------------------------------------------------------------------===//

void PoisonOp::print(OpAsmPrinter &p)
{
    if (getValue().isa<PoisonAttr>()) {
        // Short form
        p << " : " << getType();
        return;
    }

    p << " " << getValue();
}

ParseResult PoisonOp::parse(OpAsmParser &p, OperationState &result)
{
    TypedAttr value;
    if (!p.parseOptionalColon()) {
        // Short form
        Type type;
        if (p.parseType(type)) return failure();
        value = PoisonAttr::get(type);
    } else {
        PoisonedElementsAttr attr;
        if (p.parseCustomAttributeWithFallback(attr)) return failure();
        value = attr;
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
    OpaqueProperties,
    RegionRange,
    SmallVectorImpl<Type> &result)
{
    const auto value = attributes.getAs<TypedAttr>(getAttributeNames()[0]);
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
        // Only applies to well-defined elements.
        const auto elementsAttr =
            llvm::dyn_cast<PoisonedElementsAttr>(op.getValue());
        if (!elementsAttr || elementsAttr.isPoisoned()) return failure();

        // Delegate to the source dialect constant materializer.
        auto srcOp = elementsAttr.getSourceDialect()->materializeConstant(
            rewriter,
            elementsAttr.getElements(),
            elementsAttr.getType(),
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

    // Do not fold if the input is poisoned in any way.
    if (llvm::isa<PoisonedAttr>(adaptor.getOperand())) return {};

    // Fold to the non-poisoned operand value, but not the constant value, to
    // avoid calling our own dialect materializer on it.
    return getOperand();
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
