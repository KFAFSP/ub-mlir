/// Implements the UB dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Dialect/UB/IR/Ops.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "ub-ops"

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
// UBDialect
//===----------------------------------------------------------------------===//

void UBDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "ub-mlir/Dialect/UB/IR/Ops.cpp.inc"
        >();
}
