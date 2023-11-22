/// Implements the UBX dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Dialect/UBX/IR/Ops.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "ub-mlir/Dialect/UBX/IR/UBX.h"

using namespace mlir;
using namespace mlir::ubx;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "ub-mlir/Dialect/UBX/IR/Ops.cpp.inc"

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
    MLIRContext *,
    std::optional<Location>,
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
        if (op.getValue().isPoisoned()) return failure();

        // Try the freeze method of the PoisonAttrInterface implementation.
        OpBuilder builder(op);
        if (auto frozen = op.getValue().freeze(builder, op.getLoc())) {
            rewriter.replaceOp(op, frozen);
            return success();
        }

        return failure();
    }
};

} // namespace

void PoisonOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext *ctx)
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

    if (auto iface = llvm::dyn_cast_if_present<PoisonAttrInterface>(
            adaptor.getOperand())) {
        if (iface.isPoisoned()) {
            // Do not fold poisoned values! Even though there is a freeze method
            // on the interface, it loses information, and should only be done
            // when all else has failed.
            return {};
        }

        // Fall through here to fold away the freeze operation.
    }

    // Fold to the input value, which is known to be well-defined.
    return getOperand();
}

//===----------------------------------------------------------------------===//
// UBXDialect
//===----------------------------------------------------------------------===//

void UBXDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "ub-mlir/Dialect/UBX/IR/Ops.cpp.inc"
        >();
}
