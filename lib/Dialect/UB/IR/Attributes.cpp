/// Implements the UB dialect attributes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Dialect/UB/IR/Attributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "ub-mlir/Dialect/UB/IR/UB.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ub;

//===- Generated implementation -------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "ub-mlir/Dialect/UB/IR/Attributes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PoisonedElementsAttr
//===----------------------------------------------------------------------===//

Attribute PoisonedElementsAttr::get(
    DialectRef dialect,
    ElementsAttr elements,
    MaskAttr mask)
{
    if (!mask) return elements;
    if (mask.isSplat()) {
        if (mask.getSplatValue())
            return PoisonAttr::get(elements.getType());
        else
            return elements;
    }

    return Base::get(dialect->getContext(), dialect, elements, mask);
}

void PoisonedElementsAttr::print(AsmPrinter &p) const
{
    // `<` $sourceDialect `(`
    p << "<" << getSourceDialect()->getNamespace() << "(";

    // $elements
    p.printAttributeWithoutType(getElements());

    // `)` `[`
    p << ")[";

    // $mask
    p.printAttributeWithoutType(getMask());

    // `]` `>`
    p << "]>";
}

Attribute PoisonedElementsAttr::parse(AsmParser &p, Type type)
{
    const auto shapedTy = llvm::dyn_cast<ShapedType>(type);
    if (!shapedTy) {
        p.emitError(p.getNameLoc(), "invalid type");
        return {};
    }
    const auto i1Ty = IntegerType::get(p.getContext(), 1);
    const auto maskTy = RankedTensorType::get(shapedTy.getShape(), i1Ty);

    // `<`
    if (p.parseLess()) return {};

    // $sourceDialect `(`
    const auto dialectLoc = p.getCurrentLocation();
    llvm::StringRef sourceName;
    if (p.parseKeyword(&sourceName) || p.parseLParen()) return {};
    const auto sourceDialect = p.getContext()->getOrLoadDialect(sourceName);
    if (!sourceDialect) {
        p.emitError(dialectLoc)
            << "unknown source dialect '" << sourceName << "'";
        return {};
    }

    // $elements
    ElementsAttr elements;
    if (p.parseAttribute(elements, type)) return {};

    // `)` `[`
    if (p.parseRParen() || p.parseLSquare()) return {};

    // $mask
    MaskAttr mask;
    if (p.parseAttribute(mask, maskTy)) return {};

    // `]` `>`
    if (p.parseRSquare() || p.parseGreater()) return {};

    return getChecked(
        [&]() -> InFlightDiagnostic { return p.emitError(p.getNameLoc()); },
        p.getContext(),
        sourceDialect,
        elements,
        mask);
}

LogicalResult PoisonedElementsAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    DialectRef sourceDialect,
    ElementsAttr elements,
    MaskAttr mask)
{
    if (!sourceDialect) return emitError() << "expected source dialect";

    if (!elements) return emitError() << "expected source attribute";

    if (!mask) return emitError() << "expected mask attribute";

    if (mask.size() != elements.size())
        return emitError()
               << "mask size (" << mask.size() << ")"
               << "does not match elements size (" << elements.size() << ")";

    return success();
}

//===----------------------------------------------------------------------===//
// UBDialect
//===----------------------------------------------------------------------===//

void UBDialect::registerAttributes()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "ub-mlir/Dialect/UB/IR/Attributes.cpp.inc"
        >();
}
