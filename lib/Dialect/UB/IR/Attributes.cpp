/// Implements the UB dialect attributes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Dialect/UB/IR/Attributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "ub-mlir/Dialect/UB/IR/UB.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <bit>
#include <numeric>

using namespace mlir;
using namespace mlir::ub;

/// Gets the number of elements contained in @p type .
///
/// For a scalar type, this is 1. For a container type, this is the product of
/// its dimensions. For dynamic shapes, this is ShapedType::kDynamic.
[[nodiscard]] static std::int64_t getNumElements(Type type)
{
    // TODO: Support for more aggregate types based on an Attribute interface?
    //       E.g., it is unclear whether tuples are stored nested or flattened.

    if (const auto shapedTy = llvm::dyn_cast<ShapedType>(type)) {
        if (!shapedTy.hasStaticShape()) return ShapedType::kDynamic;

        return std::accumulate(
            shapedTy.getShape().begin(),
            shapedTy.getShape().end(),
            std::int64_t(1),
            std::multiplies<std::int64_t>{});
    }

    return 1;
}

//===- Generated implementation -------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "ub-mlir/Dialect/UB/IR/Attributes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PoisonAttr
//===----------------------------------------------------------------------===//

static PoisonAttr getImpl(
    auto getFn,
    DialectRef sourceDialect,
    TypedOrTypeAttr sourceAttr,
    PoisonMask poisonMask)
{
    const auto makeFullPoison = [&]() {
        return getFn(
            sourceAttr.getContext(),
            nullptr,
            sourceAttr,
            PoisonMask{});
    };

    assert(sourceAttr);

    if (llvm::isa<TypeAttr>(sourceAttr)) {
        // Simplify to full poison attribute.
        return makeFullPoison();
    }

    // Flatten nested hierarchies (technically, only one level may ever occur).
    while (const auto poisonAttr = llvm::dyn_cast<PoisonAttr>(sourceAttr)) {
        sourceAttr = poisonAttr.getSourceAttr();
        poisonMask.unite(poisonAttr.getPoisonMask());
    }

    // Simplify poison mask if number of elements in known.
    const auto sourceTy = llvm::cast<TypedAttr>(sourceAttr).getType();
    const auto numElements = getNumElements(sourceTy);
    if (numElements != ShapedType::kDynamic) {
        if (poisonMask.isPoison(numElements)) return makeFullPoison();
    }

    // NOTE: We can't simplify poisonMask.isEmpty(), because we must carry the
    //       sourceDialect for potentially external constant materializers!

    // Build a partially poisoned value attr.
    return getFn(
        sourceAttr.getContext(),
        sourceDialect,
        sourceAttr,
        poisonMask);
}

PoisonAttr PoisonAttr::get(
    DialectRef sourceDialect,
    TypedOrTypeAttr sourceAttr,
    const PoisonMask &poisonMask)
{
    return ::getImpl(
        [&](auto &&... args) {
            return Base::get(std::forward<decltype(args)>(args)...);
        },
        sourceDialect,
        sourceAttr,
        poisonMask);
}

PoisonAttr PoisonAttr::getChecked(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    DialectRef sourceDialect,
    TypedOrTypeAttr sourceAttr,
    const PoisonMask &poisonMask)
{
    return ::getImpl(
        [&](auto &&... args) {
            return Base::getChecked(
                emitError,
                std::forward<decltype(args)>(args)...);
        },
        sourceDialect,
        sourceAttr,
        poisonMask);
}

void PoisonAttr::print(AsmPrinter &p) const
{
    if (!getSourceDialect()) return;

    p << "<";

    // Print poison mask.
    if (!getPoisonMask().isEmpty()) p << getPoisonMask() << ", ";

    // Print source dialect name;
    p << getSourceDialect()->getNamespace();

    // Print source attribute without type.
    p << "(";
    p.printAttributeWithoutType(getSourceAttr());
    p << ")>";
}

Attribute PoisonAttr::parse(AsmParser &p, Type type)
{
    if (p.parseOptionalLess()) {
        if (!type) {
            if (p.parseColon()) return {};
            if (p.parseType(type)) return {};
        }
        return PoisonAttr::get(type);
    }

    // Parse poison mask.
    const auto optMask = FieldParser<std::optional<PoisonMask>>::parse(p);
    if (failed(optMask)) return {};
    if (optMask->has_value() && p.parseComma()) return {};

    // Parse source dialect name.
    StringRef sourceDialectName;
    if (p.parseKeyword(&sourceDialectName)) return {};

    // Parse source attribute.
    if (p.parseLParen()) return {};
    TypedOrTypeAttr sourceAttr;
    if (p.parseAttribute(sourceAttr, type)) return {};
    if (p.parseRParen()) return {};

    if (p.parseGreater()) return {};
    if (!type) {
        if (p.parseColon()) return {};
        if (p.parseType(type)) return {};
    }

    return PoisonAttr::getChecked(
        [&]() { return p.emitError(p.getNameLoc()); },
        sourceDialectName,
        sourceAttr,
        optMask->value_or(PoisonMask{}));
}

LogicalResult PoisonAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    DialectRef sourceDialect,
    TypedOrTypeAttr sourceAttr,
    PoisonMask)
{
    if (!sourceAttr) return emitError() << "expected source attribute";
    if (llvm::isa<TypeAttr>(sourceAttr)) {
        if (sourceDialect) return emitError() << "no source dialect allowed";
        return success();
    }
    if (llvm::isa<PoisonAttr>(sourceAttr))
        return emitError() << "PoisonAttr may not be nested";
    if (!sourceDialect) return emitError() << "expected source dialect";
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
