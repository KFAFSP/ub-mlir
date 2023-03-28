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
/// its dimensions. For dynamic shapes, this is ShapedType::kDynamic;
static std::int64_t getNumElements(Type type)
{
    if (const auto shapedTy = type.dyn_cast<ShapedType>()) {
        if (!shapedTy.hasStaticShape()) return ShapedType::kDynamic;

        return std::accumulate(
            shapedTy.getShape().begin(),
            shapedTy.getShape().end(),
            std::int64_t(1),
            std::multiplies<std::int64_t>{});
    }

    return 1;
}

/// Unites two bit masks regardless of their width.
static llvm::APInt uniteMasks(const llvm::APInt &lhs, const llvm::APInt &rhs)
{
    const auto bitWidth = std::max(lhs.getBitWidth(), rhs.getBitWidth());
    return lhs.zext(bitWidth) | rhs.zext(bitWidth);
}

/// Parses an optional poison mask as hex literal.
static OptionalParseResult
parseOptionalPoisonMask(AsmParser &p, llvm::APInt &result)
{
    std::string hexLiteral;
    if (p.parseOptionalString(&hexLiteral)) return std::nullopt;
    if (StringRef(hexLiteral).getAsInteger(16, result))
        return p.emitError(p.getCurrentLocation(), "expected hex literal");
    return success();
}

/// Prints a poison mask as a hex literal.
static void printPoisonMask(AsmPrinter &p, const llvm::APInt &value)
{
    p << "\"";
    // NOTE: The default APInt parser does not handle very long values, see
    //       bit::BitSequence. We thus emit a long hex literal in canonical (BE)
    //       order.
    // TODO: Factor this out into its own FieldParser and helper.
    const auto activeWords =
        ArrayRef<std::uint64_t>(value.getRawData(), value.getActiveWords());
    SmallVector<std::uint8_t, sizeof(llvm::APInt::WordType)> buffer;
    for (auto word : llvm::reverse(activeWords)) {
        const auto begin = reinterpret_cast<const std::uint8_t*>(&word);
        const auto end = begin + sizeof(word);

        if (std::endian::native == std::endian::big) {
            buffer.assign(begin, end);
        } else {
            buffer.assign(
                std::make_reverse_iterator(end),
                std::make_reverse_iterator(begin));
        }
        p << llvm::toHex(buffer);
    }
    p << "\"";
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
    llvm::APInt poisonMask)
{
    const auto makeFullPoison = [&]() {
        return getFn(
            sourceAttr.getContext(),
            nullptr,
            sourceAttr,
            llvm::APInt(0U, 0UL));
    };

    assert(sourceAttr);

    if (sourceAttr.isa<TypeAttr>()) {
        // Simplify to full poison attribute.
        return makeFullPoison();
    }

    // Flatten nested hierarchies (technically, only one level may ever occur).
    while (const auto poisonAttr = sourceAttr.dyn_cast<PoisonAttr>()) {
        sourceAttr = poisonAttr.getSourceAttr();
        poisonMask = uniteMasks(poisonMask, poisonAttr.getPoisonMask());
    }

    // Simplify poison masks.
    const auto sourceTy = sourceAttr.cast<TypedAttr>().getType();
    const auto numElements = getNumElements(sourceTy);
    if (numElements != ShapedType::kDynamic) {
        poisonMask = poisonMask.zextOrTrunc(numElements);
        const auto fullMask = llvm::APInt::getAllOnes(numElements);
        if (poisonMask == fullMask) return makeFullPoison();
    }

    // NOTE: We can't simplify poisonMask.isZero(), because we must carry the
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
    llvm::APInt poisonMask)
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
    llvm::APInt poisonMask)
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
    if (!getPoisonMask().isZero()) {
        printPoisonMask(p, getPoisonMask());
        p << ", ";
    }

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
    llvm::APInt poisonMask(0U, 0UL);
    const auto optMask = parseOptionalPoisonMask(p, poisonMask);
    if (optMask.has_value()) {
        if (optMask.value()) return {};
        if (p.parseComma()) return {};
    }

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
        poisonMask);
}

LogicalResult PoisonAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    DialectRef sourceDialect,
    TypedOrTypeAttr sourceAttr,
    llvm::APInt)
{
    // Must have a source attribute.
    if (!sourceAttr) return emitError() << "expected source attribute";

    // Full poison is indicated by a TypeAttr.
    if (sourceAttr.isa<TypeAttr>()) {
        if (sourceDialect) return emitError() << "no source dialect allowed";
        return success();
    }

    // Nesting is not allowed.
    if (sourceAttr.isa<PoisonAttr>())
        return emitError() << "PoisonAttr may not be nested";
    // Otherwise, must have a source dialect.
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
