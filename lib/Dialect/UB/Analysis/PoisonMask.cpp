/// Implements the PoisonMask bit set type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Dialect/UB/Analysis/PoisonMask.h"

#include "llvm/ADT/StringExtras.h"

#include <algorithm>
#include <bit>
#include <iterator>

using namespace mlir;
using namespace mlir::ub;

/// Gets the bytes of @p data .
static ArrayRef<std::uint8_t> getBytes(const llvm::APInt::WordType &data)
{
    return ArrayRef<std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(&data),
        sizeof(llvm::APInt::WordType));
}

/// Prints @p data as a hexadecimal string to @p os in big-endian order.
static void printAsHexBE(llvm::raw_ostream &os, ArrayRef<std::uint8_t> data)
{
    const auto printByteBE = [&](std::uint8_t byte) {
        os << llvm::hexdigit(byte >> 4, false);
        os << llvm::hexdigit(byte & 15, false);
    };

    if constexpr (std::endian::native == std::endian::big)
        llvm::for_each(data, printByteBE);
    else
        llvm::for_each(llvm::reverse(data), printByteBE);
}
/// @copydoc printAsHexBE(llvm::raw_ostream &, ArrayRef<std::uint8_t>)
static void
printAsHexBE(llvm::raw_ostream &os, const llvm::APInt::WordType &data)
{
    printAsHexBE(os, getBytes(data));
}

//===----------------------------------------------------------------------===//
// PoisonMask
//===----------------------------------------------------------------------===//

void PoisonMask::resize(size_type newSize, bool maskBit)
{
    if (!maskBit) {
        m_impl = m_impl.zextOrTrunc(newSize);
        return;
    }

    if (newSize < size()) {
        m_impl = m_impl.trunc(newSize);
        return;
    }

    // Remember the current MSB.
    const auto oldSize = size();
    const auto oldBit = m_impl.isSignBitSet();

    // Set the MSB to be able to sign-extend.
    m_impl.setSignBit();
    m_impl = m_impl.sext(newSize);

    // If the old MSB was changed, undo it.
    if (!oldBit) m_impl.clearBit(oldSize);
}

void PoisonMask::print(llvm::raw_ostream &out) const
{
    // Only get non-zero words, but at least the LSB.
    ArrayRef<std::uint64_t> activeWords(
        m_impl.getRawData(),
        m_impl.getActiveWords());
    assert(!activeWords.empty());

    // Remove the MSB for separate printing.
    auto msbBytes = getBytes(activeWords.back());
    activeWords = activeWords.drop_back(1);

    // Print the trimmed MSB.
    if constexpr (std::endian::native == std::endian::big)
        while (msbBytes.size() > 1 && msbBytes.front() == 0)
            msbBytes = msbBytes.drop_front(1);
    else
        while (msbBytes.size() > 1 && msbBytes.back() == 0)
            msbBytes = msbBytes.drop_back(1);
    printAsHexBE(out, msbBytes);

    // Print all the rest.
    llvm::for_each(llvm::reverse(activeWords), [&](const auto &word) {
        printAsHexBE(out, word);
    });
}

//===----------------------------------------------------------------------===//
// FieldParser<ub::PoisonMask>
//===----------------------------------------------------------------------===//

FailureOr<PoisonMask> mlir::FieldParser<PoisonMask>::parse(AsmParser &parser)
{
    auto optResult = FieldParser<std::optional<PoisonMask>>::parse(parser);
    if (succeeded(optResult)) {
        if (optResult->has_value()) return std::move(optResult->value());

        return parser.emitError(parser.getCurrentLocation(), "expected '\"'");
    }

    return failure();
}

//===----------------------------------------------------------------------===//
// FieldParser<std::optional<PoisonMask>>
//===----------------------------------------------------------------------===//

FailureOr<std::optional<PoisonMask>>
mlir::FieldParser<std::optional<PoisonMask>>::parse(AsmParser &parser)
{
    std::string str;
    if (parser.parseOptionalString(&str))
        return success(std::optional<PoisonMask>(std::nullopt));

    llvm::APInt result;
    if (StringRef(str).getAsInteger(16, result))
        return parser.emitError(
            parser.getCurrentLocation(),
            "expected hex literal");

    return success(PoisonMask(std::move(result)));
}
