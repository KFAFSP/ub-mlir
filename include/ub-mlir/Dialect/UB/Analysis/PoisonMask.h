/// Declares the PoisonMask bit set type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <utility>

namespace mlir::ub {

/// Holds a mask of values that are poisoned in a linearly indexed container.
///
/// Wraps an llvm::APInt to implement an underlying bit set. A bit value of `1`
/// indicates a poisoned value. By convention, all values outside of the
/// specified bit set are assumed to be `0`. Thus, the canonical representation
/// of a PoisonMask drops all zeroed bytes from the MSB.
class PoisonMask {
public:
    /// Type that holds the size of a PoisonMask.
    using size_type = unsigned;

    /// Initializes an empty poison mask.
    ///
    /// @post   `size() == 0`
    /// @post   `isEmpty()`
    /*implicit*/ PoisonMask() : m_impl(0U, 0UL)
    {
        assert(size() == 0);
        assert(isEmpty());
    }
    /// Initializes a PoisonMask of @p size to @p maskBit .
    ///
    /// @post   `size() == size`
    /// @post   `isPoison() == maskBit`
    /*implicit*/ PoisonMask(size_type size, bool maskBit = false)
            : m_impl(llvm::APInt(1U, maskBit).sext(size))
    {
        assert(this->size() == size);
        assert(isPoison(size) == maskBit);
    }
    /// Initializes a PoisonMask from @p ui .
    /*implicit*/ PoisonMask(llvm::APInt ui) : m_impl(std::move(ui)) {}

    //===------------------------------------------------------------------===//
    // Accessors
    //===------------------------------------------------------------------===//

    /// Gets the current size of the PoisonMask in bits.
    [[nodiscard]] size_type size() const { return m_impl.getBitWidth(); }

    /// Determines whether no element is masked.
    [[nodiscard]] bool isEmpty() const { return m_impl.isZero(); }
    /// Determines whether there are masked elements.
    [[nodiscard]] bool isPoisoned() const { return !isEmpty(); }
    /// Determines whether all @p ofSize elements are masked.
    [[nodiscard]] bool isPoison(size_type ofSize) const
    {
        return size() >= ofSize && m_impl.trunc(ofSize).isAllOnes();
    }

    /// Gets the PoisonMask as an llvm::APInt value.
    const llvm::APInt &asUInt() const { return m_impl; }

    //===------------------------------------------------------------------===//
    // Equality comparison
    //===------------------------------------------------------------------===//

    /// Determines whether @p other is the same mask.
    [[nodiscard]] bool operator==(const llvm::APInt &other) const
    {
        if (size() <= other.getBitWidth())
            return asUInt().zext(other.getBitWidth()) == other;
        else
            return asUInt() == other.zext(size());
    }
    /// @copydoc operator==(const llvm::APInt &)
    [[nodiscard]] bool operator==(const PoisonMask &other) const
    {
        return *this == other.asUInt();
    }

    /// Computes a hash value for @p mask .
    [[nodiscard]] friend llvm::hash_code hash_value(const PoisonMask &mask)
    {
        return llvm::hash_value(mask.asUInt());
    }

    //===------------------------------------------------------------------===//
    // Bit set manipulators
    //===------------------------------------------------------------------===//

    /// Resizes the mask to @p newSize elements, filling with @p maskBit .
    ///
    /// @post   `size() == newSize`
    void resize(size_type newSize, bool maskBit = false);

    /// Determines whether @p index is poisoned.
    [[nodiscard]] bool isPoisoned(size_type index) const
    {
        return index >= size() || m_impl[index];
    }
    /// Marks @p index as @p poison .
    ///
    /// @post   `!poison || size() > index`
    void setPoisoned(size_type index, bool poison = true)
    {
        if (poison && index >= size()) m_impl = m_impl.zext(index + 1);
        m_impl.setBitVal(index, poison);
    }

    /// Unites this PoisonMask with @p rhs .
    ///
    /// @post   `size() == std::max(size(), rhs.size())`
    void unite(const PoisonMask &rhs)
    {
        if (size() >= rhs.size())
            m_impl |= rhs.asUInt().zext(size());
        else
            m_impl = m_impl.zext(rhs.size()) | rhs.asUInt();
    }

    //===------------------------------------------------------------------===//
    // Serialization
    //===------------------------------------------------------------------===//

    /// Writes a PoisonMask to @p out as a hexadecimal literal.
    ///
    /// The format emitted by this printer is given by:
    ///
    /// @verbatim
    /// poison-mask ::= [_0-9a-fA-F]*
    /// @endverbatim
    ///
    /// The printer only emits whole bytes (pairs of nibbles / hex digits), but
    /// omits all leading zero bytes.
    void print(llvm::raw_ostream &out) const;

    /// Writes @p mask to @p out as a hexadecimal bitstring literal.
    ///
    /// See write(llvm::raw_ostream &) for more information.
    friend llvm::raw_ostream &
    operator<<(llvm::raw_ostream &out, const PoisonMask &mask)
    {
        mask.print(out);
        return out;
    }
    /// Writes @p mask to @p printer as a quoted hexadecimal literal.
    ///
    /// This overload wraps the literal in `"` quotes to ensure it can be
    /// unambiguously recovered using an associated mlir::FieldParser.
    ///
    /// See write(llvm::raw_ostream &) for more information.
    friend AsmPrinter &operator<<(AsmPrinter &printer, const PoisonMask &mask)
    {
        printer.getStream() << '"' << mask << '"';
        return printer;
    }

private:
    llvm::APInt m_impl;
};

} // namespace mlir::ub

namespace mlir {

template<>
struct FieldParser<ub::PoisonMask> {
    static FailureOr<ub::PoisonMask> parse(AsmParser &parser);
};

template<>
struct FieldParser<std::optional<ub::PoisonMask>> {
    static FailureOr<std::optional<ub::PoisonMask>> parse(AsmParser &parser);
};

} // namespace mlir
