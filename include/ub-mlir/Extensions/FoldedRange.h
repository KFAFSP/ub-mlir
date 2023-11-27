/// Declares the FoldedRange utility class.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/ValueRange.h"

#include "llvm/ADT/STLExtras.h"

#include <utility>

namespace mlir::ext {

/// Holds a range of folded operand values, accessible as OpFoldResult values.
class FoldedRange : public llvm::detail::indexed_accessor_range_base<
                        FoldedRange,
                        std::pair<ValueRange::OwnerT, const Attribute *>,
                        OpFoldResult,
                        OpFoldResult,
                        OpFoldResult> {
public:
    using OwnerT = std::pair<ValueRange::OwnerT, const Attribute *>;

    using RangeBaseT::RangeBaseT;

    /// Initializes an empty FoldedRange.
    /*implicit*/ FoldedRange() : FoldedRange({}, 0) {}

    /// Initializes a FoldedRange from @p operands and @p constantOperands .
    ///
    /// @pre    `operands.size() == constantOperands.size()`
    explicit FoldedRange(
        ValueRange operands,
        ArrayRef<Attribute> constantOperands)
            : FoldedRange(
                std::make_pair(operands.getBase(), constantOperands.data()),
                operands.size())
    {
        assert(operands.size() == constantOperands.size());
    }

    /// Gets the underlying values.
    [[nodiscard]] ValueRange getValues() const
    {
        return ValueRange(getBase().first, size());
    }
    /// Gets the underlying attributes.
    [[nodiscard]] ArrayRef<Attribute> getAttributes() const
    {
        return ArrayRef<Attribute>(getBase().second, size());
    }

    /// Gets a TypeRange over the underlying values.
    [[nodiscard]] ValueTypeRange<ValueRange> getTypes() const
    {
        return getValues().getTypes();
    }

    /// @copydoc getValues()
    [[nodiscard]] operator ValueRange() const { return getValues(); }
    /// @copydoc getAttributes()
    [[nodiscard]] operator ArrayRef<Attribute>() const
    {
        return getAttributes();
    }

private:
    /// See `llvm::detail::indexed_accessor_range_base` for details.
    [[nodiscard]] static OwnerT
    offset_base(const OwnerT &owner, std::ptrdiff_t index);
    /// See `llvm::detail::indexed_accessor_range_base` for details.
    [[nodiscard]] static OpFoldResult
    dereference_iterator(const OwnerT &owner, std::ptrdiff_t index);

    /// Allow access to `offset_base` and `dereference_iterator`.
    friend RangeBaseT;
};

} // namespace mlir::ext
