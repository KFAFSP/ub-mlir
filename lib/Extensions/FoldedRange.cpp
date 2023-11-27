/// Implements the FoldedRange utility class.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Extensions/FoldedRange.h"

using namespace mlir;
using namespace mlir::ext;

// NOTE: Unfortunately, the ValueRange::offset_base and
//       ValueRange::dereference_iterator static member functions are
//       inaccessible to us, and so we had to copy their implementation.

[[nodiscard]] static ValueRange::OwnerT
offset_base(const ValueRange::OwnerT &owner, std::ptrdiff_t index)
{
    if (const auto *value = llvm::dyn_cast_if_present<const Value *>(owner))
        return {value + index};
    if (auto *operand = llvm::dyn_cast_if_present<OpOperand *>(owner))
        return {operand + index};
    return owner.get<mlir::detail::OpResultImpl *>()->getNextResultAtOffset(
        index);
}

[[nodiscard]] static Value
dereference_iterator(const ValueRange::OwnerT &owner, std::ptrdiff_t index)
{
    if (const auto *value = llvm::dyn_cast_if_present<const Value *>(owner))
        return value[index];
    if (auto *operand = llvm::dyn_cast_if_present<OpOperand *>(owner))
        return operand[index].get();
    return owner.get<mlir::detail::OpResultImpl *>()->getNextResultAtOffset(
        index);
}

//===----------------------------------------------------------------------===//
// FoldedRange
//===----------------------------------------------------------------------===//

FoldedRange::OwnerT
FoldedRange::offset_base(const OwnerT &owner, std::ptrdiff_t index)
{
    return std::make_pair(
        ::offset_base(owner.first, index),
        owner.second + index);
}

OpFoldResult
FoldedRange::dereference_iterator(const OwnerT &owner, std::ptrdiff_t index)
{
    if (auto attr = owner.second[index]) return attr;
    return ::dereference_iterator(owner.first, index);
}
