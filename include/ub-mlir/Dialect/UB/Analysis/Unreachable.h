/// Declares helpers for reachability analysis and propagation.
///
/// Unreachability is an eagerly propagated proposition in the IR. Users are
/// expected to canonicalize the IR after adding new unreachability markers.
///
/// The isKnownUnreachable() family of functions is pessimistic and only checks
/// for trivial cases, i.e., those produced by a markAsUnreachable() operation,
/// or through propagation during canonicalization.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "ub-mlir/Dialect/UB/IR/Base.h"
#include "ub-mlir/Dialect/UB/IR/Ops.h"
#include "ub-mlir/Dialect/UB/IR/Types.h"

namespace mlir::ub {

//===----------------------------------------------------------------------===//
// isSSACFG
//===----------------------------------------------------------------------===//

/// Determines whether @p region is an SSACFG region.
///
/// An SSACFG region is either a region of an operation that does not implement
/// the RegionKindInterface, or indicated to be an SSACFG region by that
/// interface implementation.
///
/// @pre    `region`
[[nodiscard]] bool isSSACFG(Region* region);
/// Determines whether @p block is an SSACFG block.
///
/// An SSACFG block is a block declared inside an SSACFG region.
///
/// @pre    `block`
[[nodiscard]] inline bool isSSACFG(Block* block)
{
    return isSSACFG(block->getParent());
}
/// Determines whether @p op is a terminator of an SSACFG block.
///
/// An SSACFG terminator must have the `IsTerminator` trait and be declared
/// inside an SSACFG block.
///
/// @pre    `op`
[[nodiscard]] inline bool isSSACFGTerminator(Operation* op)
{
    if (!op->hasTrait<OpTrait::IsTerminator>()) return false;
    return isSSACFG(op->getBlock());
}

//===----------------------------------------------------------------------===//
// isKnownUnreachable
//===----------------------------------------------------------------------===//

/// Determines whether @p value is known to be unreachable.
///
/// A value is known to be unreachable it if is produced by a NeverOp or has the
/// NeverType.
///
/// @pre    `value`
[[nodiscard]] inline bool isKnownUnreachable(Value value)
{
    return value.getDefiningOp<NeverOp>()
           || llvm::isa<NeverType>(value.getType());
}
/// Determines whether @p op is known to be unreachable.
///
/// An operation is known to be unreachable if it is an SSACFG terminator with
/// the `ub.unreachable` attribute.
///
/// @pre    `op`
[[nodiscard]] inline bool isKnownUnreachable(Operation* op)
{
    // NOTE: The attribute is only valid on SSACFG terminators.
    return op->hasAttr(kUnreachableAttrName);
}
/// Determines whether @p block is known to be unreachable.
///
/// A block is known to be unreachable if it is empty apart from an unreachable
/// SSACFG terminator.
///
/// @pre    `block`
[[nodiscard]] inline bool isKnownUnreachable(Block* block)
{
    return block->getOperations().size() == 1
           && isKnownUnreachable(block->getTerminator());
}

//===----------------------------------------------------------------------===//
// markAsUnreachable
//===----------------------------------------------------------------------===//

/// Marks all uses of @p value as unreachable.
///
/// Replaces all uses of @p value with never values, inserted at the earliest
/// possible insertion point. Returns @c true if any IR was modified.
///
/// @pre    `value`
bool markAsUnreachable(Value value);
/// Marks @p op as a known unreachable operation.
///
/// Replaces all operands of @p op with never values. If @p op is an SSACFG
/// terminator, marks it with the `ub.unreachable` attribute. Returns @c true
/// if any IR was modified.
///
/// @pre    `op`
bool markAsUnreachable(Operation* op);
/// Marks @p block as a known unreachable block.
///
/// If @p block is not an SSCAFG block, abort with @c false .
///
/// If @p killLiveOuts is @c false , aborts with @c false if any values declared
/// within this block are used outside of it. Otherwise, these uses are replaced
/// by never values.
///
/// Removes all operations from @p block except the terminator, and marks that
/// as unreachable. Returns @c true if any IR was modified.
///
/// @pre    `block`
bool markAsUnreachable(Block* block, bool killLiveOuts = false);

} // namespace mlir::ub
