/// Declares the reachability analysis.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "ub-mlir/Dialect/UB/IR/Base.h"
#include "ub-mlir/Dialect/UB/IR/Ops.h"
#include "ub-mlir/Dialect/UB/IR/Types.h"

#include <algorithm>
#include <utility>

namespace mlir::ub {

//===----------------------------------------------------------------------===//
// IR traits
//===----------------------------------------------------------------------===//

/// Determines whether @p op is a terminator.
///
/// @pre    `op`
[[nodiscard]] inline bool isTerminator(Operation* op)
{
    return op->hasTrait<OpTrait::IsTerminator>();
}
/// Determines whether @p op is a return operation.
///
/// @pre    `op`
[[nodiscard]] inline bool isReturn(Operation* op)
{
    return op->hasTrait<OpTrait::ReturnLike>();
}
/// Determines whether @p op is a RegionBranchTerminatorOp.
///
/// @pre    `op`
[[nodiscard]] inline bool isRegionBranchTerminator(Operation* op)
{
    return llvm::isa<RegionBranchTerminatorOpInterface>(op);
}

/// Determines whether @p region is an SSACFG region.
///
/// @pre    `region`
[[nodiscard]] inline bool isSSACFG(Region* region)
{
    // NOTE: An operation that does not implement RegionKindInterface is
    //       assumed to have only SSACFG regions per MLIR core!
    auto iface =
        llvm::dyn_cast_if_present<RegionKindInterface>(region->getParentOp());
    if (!iface) return true;

    return iface.getRegionKind(region->getRegionNumber()) == RegionKind::SSACFG;
}
/// Determines whether @p block is an SSACFG block.
///
/// @pre    `block`
[[nodiscard]] inline bool isSSACFG(Block* block)
{
    return isSSACFG(block->getParent());
}
/// Determines whether @p op is inside an SSACFG block.
///
/// @pre    `op`
[[nodiscard]] inline bool isSSACFG(Operation* op)
{
    return op->getBlock() && isSSACFG(op->getBlock());
}

//===----------------------------------------------------------------------===//
// Pessimistic reachability tests
//===----------------------------------------------------------------------===//
//
// These functions test the reachability propositions `unreachable` and
// `noreturn` using shallow but inexpensive tests on the IR. More specifically,
// if the input IR is fully propagated and canonicalized, these functions will
// detect all applicable propositions.
//

/// Determines whether @p value is known to be unreachable.
///
/// A value is known to be unreachable if it has the NeverType or is produced by
/// a NeverOp.
///
/// @pre    `value`
[[nodiscard]] inline bool isKnownUnreachable(Value value)
{
    return llvm::isa_and_present<NeverOp>(value.getDefiningOp())
           || llvm::isa<NeverType>(value.getType());
}
/// Determines whether @p op is known to be unreachable.
///
/// An operation is known to be unreachable if it is an UnreachableOp, has the
/// `ub.unreachable` attribute, or any of its operands are known to be
/// unreachable.
///
/// @pre    `op`
[[nodiscard]] inline bool isKnownUnreachable(Operation* op)
{
    return llvm::isa<UnreachableOp>(op) || op->hasAttr(kUnreachableAttrName)
           || llvm::any_of(
               op->getOperands(),
               static_cast<bool (*)(Value)>(&isKnownUnreachable));
}

/// Determines whether @p op is known to not return.
///
/// An operation is known to not return if it is a NeverOp, or returns a value
/// of NeverType.
///
/// @pre    `op`
[[nodiscard]] inline bool isKnownNoReturn(Operation* op)
{
    return llvm::isa<NeverOp>(op)
           || llvm::any_of(
               op->getResultTypes(),
               [](Type type) { return llvm::isa<NeverType>(type); });
}
/// Determines whether @p block is known to not return.
///
/// A block is known to not return if its terminator is known to be unreachable.
///
/// @pre    `block`
[[nodiscard]] inline bool isKnownNoReturn(Block* block)
{
    return !block->empty() && isTerminator(&block->back())
           && isKnownUnreachable(&block->back());
}

/// Determines whether @p block is known to be unreachable.
///
/// A block is known to be unreachable if all of its predecessors are known to
/// not return.
///
/// @pre    `block`
[[nodiscard]] inline bool isKnownUnreachable(Block* block)
{
    return !block->isEntryBlock()
           && llvm::all_of(
               block->getPredecessors(),
               static_cast<bool (*)(Block*)>(&isKnownNoReturn));
}

//===----------------------------------------------------------------------===//
// Optimistic reachability constraints
//===----------------------------------------------------------------------===//
//
// These functions apply the propositions `unreachable` and `noreturn` to the
// IR in a way that the pessimistic test will detect them. The resulting IR is
// not canonicalized, and transitive reachability is not marked / detected.
//

/// Marks @p value as known to be unreachable.
///
/// Replaces all uses of @p value with a never values materialized at the
/// earliest possible insertion point. Returns @c true if any IR was modified.
///
/// @pre    `value`
bool markAsUnreachable(Value value);
/// Marks @p op as known to be unreachable.
///
/// Replaces all operands of @p op with never values materialized directly
/// before @p op . Returns @c true if any IR was modified.
///
/// Given the input `op`:
/// @code{.unparsed}
/// %r:n = <op> (%o0, ..., %om)
/// @endcode
///
/// The resulting IR will be:
/// @code{.unparsed}
/// %n0 = ub.never
/// ...
/// %nm = ub.never
/// %r:n = <op> (%n0, ..., %nm)
/// @endcode
///
/// @pre    `op`
bool markAsUnreachable(Operation* op);
/// Marks @p block as known to be unreachable, starting with @p pos .
///
/// If @p block is not an SSACFG block, returns @c false .
///
/// Otherwise, splits @p block at @p pos , and unconditionally branches to the
/// operations starting from @p pos with an unreachable `cf.br` operator.
/// Returns @c true if any IR was modified.
///
/// Given the input `^bb0` and `pos`:
/// @code{.unparsed}
/// ^bb0(%a0, ..., %an):
///     ...
///     %r:m = <pos> (%o0, ..., %om)
///     ...
/// @endcode
///
/// The resulting IR will be:
/// @code{.unparsed}
/// ^bb0(%a0, ..., %an):
///     ...
///     %n0 = ub.never
///     ...
///     %nn = ub.never
///     cf.br ^split(%n0, ..., %nn) {ub.unreachable}
///
/// ^split(%a0, ..., %an):
///     ...
/// @endcode
///
/// @pre    `block`
/// @pre    `pos >= block.begin() && pos <= block.end()`
bool markAsUnreachable(Block* block, Block::iterator pos);
/// Marks @p block as known to be unreachable.
///
/// Starts unreachability at the entry of @p block . See
/// markAsUnreachable(Block, Block::iterator) for more details.
///
/// @pre    `block`
inline bool markAsUnreachable(Block* block)
{
    return markAsUnreachable(block, block->begin());
}

/// Marks @p op as known to not return.
///
/// Marks all results of @p op known to be unreachable. Returns @c true if any
/// IR was modified.
///
/// @pre    `op`
inline bool markAsNoReturn(Operation* op)
{
    return llvm::count_if(
        op->getResults(),
        static_cast<bool (*)(Value)>(&markAsUnreachable));
}
/// Marks @p block as known to not return.
///
/// Marks the terminator as known to be unreachable, if any. Returns @c true if
/// any IR was modified.
///
/// @pre    `op`
inline bool markAsNoReturn(Block* block)
{
    return isTerminator(&block->back()) && markAsUnreachable(&block->back());
}

} // namespace mlir::ub
