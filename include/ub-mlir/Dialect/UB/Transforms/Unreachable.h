/// Declares the unreachability transform helpers.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "ub-mlir/Dialect/UB/Analysis/Unreachable.h"

namespace mlir::ub {

//===----------------------------------------------------------------------===//
// Optimistic unreachability constraints
//===----------------------------------------------------------------------===//
//
// These functions apply the propositions `unreachable` and `noreturn` to the
// IR in a way that the pessimistic tests will detect them. The resulting IR is
// not canonicalized, and transitive reachability is not marked / detected.
//

/// Marks @p value as known to be unreachable.
///
/// Replaces all uses of @p value with never values materialized after its
/// definition. Returns @c true if any IR was modified.
///
/// @pre    `value`
bool markAsUnreachable(RewriterBase &rewriter, Value value);

/// Replaces all operand uses of @p op with unreachable values.
///
/// Returns @c true if any IR was modified.
///
/// @pre    `op`
bool makeOperandsUnreachable(RewriterBase &rewriter, Operation* op);

/// Replaces all results of @p op with unreachable values.
///
/// Returns @c true if any IR was modified.
///
/// @pre    `op`
bool makeResultsUnreachable(RewriterBase &rewriter, Operation* op);

/// Marks @p term as known to be unreachable.
///
/// If @p term is a non-special, non-return-like SSACFG terminator, it is
/// replaced with an UnreachableTerminator. Otherwise, it is marked as un-
/// reachable in place.
///
/// @pre    `term`
bool markAsUnreachable(RewriterBase &rewriter, ControlFlowTerminator term);

/// Marks @p op as known to be unreachable.
///
/// Replaces all operands of @p op with never values materialized directly
/// before @p op . If @p op is a ControlFlowTerminator, makes it unreachable.
/// Returns @c true if any IR was modified.
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
inline bool markAsUnreachable(RewriterBase &rewriter, Operation* op)
{
    return llvm::TypeSwitch<Operation*, bool>(op)
        .Case([&](ControlFlowTerminator term) {
            return markAsUnreachable(rewriter, term);
        })
        .Default([&](Operation* op) {
            return makeOperandsUnreachable(rewriter, op);
        });
}

/// Marks @p block as known to be unreachable, starting with @p pos .
///
/// If @p block is not an SSACFG block, returns @c false .
///
/// Otherwise, splits @p block at @p pos , and terminates @p block using an
/// UnreachableTerminator operation.
///
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
bool markAsUnreachable(
    RewriterBase &rewriter,
    Block* block,
    Block::iterator pos);

/// Marks @p block as known to be unreachable.
///
/// Starts unreachability at the entry of @p block . See
/// markAsUnreachable(Block, Block::iterator) for more details.
///
/// @pre    `block`
inline bool markAsUnreachable(RewriterBase &rewriter, Block* block)
{
    return markAsUnreachable(rewriter, block, block->begin());
}

/// Marks @p op as known to not return.
///
/// Marks all results of @p op as known to be unreachable. Returns @c true if
/// any IR was modified.
///
/// @pre    `op`
inline bool markAsNoReturn(RewriterBase &rewriter, Operation* op)
{
    return llvm::count_if(
        op->getResults(),
        [&](Value result) { return markAsUnreachable(rewriter, result); });
}

/// Marks @p block as known to not return.
///
/// Marks the terminator of @p block as known to be unreachable, if any. Returns
/// @c true if any IR was modified.
///
/// @pre    `block`
inline bool markAsNoReturn(RewriterBase &rewriter, Block* block)
{
    if (block->empty()) return false;
    if (auto term = llvm::dyn_cast<ControlFlowTerminator>(&block->back()))
        return markAsUnreachable(rewriter, term);
    return false;
}

/// Marks @p region as known to not return.
///
/// Marks all RegionBranchTerminatorOpInterface ops directly nested within
/// @p region as known to not return. Returns @c true if any IR was modified.
///
/// @pre    `region`
inline bool markAsNoReturn(RewriterBase &rewriter, Region* region)
{
    return llvm::count_if(
        region->getOps<RegionBranchTerminatorOpInterface>(),
        [&](RegionBranchTerminatorOpInterface term) {
            return markAsUnreachable(rewriter, ControlFlowTerminator(term));
        });
}

} // namespace mlir::ub
