/// Implements the unreachability transform helpers.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Dialect/UB/Transforms/Unreachable.h"

using namespace mlir;
using namespace mlir::ub;

//===----------------------------------------------------------------------===//
// Optimistic unreachability constraints
//===----------------------------------------------------------------------===//

bool mlir::ub::markAsUnreachable(RewriterBase &rewriter, Value value)
{
    // Do not modify IR if already marked.
    if (isKnownUnreachable(value)) return false;

    OpBuilder::InsertionGuard guard(rewriter);

    // Find the earliest possible insertion point.
    const auto loc =
        llvm::TypeSwitch<Value, std::optional<Location>>(value)
            .Case([&](OpResult result) {
                rewriter.setInsertionPointAfter(result.getOwner());
                return result.getOwner()->getLoc();
            })
            .Case([&](BlockArgument arg) {
                rewriter.setInsertionPointToStart(arg.getParentBlock());
                return arg.getLoc();
            })
            .Default(
                [](auto) -> std::optional<Location> { return std::nullopt; });
    if (!loc) return false;

    // Replace all uses with the earliest-possible materialized never value.
    value.replaceAllUsesWith(
        rewriter.create<NeverOp>(*loc, value.getType()).getResult());
    return true;
}

bool mlir::ub::makeOperandsUnreachable(RewriterBase &rewriter, Operation* op)
{
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    return llvm::count_if(op->getOpOperands(), [&](OpOperand &operand) {
        if (isKnownUnreachable(operand.get())) return false;
        operand.set(
            rewriter.create<NeverOp>(op->getLoc(), operand.get().getType())
                .getResult());
        return true;
    });
}

bool mlir::ub::makeResultsUnreachable(RewriterBase &rewriter, Operation* op)
{
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);

    return llvm::count_if(op->getResults(), [&](Value result) {
        if (isKnownUnreachable(result)) return false;
        result.replaceAllUsesWith(
            rewriter.create<NeverOp>(op->getLoc(), result.getType())
                .getResult());
        return true;
    });
}

bool mlir::ub::markAsUnreachable(
    RewriterBase &rewriter,
    ControlFlowTerminator term)
{
    if (isSSACFG(term) && !isRegionReturnLike(term)) {
        // Non-special SSACFG terminators can be replaced entirely.
        rewriter.replaceOpWithNewOp<UnreachableTerminator>(term);
        return true;
    }

    // Ensure all operands are unreachable and the terminator is marked.
    auto modified = makeOperandsUnreachable(rewriter, term);
    modified |= term.markAsUnreachable();
    return modified;
}

bool mlir::ub::markAsUnreachable(
    RewriterBase &rewriter,
    Block* block,
    Block::iterator pos)
{
    std::optional<Location> maybeLoc;
    if (!block->empty()) {
        // Try to avoid splitting by just marking the terminator.
        const auto lastPos =
            pos == block->end() ? Block::iterator(&block->back()) : pos;
        if (auto term = llvm::dyn_cast<ControlFlowTerminator>(&*lastPos))
            return markAsUnreachable(rewriter, term);

        // Split the block before pos.
        auto unreachableBlock = rewriter.splitBlock(block, pos);
        if (!unreachableBlock->empty())
            maybeLoc = unreachableBlock->front().getLoc();
    }

    if (!maybeLoc && !block->empty()) maybeLoc = block->back().getLoc();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(block);
    rewriter.create<UnreachableTerminator>(
        maybeLoc.value_or(rewriter.getUnknownLoc()));
    return true;
}
