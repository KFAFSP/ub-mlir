/// Implements the reachability analysis.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Dialect/UB/Analysis/ReachabilityAnalysis.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ub;

//===----------------------------------------------------------------------===//
// Optimistic reachability constraints
//===----------------------------------------------------------------------===//

bool mlir::ub::markAsUnreachable(Value value)
{
    // Do not modify IR if already marked.
    if (isKnownUnreachable(value)) return false;

    // Find the earliest possible insertion point.
    OpBuilder builder(value.getContext());
    const auto loc =
        llvm::TypeSwitch<Value, std::optional<Location>>(value)
            .Case([&](OpResult result) {
                builder.setInsertionPointAfter(result.getOwner());
                return result.getOwner()->getLoc();
            })
            .Case([&](BlockArgument arg) {
                builder.setInsertionPointToStart(arg.getParentBlock());
                return arg.getLoc();
            })
            .Default(
                [](auto) -> std::optional<Location> { return std::nullopt; });
    if (!loc) return false;

    // Replace all uses with the earliest-possible materialized never value.
    value.replaceAllUsesWith(
        builder.create<NeverOp>(*loc, value.getType()).getResult());
    return true;
}

bool mlir::ub::markAsUnreachable(Operation* op)
{
    // Replace all operands with never values.
    // NOTE: We could use an OperationFolder here too, but that seems overkill
    //       considering users are expected to keep canonicalizing.
    OpBuilder builder(op);
    const bool modified =
        llvm::count_if(op->getOpOperands(), [&](OpOperand &operand) {
            if (isKnownUnreachable(operand.get())) return false;
            operand.set(
                builder.create<NeverOp>(op->getLoc(), operand.get().getType())
                    .getResult());
            return true;
        });

    return modified
           | llvm::dyn_cast<ControlFlowTerminator>(op).markAsUnreachable();
}

bool mlir::ub::markAsUnreachable(Block* block, Block::iterator pos)
{
    std::optional<Location> maybeLoc;
    if (!block->empty()) {
        // Try to avoid splitting by just marking the terminator.
        const auto lastPos =
            pos == block->end() ? Block::iterator(&block->back()) : pos;
        if (auto iface = llvm::dyn_cast<ControlFlowTerminator>(&*lastPos))
            return iface.markAsUnreachable();

        // Split the block before pos.
        auto unreachableBlock = block->splitBlock(pos);
        if (!unreachableBlock->empty())
            maybeLoc = unreachableBlock->front().getLoc();
    }

    if (!maybeLoc && !block->empty()) maybeLoc = block->back().getLoc();

    OpBuilder builder(block, block->end());
    builder.setInsertionPointToEnd(block);
    builder.create<UnreachableOp>(maybeLoc.value_or(builder.getUnknownLoc()));
    return true;
}
