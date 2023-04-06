/// Implements helpers for reachability analysis and propagation.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Dialect/UB/Analysis/Unreachable.h"

#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Transforms/FoldUtils.h"
#include "ub-mlir/Dialect/UB/IR/UB.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ub;

//===----------------------------------------------------------------------===//
// isSSACFG
//===----------------------------------------------------------------------===//

bool mlir::ub::isSSACFG(Region* region)
{
    // NOTE: An operation that does not implement RegionKindInterface is
    //       assumed to have only SSACFG regions per MLIR core!
    if (auto regionKindIface = llvm::dyn_cast_if_present<RegionKindInterface>(
            region->getParentOp())) {
        if (regionKindIface.getRegionKind(region->getRegionNumber())
            != RegionKind::SSACFG)
            return false;
    }

    return true;
}

//===----------------------------------------------------------------------===//
// markAsUnreachable
//===----------------------------------------------------------------------===//

bool mlir::ub::markAsUnreachable(Value value)
{
    if (isKnownUnreachable(value)) return false;

    OpBuilder builder(value.getContext());
    const auto loc =
        llvm::TypeSwitch<Value, Optional<Location>>(value)
            .Case([&](OpResult result) {
                builder.setInsertionPointAfter(result.getOwner());
                return result.getOwner()->getLoc();
            })
            .Case([&](BlockArgument arg) {
                builder.setInsertionPointToStart(arg.getParentBlock());
                return arg.getLoc();
            })
            .Default([](auto) -> Optional<Location> { return std::nullopt; });
    if (!loc) return false;

    value.replaceAllUsesWith(
        builder.create<NeverOp>(*loc, value.getType()).getResult());
    return true;
}

bool mlir::ub::markAsUnreachable(Operation* op)
{
    auto modified = false;

    // Replace all operands with never values.
    // NOTE: We could use an OperationFolder here too, but that seems overkill
    //       considering users are expected to keep canonicalizing.
    OpBuilder builder(op);
    op->setOperands(
        llvm::to_vector(llvm::map_range(op->getOperands(), [&](Value operand) {
            if (isKnownUnreachable(operand)) return operand;
            modified = true;
            return builder.create<NeverOp>(op->getLoc(), operand.getType())
                .getResult();
        })));

    // Set the `ub.unreachable` attribute if appropriate.
    if (isSSACFGTerminator(op)) {
        op->setAttr(kUnreachableAttrName, UnitAttr::get(op->getContext()));
        modified = true;
    }

    return modified;
}

bool mlir::ub::markAsUnreachable(Block* block, bool killLiveOuts)
{
    if (!isSSACFG(block)) return false;

    auto modified = false;

    if (!killLiveOuts) {
        // Ensure there are no live-out values.
        for (auto &op : *block) {
            if (!op.isUsedOutsideOfBlock(block)) continue;
            return false;
        }
    } else {
        const auto ctx = block->getParent()->getContext();
        const auto dialect = ctx->getLoadedDialect<UBDialect>();
        OpBuilder builder(block, block->begin());
        OperationFolder folder(ctx);
        const auto getNever = [&](Location loc, Type type) {
            return folder.getOrCreateConstant(
                builder,
                dialect,
                NeverAttr::get(type),
                type,
                loc);
        };

        // Kill all live-out values.
        for (auto &op : *block) {
            for (auto result : op.getResults()) {
                if (!result.isUsedOutsideOfBlock(block)) continue;
                modified = true;
                result.replaceAllUsesWith(
                    getNever(op.getLoc(), result.getType()));
            }
        }
    }

    // Erase everything but the terminator.
    while (block->getOperations().size() != 1) {
        modified = true;
        block->getOperations().front().dropAllReferences();
        block->getOperations().front().dropAllUses();
        block->getOperations().front().erase();
    }

    // Mark the terminator as unreachable.
    return modified | markAsUnreachable(block->getTerminator());
}
