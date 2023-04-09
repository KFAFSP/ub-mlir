/// Implements the unreachability analysis.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Dialect/UB/Analysis/Unreachable.h"

#include "mlir/Interfaces/CallInterfaces.h"

#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::ub;

SmallVector<Attribute> getConstantOperands(Operation* op)
{
    return llvm::to_vector(llvm::map_range(
        op->getOperands(),
        [](Value operand) -> Attribute {
            auto res = llvm::dyn_cast<OpResult>(operand);
            if (!res || !res.getOwner()->hasTrait<OpTrait::ConstantLike>())
                return {};
            SmallVector<OpFoldResult> folded;
            const auto ok = res.getOwner()->fold({}, folded);
            assert(succeeded(ok));
            assert(folded[res.getResultNumber()].dyn_cast<Attribute>());
            return folded[res.getResultNumber()].dyn_cast<Attribute>();
        }));
}

static bool isReachable(
    UnreachabilityAnalysis &analysis,
    RegionBranchOpInterface op,
    Region* region)
{
    SmallVector<Region*> workList;
    llvm::SmallPtrSet<Region*, 8> reachable;
    const auto visit = [&](Region* predecessor, Region* successor) {
        if (predecessor && analysis.isNoReturn(predecessor)) return false;
        if (successor == region) return true;
        if (successor && reachable.insert(successor).second)
            workList.push_back(successor);
        return false;
    };

    // Initialize work list with entry regions.
    {
        const auto constOperands = getConstantOperands(op);
        SmallVector<RegionSuccessor> entrys;
        op.getSuccessorRegions(std::nullopt, constOperands, entrys);
        if (llvm::any_of(entrys, [&](const RegionSuccessor &entry) {
                return visit(nullptr, entry.getSuccessor());
            }))
            return true;
    }

    // Keep visiting reachable regions.
    while (!workList.empty()) {
        const auto current = workList.pop_back_val();
        SmallVector<RegionSuccessor> successors;
        op.getSuccessorRegions(current->getRegionNumber(), successors);
        for (const auto &successor : successors)
            if (visit(current, successor.getSuccessor())) return true;
    }

    // Target region was not found.
    return false;
}

//===----------------------------------------------------------------------===//
// UnreachabilityAnalysis
//===----------------------------------------------------------------------===//

bool UnreachabilityAnalysis::computeUnreachable(Operation* op)
{
    assert(!isKnownUnreachable(op));

    // An operation is unreachable if any operand is unreachable.
    if (llvm::any_of(op->getOperands(), [&](Value operand) {
            return isUnreachable(operand);
        }))
        return true;

    auto block = op->getBlock();

    // We can't reason about top-level ops like ModuleOp.
    if (!block) return false;
    // An operation in an unreachable block is unreachable.
    if (isUnreachable(block)) return true;
    // We can't apply control-flow reasoning inside non-SSACFG blocks.
    if (!isSSACFG(block)) return false;

    // An operation is unreachable if any of its predecessors are noreturn.
    if (std::any_of(
            Block::reverse_iterator(op),
            block->rend(),
            [&](Operation &op) { return isNoReturn(&op); }))
        return true;

    // Probably reachable.
    return false;
}

bool UnreachabilityAnalysis::computeUnreachable(Block* block)
{
    assert(!isKnownUnreachable(block));

    // A non-entry block is unreachable if all its predecessors are unreachable.
    if (!block->isEntryBlock()
        && llvm::all_of(block->getPredecessors(), [&](Block* block) {
               return isUnreachable(block);
           }))
        return true;

    // An entry block is unreachable if its region is unreachable.
    if (block->isEntryBlock() && isUnreachable(block->getParent())) return true;

    // Probably reachable.
    return false;
}

bool UnreachabilityAnalysis::computeUnreachable(Region* region)
{
    // We can only reason about regions inside RegionBranchOp ops.
    auto iface = llvm::dyn_cast<RegionBranchOpInterface>(region->getParentOp());
    if (!iface) return false;

    // A region is unreachable if the parent is unreachable.
    if (isUnreachable(iface)) return true;

    // A region is unreachable if all of its predecessors are noreturn.
    if (!isReachable(*this, iface, region)) return true;

    // Probably reachable.
    return false;
}

bool UnreachabilityAnalysis::computeNoReturn(Operation* op)
{
    assert(!isKnownNoReturn(op));

    return llvm::TypeSwitch<Operation*, bool>(op)
        .Case([&](RegionBranchOpInterface op) {
            // A RegionBranchOp op is noreturn if it is not reachable.
            return !isReachable(*this, op, nullptr);
        })
        .Case([&](CallableOpInterface op) {
            // Callable ops are noreturn if all their returns are unreachable.
            for (auto &op : op.getCallableRegion()->getOps())
                if (op.hasTrait<OpTrait::ReturnLike>() && !isUnreachable(&op))
                    return false;
            return true;
        })
        .Case([&](CallOpInterface op) {
            // Call ops are noreturn if their callable is noreturn.
            if (auto callable = op.resolveCallable())
                return isNoReturn(callable);
            return false;
        })
        .Default([](auto) { return false; });
}

bool UnreachabilityAnalysis::computeNoReturn(Block* block)
{
    assert(!isKnownNoReturn(block));

    // We can't reason about empty blocks.
    if (block->empty()) return false;

    // A block with an unreachable control flow terminator is noreturn.
    if (auto term = llvm::dyn_cast<ControlFlowTerminator>(&block->back())) {
        if (isUnreachable(term)) return true;
    }

    // We can't apply control-flow reasoning inside non-SSACFG blocks.
    if (!isSSACFG(block)) return false;

    // A block is noreturn if any operation in it is noreturn.
    if (llvm::any_of(block->getOperations(), [&](Operation &op) {
            return isNoReturn(&op);
        }))
        return true;

    // Probably reachable.
    return false;
}

bool UnreachabilityAnalysis::computeNoReturn(Region* region)
{
    assert(!isKnownNoReturn(region));

    // Inside RegionBranchOp ops, regions do not return if all their
    // RegionBranchTerminatorOp ops are unreachable.
    if (auto iface =
            llvm::dyn_cast<RegionBranchOpInterface>(region->getParentOp())) {
        if (llvm::all_of(
                region->getOps<RegionBranchTerminatorOpInterface>(),
                [&](Operation* op) { return isUnreachable(op); }))
            return true;
    }

    // Probably reachable.
    return false;
}
