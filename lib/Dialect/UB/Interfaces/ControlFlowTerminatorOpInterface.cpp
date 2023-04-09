/// Implements the UB dialect ControlFlowTerminatorOpInterface interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Dialect/UB/Interfaces/ControlFlowTerminatorOpInterface.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "ub-mlir/Dialect/UB/IR/UB.h"

using namespace mlir;
using namespace mlir::ub;

//===----------------------------------------------------------------------===//
// SSACFG traits
//===----------------------------------------------------------------------===//

bool mlir::ub::isSSACFG(Region* region)
{
    // NOTE: An operation that does not implement RegionKindInterface is
    //       assumed to have only SSACFG regions per MLIR core!
    auto iface =
        llvm::dyn_cast_if_present<RegionKindInterface>(region->getParentOp());
    if (!iface) return true;

    return iface.getRegionKind(region->getRegionNumber()) == RegionKind::SSACFG;
}

//===----------------------------------------------------------------------===//
// Interface defaults
//===----------------------------------------------------------------------===//

bool mlir::ub::control_flow_terminator_op_interface_defaults::
    isKnownUnreachable(Operation* op)
{
    return op->hasAttr(kUnreachableAttrName)
           || llvm::any_of(
               op->getOperands(),
               [](Value value) { return llvm::isa<UnreachableValue>(value); });
}

bool mlir::ub::control_flow_terminator_op_interface_defaults::markAsUnreachable(
    Operation* op)
{
    if (op->hasAttr(kUnreachableAttrName)) return false;
    op->setAttr(kUnreachableAttrName, UnitAttr::get(op->getContext()));
    return true;
}

//===- Generated implementation -------------------------------------------===//

#include "ub-mlir/Dialect/UB/Interfaces/ControlFlowTerminatorOpInterface.cpp.inc"

//===----------------------------------------------------------------------===//

namespace {

struct UnreachableOpModel
        : ControlFlowTerminatorOpInterface::
              ExternalModel<UnreachableOpModel, LLVM::UnreachableOp> {
    bool isKnownUnreachable(Operation*) const { return true; }
    bool markAsUnreachable(Operation*) const { return false; }
};

} // namespace

void mlir::ub::registerControlFlowTerminatorOpInterfaceModels(MLIRContext &ctx)
{
    LLVM::UnreachableOp::attachInterface<UnreachableOpModel>(ctx);
}
