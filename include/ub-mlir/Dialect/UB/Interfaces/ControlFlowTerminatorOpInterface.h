/// Declares the UB ControlFlowTerminatorOpInterface interface.
///
/// The ControlFlowTerminatorOpInterface allows operation that are terminators
/// in well-defined control flow regions to participate in reachability
/// analysis, even if they do not implement a standard control flow interface.
/// Additionally, it allows the implementing operations to customize their
/// unreachable representation.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir::ub::control_flow_terminator_op_interface_defaults {

/// Determines whether @p op is known to be unreachable.
///
/// This default implementation evaluates the following criteria:
///     - Is the `ub.unreachable` discardable attribute set?
///     - Are any of the operands unreachable?
///
/// @pre    `llvm::isa<ControlFlowTerminatorOpInterface>(op)`
[[nodiscard]] bool isKnownUnreachable(Operation* op);

/// Marks @p op as known to be unreachable.
///
/// This default implementation replaces @p op with an UnreachableOp if:
///     - It is within an SSACFG block.
///     - It is does not implement `RegionBranchTerminatorOpInterface`.
///
/// Otherwise, attaches the `ub.unreachable` discardable attribute.
///
/// @pre    `llvm::isa<ControlFlowTerminatorOpInterface>(op)`
bool markAsUnreachable(Operation* op);

} // namespace mlir::ub::control_flow_terminator_op_interface_defaults

//===- Generated includes -------------------------------------------------===//

#include "ub-mlir/Dialect/UB/Interfaces/ControlFlowTerminatorOpInterface.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ub {

/// Concept for an operation that is a well-defined control flow terminator.
///
/// This concept is satisfied by:
///
///     - Implementations of BranchOpInterface.
///     - Implementations of RegionBranchTerminatorOpInterface.
///     - Implementations of ControlFlowTerminatorOpInterface.
///
/// This concept indicates the the given terminator passes control flow to some
/// successor(s), leaving the current block. Therefore, it can be part of
/// reachability analysis, and may be marked as unreachable.
class ControlFlowTerminator : OpState {
public:
    /// @copydoc classof(Operation*)
    static bool classof(BranchOpInterface) { return true; }
    /// @copydoc classof(Operation*)
    static bool classof(RegionBranchTerminatorOpInterface) { return true; }
    /// @copydoc classof(Operation*)
    static bool classof(ControlFlowTerminatorOpInterface) { return true; }
    /// Determines whether @p op is a ControlFlowTerminator.
    ///
    /// @pre    `op`
    static bool classof(Operation* op)
    {
        return llvm::TypeSwitch<Operation*, bool>(op)
            .Case([](BranchOpInterface) { return true; })
            .Case([](RegionBranchTerminatorOpInterface) { return true; })
            .Case([](ControlFlowTerminatorOpInterface) { return true; })
            .Default([](auto) { return false; });
    }

    /// Initializes a null ControlFlowTerminator.
    explicit ControlFlowTerminator() : OpState(nullptr) {}
    /// Initializes a null ControlFlowTerminator.
    /*implicit*/ ControlFlowTerminator(std::nullptr_t) : OpState(nullptr) {}
    /// Initializes a ControlFlowTerminator from @p op.
    ///
    /// @pre    `llvm::isa<ControlFlowTerminator>(op)`
    explicit ControlFlowTerminator(Operation* op) : OpState(op) {}

    /// Initializes a ControlFlowTerminator from @p op .
    /*implicit*/ ControlFlowTerminator(BranchOpInterface op) : OpState(op) {}
    /// @copydoc ControlFlowTerminator(BranchOpInterface)
    /*implicit*/ ControlFlowTerminator(RegionBranchTerminatorOpInterface op)
            : OpState(op)
    {}
    /// @copydoc ControlFlowTerminator(BranchOpInterface)
    /*implicit*/ ControlFlowTerminator(ControlFlowTerminatorOpInterface op)
            : OpState(op)
    {}

    /*implicit*/ operator bool() { return OpState::operator bool(); }

    /// Determines whether this terminator is known to be unreachable.
    [[nodiscard]] bool isKnownUnreachable()
    {
        return llvm::TypeSwitch<Operation*, bool>(*this)
            .Case([](ControlFlowTerminatorOpInterface iface) {
                return iface.isKnownUnreachable();
            })
            .Default([](Operation* op) {
                return control_flow_terminator_op_interface_defaults::
                    isKnownUnreachable(op);
            });
    }

    /// Marks this terminator as known to be unreachable.
    ///
    /// Returns @c true if any IR was modified.
    bool markAsUnreachable()
    {
        return llvm::TypeSwitch<Operation*, bool>(*this)
            .Case([](ControlFlowTerminatorOpInterface iface) {
                return iface.markAsUnreachable();
            })
            .Default([](Operation* op) {
                return control_flow_terminator_op_interface_defaults::
                    markAsUnreachable(op);
            });
    }
};

/// Implements the ControlFlowTerminatorOpInterface interface for built-in ops.
///
/// Registers external models of the ControlFlowTerminatorOpInterface for:
///
///     - llvm.unreachable
void registerControlFlowTerminatorOpInterfaceModels(MLIRContext &ctx);

} // namespace mlir::ub
