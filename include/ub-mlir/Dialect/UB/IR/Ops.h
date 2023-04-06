/// Declaration of the UB dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "ub-mlir/Dialect/UB/IR/Types.h"

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "ub-mlir/Dialect/UB/IR/Ops.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ub {

//===----------------------------------------------------------------------===//
// Unreachable attribute
//===----------------------------------------------------------------------===//

/// Determines whether @p op is known to be unreachable.
///
/// @pre    `op`
[[nodiscard]] inline bool isKnownUnreachable(Operation* op)
{
    return op->hasAttr(kUnreachableAttrName);
}

/// Tries to mark @p op as a known unreachable terminator.
///
/// Returns @c false if @p op is not a terminator or not in an SSACFG region.
///
/// @pre    `op`
inline bool markAsUnreachable(Operation* op)
{
    if (!isSSACFGTerminator(op)) return false;
    op->setAttr(kUnreachableAttrName, UnitAttr::get(op->getContext()));
    return true;
}

} // namespace mlir::ub
