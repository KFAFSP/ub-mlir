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
