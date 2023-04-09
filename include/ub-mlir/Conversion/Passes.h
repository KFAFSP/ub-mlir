/// Declares the conversion passes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "ub-mlir/Conversion/UBToLLVM/UBToLLVM.h"

namespace mlir::ub {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "ub-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir::ub
