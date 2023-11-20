/// Declares the conversion passes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "ub-mlir/Conversion/UBXToLLVM/UBXToLLVM.h"

namespace ub_mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "ub-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace ub_mlir
