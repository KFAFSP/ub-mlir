/// Declares the UBX passes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::ubx {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL
#include "ub-mlir/Dialect/UBX/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

/// Adds the ubx-freeze pass patterns to @p patterns .
void populateFreezePatterns(RewritePatternSet &patterns);

/// Constructs the ubx-freeze pass.
std::unique_ptr<Pass> createFreezePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "ub-mlir/Dialect/UBX/Transforms/Passes.h.inc"

} // namespace mlir::ubx
