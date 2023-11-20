/// Declares the UBX passes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::ubx {

/// Enumeration that controls propagation of never values through ops.
enum class OpNeverPropagation {
    /// Never propagate through ops.
    Never,
    /// Propagate through constant expressions that are unfoldable.
    Constexpr,
    /// Propagate through pure ops.
    Pure,
    /// Always propagate through ops.
    Always
};

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL
#include "ub-mlir/Dialect/UBX/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

/// Adds the ubx-freeze pass patterns to @p patterns .
void populateFreezePatterns(RewritePatternSet &patterns);

/// Constructs the ubx-freeze pass.
std::unique_ptr<Pass> createFreezePass();

/// Adds the ubx-propagate-never patterns to @p patterns .
void populatePropagateNeverPatterns(
    RewritePatternSet &patterns,
    OpNeverPropagation throughOps = OpNeverPropagation::Constexpr);

/// Constructs the ubx-propagate-never pass.
std::unique_ptr<Pass> createPropagateNeverPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "ub-mlir/Dialect/UBX/Transforms/Passes.h.inc"

} // namespace mlir::ubx
