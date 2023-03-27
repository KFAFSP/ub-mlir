/// Declaration of the UB dialect attributes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/TypeUtilities.h"
#include "ub-mlir/Dialect/UB/IR/Base.h"

#include "llvm/ADT/APInt.h"

#include <algorithm>
#include <optional>

namespace mlir::ub {

/// Reference to a loaded dialect.
using DialectRef = Dialect*;

} // namespace mlir::ub

//===- Generated includes -------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "ub-mlir/Dialect/UB/IR/Attributes.h.inc"

//===----------------------------------------------------------------------===//
