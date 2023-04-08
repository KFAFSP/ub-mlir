/// Declaration of the UB dialect base.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "ub-mlir/Dialect/UB/Interfaces/Interfaces.h"

#include <string_view>

//===- Generated includes -------------------------------------------------===//

#include "ub-mlir/Dialect/UB/IR/Base.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ub {

//===----------------------------------------------------------------------===//
// Unreachable attribute
//===----------------------------------------------------------------------===//

/// Name of the `ub.unreachable` discardable attribute.
static constexpr std::string_view kUnreachableAttrName = "ub.unreachable";

} // namespace mlir::ub
