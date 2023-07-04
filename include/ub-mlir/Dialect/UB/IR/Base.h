/// Declaration of the UB dialect base.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include <string_view>

//===- Generated includes -------------------------------------------------===//

#include "ub-mlir/Dialect/UB/IR/Base.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ub {

/// Concept for an MLIR Type constraint.
template<class T>
concept TypeConstraint = std::is_base_of_v<Type, T>;

/// Concept for an MLIR Attribute constraint.
template<class T>
concept AttrConstraint = std::is_base_of_v<Attribute, T>;

} // namespace mlir::ub
