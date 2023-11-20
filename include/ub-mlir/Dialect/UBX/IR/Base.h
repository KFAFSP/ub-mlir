/// Declaration of the UBX dialect base.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include <string_view>

namespace mlir::ubx {

/// Concept for an MLIR Type constraint.
template<class T>
concept TypeConstraint = std::is_base_of_v<Type, T>;

/// Concept for an MLIR Attribute constraint.
template<class T>
concept AttrConstraint = std::is_base_of_v<Attribute, T>;

} // namespace mlir::ubx

//===- Generated includes -------------------------------------------------===//

#include "ub-mlir/Dialect/UBX/IR/Base.h.inc"

//===----------------------------------------------------------------------===//
