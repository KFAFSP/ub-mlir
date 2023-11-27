/// Declares C++ concepts for use with MLIR.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

#include <type_traits>

namespace mlir::ext {

//===----------------------------------------------------------------------===//
// IR constraints
//===----------------------------------------------------------------------===//

/// Concept for an MLIR Type constraint.
template<class T>
concept TypeConstraint = std::is_base_of_v<Type, T>;
/// Concept for an MLIR Attribute constraint.
template<class T>
concept AttrConstraint = std::is_base_of_v<Attribute, T>;
/// Concept for an MLIR Op constraint.
template<class T>
concept OpConstraint = std::is_base_of_v<OpState, T>;
/// Concept for an MLIR Value constraint.
template<class T>
concept ValueConstraint = std::is_base_of_v<Value, T>;

} // namespace mlir::ext
