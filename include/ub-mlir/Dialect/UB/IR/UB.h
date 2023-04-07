/// Convenience include for the UB dialect.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "ub-mlir/Dialect/UB/Analysis/ReachabilityAnalysis.h"
#include "ub-mlir/Dialect/UB/IR/Ops.h"

namespace mlir::ub {

//===----------------------------------------------------------------------===//
// isPoison
//===----------------------------------------------------------------------===//

/// Determines whether @p attr is guaranteed poison.
///
/// @pre    `attr`
[[nodiscard]] inline bool isPoison(PoisonAttr attr) { return attr.isPoison(); }
/// Determines whether @p attr is guaranteed poison.
///
/// @pre    `attr`
[[nodiscard]] inline bool isPoison(Attribute attr)
{
    if (const auto poisonAttr = llvm::dyn_cast<PoisonAttr>(attr))
        return isPoison(poisonAttr);
    return false;
}
/// Determines whether @p result is guaranteed poison.
///
/// @pre    `result`
[[nodiscard]] bool isPoison(OpResult result);
/// Determines whether @p value is guaranteed poison.
///
/// @pre    `value`
[[nodiscard]] inline bool isPoison(Value value)
{
    if (const auto result = llvm::dyn_cast<OpResult>(value))
        return isPoison(result);
    return false;
}
/// Determines whether @p value is guaranteed poison.
///
/// @pre    `value`
[[nodiscard]] inline bool isPoison(OpFoldResult value)
{
    if (const auto attr = value.dyn_cast<Attribute>()) return isPoison(attr);
    return false;
}

} // namespace mlir::ub
