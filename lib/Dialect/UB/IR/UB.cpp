/// Implements UB dialect helper functions.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Dialect/UB/IR/UB.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::ub;

//===----------------------------------------------------------------------===//
// isPoison
//===----------------------------------------------------------------------===//

bool mlir::ub::isPoison(OpResult result)
{
    const auto def = result.getOwner();
    if (!def->hasTrait<OpTrait::ConstantLike>()) return false;
    SmallVector<OpFoldResult> results;
    if (failed(def->fold({}, results))) return false;
    return isPoison(results[result.getResultNumber()]);
}
