/// Implements the UB dialect base.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Dialect/UB/IR/Base.h"

#include "mlir/Transforms/InliningUtils.h"
#include "ub-mlir/Dialect/UB/IR/UB.h"

using namespace mlir;
using namespace mlir::ub;

//===- Generated implementation -------------------------------------------===//

#include "ub-mlir/Dialect/UB/IR/Base.cpp.inc"

//===----------------------------------------------------------------------===//

namespace {

struct UBInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(Operation*, Region*, bool, IRMapping &) const final
    {
        return true;
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// UBDialect
//===----------------------------------------------------------------------===//

Operation* UBDialect::materializeConstant(
    OpBuilder &builder,
    Attribute value,
    Type type,
    Location location)
{
    // Only handles the PoisonAttr.
    const auto poisonAttr = value.dyn_cast<PoisonAttr>();
    if (!poisonAttr || poisonAttr.getType() != type) return nullptr;

    // If the value is not poisoned, materialize natively.
    if (!poisonAttr.isPoisoned())
        return poisonAttr.getSourceDialect()->materializeConstant(
            builder,
            poisonAttr.getSourceAttr(),
            type,
            location);

    // Otherwise, if the source dialect supports it, it may re-evaluate the
    // poisoned representation. If it fails, the current producing op will
    // remain as it is, which is fine.
    return poisonAttr.getSourceDialect()
        ->materializeConstant(builder, poisonAttr, type, location);
}

void UBDialect::initialize()
{
    registerAttributes();
    registerOps();

    // Implement the inliner interface.
    addInterfaces<UBInlinerInterface>();
}
