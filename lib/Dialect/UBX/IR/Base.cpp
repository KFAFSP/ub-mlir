/// Implements the UBX dialect base.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Dialect/UBX/IR/Base.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/InliningUtils.h"
#include "ub-mlir/Dialect/UBX/IR/UBX.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ubx;

//===- Generated implementation -------------------------------------------===//

#include "ub-mlir/Dialect/UBX/IR/Base.cpp.inc"

//===----------------------------------------------------------------------===//

namespace {

struct UBXInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final
    {
        return true;
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// UBXDialect
//===----------------------------------------------------------------------===//

Operation *UBXDialect::materializeConstant(
    OpBuilder &builder,
    Attribute value,
    Type type,
    Location location)
{
    return llvm::TypeSwitch<Attribute, Operation *>(value)
        .Case([&](PoisonAttr attr) -> Operation * {
            if (attr.getType() != type) return nullptr;

            // Fully-poisoned values are always materialized by our dialect.
            return builder.create<PoisonOp>(location, attr);
        })
        .Case([&](PoisonedElementsAttr attr) -> Operation * {
            if (attr.getType() != type) return nullptr;

            // If the value is not poisoned, materialize natively.
            if (!attr.isPoisoned())
                return attr.getSourceDialect()->materializeConstant(
                    builder,
                    attr.getElements(),
                    type,
                    location);

            // Otherwise, if the source dialect supports it, it may re-evaluate
            // the poisoned representation. If it fails, the current producing
            // op will remain as it is, which is fine.
            return attr.getSourceDialect()
                ->materializeConstant(builder, attr, type, location);
        })
        .Default(static_cast<Operation *>(nullptr));
}

void UBXDialect::initialize()
{
    registerAttributes();
    registerOps();
    registerTypes();

    // Implement the inliner interface.
    addInterfaces<UBXInlinerInterface>();
}
