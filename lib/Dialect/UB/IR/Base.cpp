/// Implements the UB dialect base.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Dialect/UB/IR/Base.h"

#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Transforms/InliningUtils.h"
#include "ub-mlir/Dialect/UB/IR/UB.h"

#include "llvm/ADT/TypeSwitch.h"

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
    return llvm::TypeSwitch<Attribute, Operation*>(value)
        .Case([&](PoisonAttr attr) -> Operation* {
            if (attr.getType() != type) return nullptr;

            // If the value is not poisoned, materialize natively.
            if (!attr.isPoisoned())
                return attr.getSourceDialect()->materializeConstant(
                    builder,
                    attr.getSourceAttr(),
                    type,
                    location);

            // Otherwise, if the source dialect supports it, it may re-evaluate
            // the poisoned representation. If it fails, the current producing
            // op will remain as it is, which is fine.
            return attr.getSourceDialect()
                ->materializeConstant(builder, attr, type, location);
        })
        .Case([&](NeverAttr) { return builder.create<NeverOp>(location); })
        .Default(static_cast<Operation*>(nullptr));
}

LogicalResult
UBDialect::verifyOperationAttribute(Operation* op, NamedAttribute attr)
{
    if (attr.getName() == kUnreachableAttrName) {
        if (!op->hasTrait<OpTrait::IsTerminator>())
            return op->emitError() << "attribute '" << kUnreachableAttrName
                                   << "' is only applicable to terminators";

        // NOTE: An operation that does not implement RegionKindInterface is
        //       assumed to have only SSACFG regions per MLIR core!
        if (auto regionKindIface =
                llvm::dyn_cast_if_present<RegionKindInterface>(
                    op->getParentOp())) {
            const auto regionIdx = op->getParentRegion()->getRegionNumber();
            if (regionKindIface.getRegionKind(regionIdx) != RegionKind::SSACFG)
                return op->emitError()
                       << "attribute '" << kUnreachableAttrName
                       << "' is only applicable in SSACFG regions";
        }

        return success();
    }

    return op->emitError()
           << "attribute '" << attr.getName()
           << "' not supported as an op attribute by the ub dialect";
}

void UBDialect::initialize()
{
    registerAttributes();
    registerOps();
    registerTypes();

    // Implement the inliner interface.
    addInterfaces<UBInlinerInterface>();
}
