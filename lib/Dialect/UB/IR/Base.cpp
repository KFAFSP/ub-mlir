/// Implements the UB dialect base.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Dialect/UB/IR/Base.h"

#include "mlir/IR/PatternMatch.h"
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
        .Case(
            [&](NeverAttr) { return builder.create<NeverOp>(location, type); })
        .Default(static_cast<Operation*>(nullptr));
}

namespace {

struct UnreachableIsNoReturn : RewritePattern {
    static constexpr StringRef generatedNames[] = {"ub.never"};

    UnreachableIsNoReturn(MLIRContext* ctx)
            : RewritePattern(MatchAnyOpTypeTag{}, 1, ctx, generatedNames)
    {}

    virtual LogicalResult
    matchAndRewrite(Operation* op, PatternRewriter &rewriter) const override
    {
        const auto isConstant = [](Value value) {
            if (auto def = value.getDefiningOp())
                return def->hasTrait<OpTrait::ConstantLike>();
            return false;
        };
        if (!isKnownUnreachable(op)
            || !llvm::all_of(op->getOperands(), isConstant))
            return failure();

        if (op->getNumResults() == 0) {
            if (&op->getBlock()->back() == op) {
                // Operation is potentially a terminator, so just optimistically
                // mark it as unreachable.
                return success(markAsUnreachable(op));
            }

            // Operation is never scheduled.
            rewriter.eraseOp(op);
            return success();
        }

        // Attempt folding the operation away.
        SmallVector<Value> foldResults;
        if (succeeded(rewriter.tryFold(op, foldResults))) {
            rewriter.replaceOp(op, foldResults);
            return success();
        }

        // Replace the operation with never values.
        rewriter.replaceOp(
            op,
            llvm::to_vector(
                llvm::map_range(op->getResultTypes(), [&](Type type) {
                    return rewriter.create<NeverOp>(op->getLoc(), type)
                        .getResult();
                })));
        return success();
    }
};

} // namespace

void UBDialect::getCanonicalizationPatterns(RewritePatternSet &patterns) const
{
    patterns.add<UnreachableIsNoReturn>(patterns.getContext());
}

LogicalResult
UBDialect::verifyOperationAttribute(Operation* op, NamedAttribute attr)
{
    if (attr.getName() == kUnreachableAttrName) {
        if (!llvm::isa<ControlFlowTerminator>(op))
            return op->emitError()
                   << "attribute '" << kUnreachableAttrName
                   << "' is only applicable to control flow terminators";

        return success();
    }

    return op->emitError()
           << "attribute '" << attr.getName().getValue()
           << "' not supported as an op attribute by the ub dialect";
}

void UBDialect::initialize()
{
    registerAttributes();
    registerOps();
    registerTypes();

    // Implement the inliner interface.
    addInterfaces<UBInlinerInterface>();

    // Registers interface models for built-ins.
    registerControlFlowTerminatorOpInterfaceModels(*getContext());
}
