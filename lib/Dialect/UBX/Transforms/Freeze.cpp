/// Implements the FreezePass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ub-mlir/Dialect/UBX/IR/UBX.h"

using namespace mlir;
using namespace mlir::ubx;

//===- Generated includes -------------------------------------------------===//

namespace mlir::ubx {

#define GEN_PASS_DEF_FREEZE
#include "ub-mlir/Dialect/UBX/Transforms/Passes.h.inc"

} // namespace mlir::ubx

//===----------------------------------------------------------------------===//

namespace {

struct Freeze : OpRewritePattern<FreezeOp> {
    using OpRewritePattern::OpRewritePattern;

    virtual LogicalResult
    matchAndRewrite(FreezeOp op, PatternRewriter &rewriter) const override
    {
        // Try to match a constant operand.
        Attribute hint = {};
        if (matchPattern(op.getOperand(), m_Constant(&hint))) {
            // Do not attempt to remove run-time dependent freeze operations!
            // It is the responsibility of the poison-generating operation to
            // guarantee that a value will be produced (no trap occurs, ...).
            return failure();
        }

        // Try the attribute interface freeze method.
        if (auto iface = llvm::dyn_cast<PoisonAttrInterface>(hint)) {
            OpBuilder::InsertionGuard guard(rewriter);
            if (auto frozen = iface.freeze(rewriter, op.getLoc())) {
                rewriter.replaceOp(op, frozen);
                return success();
            }
        }

        // Try the type interface freeze method.
        if (auto iface = llvm::dyn_cast<FreezableTypeInterface>(op.getType())) {
            OpBuilder::InsertionGuard guard(rewriter);
            if (auto frozen = iface.freeze(rewriter, op.getLoc(), hint)) {
                rewriter.replaceOp(op, frozen);
                return success();
            }
        }

        // We have no known way of instantiating a frozen value.
        return failure();
    }
};

struct FreezePass : mlir::ubx::impl::FreezeBase<FreezePass> {
    using FreezeBase::FreezeBase;

    void runOnOperation() override;
};

} // namespace

void FreezePass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());

    populateFreezePatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(
            getOperation(),
            FrozenRewritePatternSet(std::move(patterns)))))
        return signalPassFailure();
}

void mlir::ubx::populateFreezePatterns(RewritePatternSet &patterns)
{
    patterns.add<Freeze>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::ubx::createFreezePass()
{
    return std::make_unique<FreezePass>();
}
