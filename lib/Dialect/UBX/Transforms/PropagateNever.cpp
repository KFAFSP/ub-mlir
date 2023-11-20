/// Implements the PropagateNeverPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ub-mlir/Dialect/UBX/IR/UBX.h"

#include <cassert>

using namespace mlir;
using namespace mlir::ubx;

//===- Generated includes -------------------------------------------------===//

namespace mlir::ubx {

#define GEN_PASS_DEF_PROPAGATENEVER
#include "ub-mlir/Dialect/UBX/Transforms/Passes.h.inc"

} // namespace mlir::ubx

//===----------------------------------------------------------------------===//

[[nodiscard]] static bool isUnreachable(Operation *op)
{
    assert(op);

    for (auto operand : op->getOperands()) {
        NeverAttr never;
        if (matchPattern(operand, m_Constant(&never))) return true;
    }

    return false;
}

static void makeNoReturn(Operation *op, PatternRewriter &rewriter)
{
    assert(op);

    auto results =
        llvm::to_vector(llvm::map_range(op->getResultTypes(), [&](Type type) {
            return rewriter.create<NeverOp>(op->getLoc(), type).getResult();
        }));
    rewriter.replaceAllUsesWith(op->getResults(), results);
}

static LogicalResult completeFold(
    Operation *op,
    ArrayRef<OpFoldResult> results,
    PatternRewriter &rewriter)
{
    OpBuilder builder(op->getContext());
    SmallVector<Value> foldedResults;
    SmallVector<Operation *> insertedConstants;
    for (auto [idx, result] : llvm::enumerate(results)) {
        if (auto value = result.dyn_cast<Value>()) {
            foldedResults.push_back(value);
            continue;
        }

        if (auto attr = result.dyn_cast<Attribute>()) {
            // Attempt to materialize a constant value.
            if (auto constant = op->getDialect()->materializeConstant(
                    builder,
                    attr,
                    op->getResultTypes()[idx],
                    op->getLoc())) {
                assert(constant->getNumResults() == 1);
                assert(
                    constant->getResultTypes()[0] == op->getResultTypes()[idx]);
                foldedResults.push_back(constant->getResult(0));
                insertedConstants.push_back(constant);
                continue;
            }
        }

        // We failed to materialize a result. Roll back and abort.
        for (auto constant : insertedConstants) constant->erase();
        return failure();
    }

    // Insert the constants we materialized and replace the old op.
    for (auto constant : insertedConstants) rewriter.insert(constant);
    rewriter.replaceOp(op, foldedResults);
    return success();
}

namespace {

struct ThroughAnyOp : RewritePattern {
    ThroughAnyOp(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, 1, ctx)
    {}

    virtual LogicalResult
    matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override
    {
        if (!isUnreachable(op) || op->use_empty()) return failure();

        makeNoReturn(op, rewriter);
        return success();
    }
};

struct ThroughPureOp : RewritePattern {
    ThroughPureOp(MLIRContext *ctx)
            : RewritePattern(MatchAnyOpTypeTag{}, 1, ctx)
    {}

    virtual LogicalResult
    matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override
    {
        if (!isPure(op) || !isUnreachable(op) || op->use_empty())
            return failure();

        makeNoReturn(op, rewriter);
        return success();
    }
};

struct ThroughConstexpr : RewritePattern {
    ThroughConstexpr(MLIRContext *ctx)
            : RewritePattern(MatchAnyOpTypeTag{}, 1, ctx)
    {}

    virtual LogicalResult
    matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override
    {
        if (!isPure(op) || op->use_empty()) return failure();

        // Collect all constant operands.
        SmallVector<Attribute> operands(op->getNumOperands(), {});
        for (auto [idx, operand] : llvm::enumerate(op->getOperands()))
            if (!matchPattern(operand, m_Constant(&operands[idx])))
                return failure();
        assert(llvm::count(operands, Attribute{}) == 0);

        // Attempt to fold the expression.
        SmallVector<OpFoldResult> results;
        if (succeeded(op->fold(operands, results))) {
            // Not what we were after but still great.
            return completeFold(op, results, rewriter);
        }

        makeNoReturn(op, rewriter);
        return success();
    }
};

struct PropagateNeverPass
        : mlir::ubx::impl::PropagateNeverBase<PropagateNeverPass> {
    using PropagateNeverBase::PropagateNeverBase;

    void runOnOperation() override;
};

} // namespace

void PropagateNeverPass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());

    populatePropagateNeverPatterns(patterns, throughOps.getValue());

    if (failed(applyPatternsAndFoldGreedily(
            getOperation(),
            FrozenRewritePatternSet(std::move(patterns)))))
        return signalPassFailure();
}

void mlir::ubx::populatePropagateNeverPatterns(
    RewritePatternSet &patterns,
    OpNeverPropagation throughOps)
{
    switch (throughOps) {
    case OpNeverPropagation::Always:
        patterns.add<ThroughAnyOp>(patterns.getContext());
        break;
    case OpNeverPropagation::Pure:
        patterns.add<ThroughPureOp>(patterns.getContext());
        break;
    case OpNeverPropagation::Constexpr:
        patterns.add<ThroughConstexpr>(patterns.getContext());
        break;
    default: break;
    }
}

std::unique_ptr<Pass> mlir::ubx::createPropagateNeverPass()
{
    return std::make_unique<PropagateNeverPass>();
}
