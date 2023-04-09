/// Implements the ConvertUBToLLVMPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Conversion/UBToLLVM/UBToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ub-mlir/Dialect/UB/IR/UB.h"

using namespace mlir;
using namespace mlir::ub;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTUBTOLLVM
#include "ub-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//

namespace {

struct ConvertUBToLLVMPass
        : mlir::impl::ConvertUBToLLVMBase<ConvertUBToLLVMPass> {
    using ConvertUBToLLVMBase::ConvertUBToLLVMBase;

    void runOnOperation() override;
};

struct ConvertPoison : ConvertOpToLLVMPattern<PoisonOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    virtual LogicalResult matchAndRewrite(
        PoisonOp op,
        PoisonOp::Adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        if (!op.getValue().isPoison()) {
            // TODO: Requires a dynamic aggregate poisoning mechanism to
            //       implement partial poisoning:
            //          - Turn into an insertion of non-poisoned into fully-
            //            poisoned?
            return rewriter.notifyMatchFailure(
                op,
                "can only convert fully poisoned values");
        }

        if (!getTypeConverter()->isLegal(op.getType())) {
            return rewriter.notifyMatchFailure(
                op,
                "result type is not a legal LLVM type");
        }

        rewriter.replaceOpWithNewOp<LLVM::PoisonOp>(op, op.getType());
        return success();
    }
};

using ConvertFreeze = OneToOneConvertToLLVMPattern<FreezeOp, LLVM::FreezeOp>;

using ConvertUnreachable =
    OneToOneConvertToLLVMPattern<UnreachableOp, LLVM::UnreachableOp>;

} // namespace

void ConvertUBToLLVMPass::runOnOperation()
{
    // Set LLVM lowering options.
    LowerToLLVMOptions options(&getContext());
    LLVMTypeConverter typeConverter(&getContext(), options);

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    ub::populateConvertUBToLLVMPatterns(typeConverter, patterns);

    // All operations must be converted to LLVM.
    target.addIllegalDialect<UBDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

void mlir::ub::populateConvertUBToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertPoison, ConvertFreeze, ConvertUnreachable>(
        typeConverter);
}

std::unique_ptr<Pass> mlir::createConvertUBToLLVMPass()
{
    return std::make_unique<ConvertUBToLLVMPass>();
}
