/// Implements the ConvertUBXToLLVMPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ub-mlir/Conversion/UBXToLLVM/UBXToLLVM.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ub-mlir/Dialect/UBX/IR/UBX.h"

using namespace mlir;
using namespace mlir::ubx;

//===- Generated includes -------------------------------------------------===//

namespace mlir::ubx {

#define GEN_PASS_DEF_CONVERTUBXTOLLVM
#include "ub-mlir/Conversion/Passes.h.inc"

} // namespace mlir::ubx

//===----------------------------------------------------------------------===//

namespace {

struct ConvertPoison : ConvertOpToLLVMPattern<PoisonOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    virtual LogicalResult matchAndRewrite(
        PoisonOp op,
        PoisonOp::Adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Ensure the result type is an LLVM compatible type.
        auto llvmTy = getTypeConverter()->convertType(op.getType());
        if (!llvmTy) llvmTy = op.getType();
        if (LLVM::isCompatibleType(llvmTy)) {
            // Source materialization will take place if there was a conversion.
            rewriter.replaceOpWithNewOp<LLVM::PoisonOp>(op, llvmTy);
            return success();
        }

        return rewriter.notifyMatchFailure(op, "unsupported type");
    }
};

using ConvertFreeze = OneToOneConvertToLLVMPattern<FreezeOp, LLVM::FreezeOp>;

struct ConvertUBXToLLVMPass
        : mlir::ubx::impl::ConvertUBXToLLVMBase<ConvertUBXToLLVMPass> {
    using ConvertUBXToLLVMBase::ConvertUBXToLLVMBase;

    void runOnOperation() override;
};

struct UBXToLLVMInterface : public ConvertToLLVMPatternInterface {
    using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

    virtual void populateConvertToLLVMConversionPatterns(
        ConversionTarget &,
        LLVMTypeConverter &typeConverter,
        RewritePatternSet &patterns) const override
    {
        populateConvertUBXToLLVMPatterns(typeConverter, patterns);
    }
};

} // namespace

void ConvertUBXToLLVMPass::runOnOperation()
{
    // Set LLVM lowering options.
    LowerToLLVMOptions options(&getContext());
    LLVMTypeConverter typeConverter(&getContext(), options);

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    ubx::populateConvertUBXToLLVMPatterns(typeConverter, patterns);

    // All operations must be converted to LLVM.
    target.addIllegalDialect<ubx::UBXDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

void mlir::ubx::populateConvertUBXToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    // When we're converting to LLVM, we implicitly perform ub-freeze, which
    // will try to instantiate values for constant operand 'ubx.freeze' ops
    // that we otherwise might be unable to convert.
    populateFreezePatterns(patterns);

    patterns.add<ConvertPoison, ConvertFreeze>(typeConverter);
}

std::unique_ptr<Pass> mlir::ubx::createConvertUBXToLLVMPass()
{
    return std::make_unique<ConvertUBXToLLVMPass>();
}

void mlir::ubx::registerConvertUBXToLLVMInterface(DialectRegistry &registry)
{
    registry.addExtension(+[](MLIRContext *, ubx::UBXDialect *dialect) {
        dialect->addInterfaces<UBXToLLVMInterface>();
    });
}
