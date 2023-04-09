//===- Passes.td - Conversion passes ------------------------*- tablegen -*-===//
//
// This is the definitions file for the ub-mlir dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef UB_MLIR_CONVERSION_PASSES
#define UB_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertUBToLLVM : Pass<"convert-ub-to-llvm"> {
    let summary = "Convert UB dialect to LLVM";

    let constructor = "mlir::createConvertUBToLLVMPass()";

    let dependentDialects = [
        "ub::UBDialect",
        "LLVM::LLVMDialect"
    ];
}

#endif
