//===- Passes.td - Conversion passes ------------------------*- tablegen -*-===//
//
// This is the definitions file for the ub-mlir dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef UB_MLIR_CONVERSION_PASSES
#define UB_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertUBXToLLVM : Pass<"convert-ubx-to-llvm"> {
    let summary = "Convert UBX dialect to LLVM dialect";

    let dependentDialects = [
        "::mlir::ubx::UBXDialect",
        "::mlir::LLVM::LLVMDialect"
    ];

    let constructor = "::mlir::ubx::createConvertUBXToLLVMPass()";
}

#endif
