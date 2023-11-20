/// Declares the UBX to LLVM conversion pass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::ubx {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTUBXTOLLVM
#include "ub-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

/// Adds the convert-ubx-to-llvm pass patterns to @p patterns .
void populateConvertUBXToLLVMPatterns(
    LLVMTypeConverter &converter,
    RewritePatternSet &patterns);

/// Constructs the convert-ubx-to-llvm pass.
std::unique_ptr<Pass> createConvertUBXToLLVMPass();

} // namespace mlir::ubx
