/// Declaration of the UB to LLVM conversion pass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTUBTOLLVM
#include "ub-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace ub {

/// Adds the convert-ub-to-llvm pass patterns to @p patterns .
void populateConvertUBToLLVMPatterns(
    LLVMTypeConverter &converter,
    RewritePatternSet &patterns);

} // namespace ub

/// Constructs the convert-ub-to-llvm pass.
std::unique_ptr<Pass> createConvertUBToLLVMPass();

} // namespace mlir
