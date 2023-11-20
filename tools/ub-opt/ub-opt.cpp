/// Main entry point for the ub-mlir optimizer driver.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "ub-mlir/Conversion/Passes.h"
#include "ub-mlir/Dialect/UBX/IR/UBX.h"

using namespace mlir;

#if MLIR_INCLUDE_TESTS
namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test
#endif

int main(int argc, char* argv[])
{
    DialectRegistry registry;
    registerAllDialects(registry);

    registry.insert<ubx::UBXDialect>();
#if MLIR_INCLUDE_TESTS
    test::registerTestDialect(registry);
#endif

    registerAllPasses();
    ub_mlir::registerConversionPasses();

    return asMainReturnCode(
        MlirOptMain(argc, argv, "ub-mlir optimizer driver\n", registry));
}
