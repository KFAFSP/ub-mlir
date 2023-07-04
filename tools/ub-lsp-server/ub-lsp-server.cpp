/// Main entry point for the ub-mlir MLIR language server.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "ub-mlir/Dialect/UB/IR/UB.h"

using namespace mlir;

#if MLIR_INCLUDE_TESTS
namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test
#endif

static int asMainReturnCode(LogicalResult r)
{
    return r.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char* argv[])
{
    DialectRegistry registry;
    registerAllDialects(registry);

    registry.insert<ub::UBDialect>();
#if MLIR_INCLUDE_TESTS
    test::registerTestDialect(registry);
#endif

    return asMainReturnCode(MlirLspServerMain(argc, argv, registry));
}
