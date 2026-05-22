
// Register TPP dialect & passes in your tool (e.g., tessera-opt)
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

// If you have Tessera global init headers, include them as well.
extern "C" void registerTPPPipelineAlias(void*);

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  // Register all core MLIR dialects; add your Tessera ones too.
  mlir::registerAllDialects(registry);
  // TODO: register Tessera base dialects/passes here.

  // Force-load TPP dialect library by referencing a symbol (linker keeps it).
  // The pipeline registration will happen when the shared object is linked,
  // but we also explicitly call the alias function to be safe.
  registerTPPPipelineAlias(nullptr);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "tessera-opt with TPP pipeline",
                        registry));
}
