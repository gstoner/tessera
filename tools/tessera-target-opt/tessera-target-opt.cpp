// Target-IR-only parser/verifier.  A successful parse here proves no Graph or
// Tile IR leaked across the target boundary: neither dialect is registered.
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#ifdef TESSERA_HAVE_NVIDIA_TARGET_IR
#include "tessera/gpu/BackendRegistration.h"
#endif

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
#ifdef TESSERA_HAVE_NVIDIA_TARGET_IR
  tessera::registerTesseraNVIDIATargetDialect(registry);
#endif
  return failed(mlir::MlirOptMain(
      argc, argv,
      "tessera-target-opt\n"
      "  Target-IR-only parser/verifier; Graph and Tile IR are intentionally "
      "unregistered.\n",
      registry));
}
