#include "Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

std::unique_ptr<mlir::Pass> createLowerP3DPass();
std::unique_ptr<mlir::Pass> createAutotuneP3DPass();

using namespace mlir;

namespace {
struct RegisterAllP3DPasses {
  RegisterAllP3DPasses() {
    PassRegistration<> lower("tessera-lower-p3d", "Lower P3D ops to Tessera Tile/Target IR.",
      [](){ return createLowerP3DPass(); });
    PassRegistration<> tune("tessera-autotune-p3d", "Attach autotuning spaces for P3D.",
      [](){ return createAutotuneP3DPass(); });
  }
} registerAllP3DPasses;
} // namespace
