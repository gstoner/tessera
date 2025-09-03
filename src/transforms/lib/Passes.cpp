
#include "Tessera/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace tessera {
void registerTesseraPasses() {
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-canonicalize", "Canonicalize Tessera IR patterns",
      []() { return createCanonicalizeTesseraIRPass(); });
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-verify", "Module-level verification for Tessera IR",
      []() { return createVerifyTesseraIRPass(); });
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-migrate-ir", "Migrate Tessera IR across versions",
      []() { return createMigrateTesseraIRPass(); });

  ::mlir::PassPipelineRegistration<>
    cleanupPipeline("tessera-cleanup", "Migration + canonicalization cleanup pipeline",
      [](OpPassManager &pm) {
        pm.addPass(createMigrateTesseraIRPass());
        pm.addPass(createCanonicalizeTesseraIRPass());
      });
}
} // namespace tessera
