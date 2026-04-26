
#include "Tessera/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace tessera {
void registerTesseraPasses() {
  // ── Phase 1 passes ────────────────────────────────────────────────────────
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
    cleanupPipeline("tessera-cleanup", "Migration + canonicalization pipeline",
      [](OpPassManager &pm) {
        pm.addPass(createMigrateTesseraIRPass());
        pm.addPass(createCanonicalizeTesseraIRPass());
      });

  // ── Phase 2 passes ────────────────────────────────────────────────────────
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-distribution-lowering",
      "Lower tessera.shard attrs to schedule.mesh.define + mesh.region",
      []() { return createDistributionLoweringPass(); });
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-effect-annotation",
      "Infer and annotate tessera.effect on each func.func",
      []() { return createEffectAnnotationPass(); });
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-tiling",
      "Tile tessera.matmul into scf.for M×N loop nests",
      []() { return createTilingPass(); });
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-tile-to-x86",
      "Lower tiled tessera.matmul to tessera_x86_backend C function calls",
      []() { return createTileToX86Pass(); });

  // Full Phase 2 lowering chain: Graph IR → x86 CPU calls.
  // Equivalent to:
  //   tessera-distribution-lowering
  //   tessera-effect-annotation
  //   tessera-tiling
  //   tessera-tile-to-x86
  ::mlir::PassPipelineRegistration<>
    lowerToX86("tessera-lower-to-x86",
               "Full Phase 2 lowering chain to x86 AMX/AVX-512 backend",
      [](OpPassManager &pm) {
        pm.addPass(createDistributionLoweringPass());
        pm.addPass(createEffectAnnotationPass());
        pm.addPass(createTilingPass());
        pm.addPass(createTileToX86Pass());
      });
}
} // namespace tessera
