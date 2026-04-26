
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
  //
  // Pass order (normative — matches docs/spec/LOWERING_PIPELINE_SPEC.md §2.1):
  //   1. tessera-effect-annotation     — annotate tessera.effect on func.func
  //   2. tessera-canonicalize          — fuse/simplify Graph IR patterns
  //   3. tessera-distribution-lowering — tessera.shard → schedule.mesh.*
  //   4. tessera-tiling                — tessera.matmul → scf.for tile loops
  //   5. tessera-tile-to-x86           — tiled matmul → func.call @tessera_x86_*
  ::mlir::PassPipelineRegistration<>
    lowerToX86("tessera-lower-to-x86",
               "Full Phase 2 lowering chain to x86 AMX/AVX-512 backend",
      [](OpPassManager &pm) {
        pm.addPass(createEffectAnnotationPass());
        pm.addPass(createCanonicalizeTesseraIRPass());
        pm.addPass(createDistributionLoweringPass());
        pm.addPass(createTilingPass());
        pm.addPass(createTileToX86Pass());
      });

  // ── Phase 3 passes ────────────────────────────────────────────────────────
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-tile-ir-lowering",
      "Lower tessera.flash_attn / tessera.matmul to FA-4 Tile IR ops",
      []() { return createTileIRLoweringPass(); });
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-warp-specialization",
      "Assign producer/consumer warp roles; insert tessera.queue barriers",
      []() { return createWarpSpecializationPass(); });
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-async-copy-lowering",
      "Lower tile.async_copy/wait_async to TMA (SM≥90) or cp.async (SM<90)",
      []() { return createAsyncCopyLoweringPass(); });
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-nvwgmma-lowering",
      "Lower tile.mma to wgmma.mma_async PTX (SM≥90) or nvgpu.mma.sync fallback",
      []() { return createNVWGMMALoweringPass(); });
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-nvtma-descriptor",
      "Hoist TMA descriptors to kernel preamble; assign mbarrier slots",
      []() { return createNVTMADescriptorPass(); });
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-nvflash-attn-emitter",
      "Finalise SM_90 FlashAttention kernel: scale, mbarrier, launch bounds",
      []() { return createNVFlashAttnKernelEmitterPass(); });

  // Full Phase 3 GPU lowering chain: Graph IR → SM_90 PTX.
  //
  // Pass order (normative — matches docs/spec/LOWERING_PIPELINE_SPEC.md §2.2):
  //   1. tessera-effect-annotation     — annotate tessera.effect on func.func
  //   2. tessera-canonicalize          — fuse/simplify Graph IR patterns
  //   3. tessera-distribution-lowering — tessera.shard → schedule.mesh.*
  //   4. tessera-tile-ir-lowering      — schedule.mesh.region → tile.* + attn.*
  //   5. tessera-warp-specialization   — warp role assignment + queue barriers
  //   6. tessera-async-copy-lowering   — tile.async_copy → TMA / cp.async
  //   7. tessera-nvwgmma-lowering      — tile.mma → wgmma.mma_async PTX
  //   8. tessera-nvtma-descriptor      — TMA descriptor hoisting + mbarrier init
  //   9. tessera-nvflash-attn-emitter  — FA-4 kernel finalisation
  ::mlir::PassPipelineRegistration<>
    lowerToGPU("tessera-lower-to-gpu",
               "Full Phase 3 lowering chain to NVIDIA SM_90 GPU backend",
      [](OpPassManager &pm) {
        pm.addPass(createEffectAnnotationPass());
        pm.addPass(createCanonicalizeTesseraIRPass());
        pm.addPass(createDistributionLoweringPass());
        pm.addPass(createTileIRLoweringPass());
        pm.addPass(createWarpSpecializationPass());
        pm.addPass(createAsyncCopyLoweringPass());
        pm.addPass(createNVWGMMALoweringPass());
        pm.addPass(createNVTMADescriptorPass());
        pm.addPass(createNVFlashAttnKernelEmitterPass());
      });
}
} // namespace tessera
