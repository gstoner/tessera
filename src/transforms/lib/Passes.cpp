
#include "Tessera/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace tessera {
void registerTesseraPasses() {
  // ── Phase 1 passes ────────────────────────────────────────────────────────
  ::mlir::registerPass([]() { return createCanonicalizeTesseraIRPass(); });
  ::mlir::registerPass([]() { return createVerifyTesseraIRPass(); });
  ::mlir::registerPass([]() { return createMigrateTesseraIRPass(); });

  ::mlir::PassPipelineRegistration<>
    cleanupPipeline("tessera-cleanup", "Migration + canonicalization pipeline",
      [](OpPassManager &pm) {
        pm.addPass(createMigrateTesseraIRPass());
        pm.addPass(createCanonicalizeTesseraIRPass());
      });

  // ── Phase 2 passes ────────────────────────────────────────────────────────
  ::mlir::registerPass([]() { return createDistributionLoweringPass(); });
  ::mlir::registerPass([]() { return createEffectAnnotationPass(); });
  ::mlir::registerPass([]() { return createTilingPass(); });
  ::mlir::registerPass([]() { return createTileToX86Pass(); });

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
  ::mlir::registerPass([]() { return createTileIRLoweringPass(); });
  ::mlir::registerPass([]() { return createWarpSpecializationPass(); });
  ::mlir::registerPass([]() { return createAsyncCopyLoweringPass(); });
  ::mlir::registerPass([]() { return createNVWGMMALoweringPass(); });
  ::mlir::registerPass([]() { return createNVTMADescriptorPass(); });
  ::mlir::registerPass([]() { return createNVFlashAttnKernelEmitterPass(); });

  // ── Phase 4 passes ────────────────────────────────────────────────────────
  ::mlir::registerPass([]() { return createGPUCollectiveInsertionPass(); });
  ::mlir::registerPass([]() { return createPipelineStageInsertionPass(); });

  // ── Phase F4 autodiff (reverse-mode via AdjointInterface) ──────────────────
  // ODS scaffold: src/compiler/ir/include/Tessera/AdjointInterface.td
  // Pass body: src/transforms/lib/AutodiffPass.cpp
  // Registers as `--tessera-autodiff` for `tessera-opt`. Until tablegen on
  // the .td runs, the pass body is a registered no-op (Python-tape autodiff
  // remains the production path). See docs/spec/AUTODIFF_SPEC.md §Phase F4.
  ::mlir::registerPass([]() { return createAutodiffPass(); });

  // ── Phase 8.4.8 SwiGLU fusion (Stage 2b of SwiGLU Performance Plan) ───────
  // Matches the 3-op SwiGLU chain at the Graph IR layer and emits
  // `tessera.swiglu_fused`. Runs ahead of backend-specific lowering so
  // each backend gets the fused op as input (longest-fusion-first matches
  // Apple GPU pipeline ordering — see `apple_gpu_overview.md`).
  ::mlir::registerPass([]() { return createSwigluFusionPass(); });

  // ── attention_variants_plan, MLA-1 — DeepSeek MLA decode fusion ────────
  // Matches the (compress → expand_k/v → flash_attn) chain and emits
  // `tessera.mla_decode_fused`. Runs ahead of backend-specific lowering
  // for the same reason as SwigluFusion.
  ::mlir::registerPass([]() { return createMLAFusionPass(); });

  // ── attention_variants_plan, NSA-4 — DeepSeek NSA fusion ────────────────
  // Matches the three-branch NSA shape and emits
  // `tessera.native_sparse_attn_fused`.
  ::mlir::registerPass([]() { return createNativeSparseAttnFusionPass(); });

  // ── Phase F5 adjoint collective insertion ──────────────────────────────────
  // Runs after AutodiffPass on functions carrying both
  // tessera.autodiff="reverse" and tessera.weight_sharding. Plans
  // reduce_scatter/all_gather/all_reduce per arg and records the choice as
  // `tessera.adjoint_collective_plan`.
  ::mlir::registerPass([]() { return createAdjointCollectiveInsertionPass(); });

  // Full reverse-mode autodiff pipeline: forward → autodiff → collective insertion.
  ::mlir::PassPipelineRegistration<>
    autodiffPipeline("tessera-autodiff-pipeline",
                     "Phase F4+F5 — reverse-mode autodiff with adjoint collective insertion",
      [](OpPassManager &pm) {
        pm.addPass(createAutodiffPass());
        pm.addPass(createAdjointCollectiveInsertionPass());
      });

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
