
#include "Tessera/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace tessera {

// Shared Graph IR pre-lowering stage (audit 2026-06-10). Previously this
// sequence was copy-pasted into tessera-lower-to-x86, tessera-lower-to-gpu,
// and the CUDA-13 pipeline builder — adding a fusion pass required three
// edits. It also gains the standard upstream cleanup that no named pipeline
// scheduled before: canonicalizer (folding + DCE of unused Pure ops) and CSE
// (dedup of repeated pure computations), run AFTER the Tessera fusion passes
// so lowering sees fused, deduplicated IR.
static void addGraphIRPreLoweringPasses(OpPassManager &pm) {
  pm.addPass(createEffectAnnotationPass());
  pm.addPass(createCanonicalizeTesseraIRPass());
  pm.addPass(createSwigluFusionPass());
  pm.addPass(createMLAFusionPass());
  pm.addPass(createNativeSparseAttnFusionPass());
  pm.addPass(createHybridAttnExpandPass());
  pm.addPass(createLightningAttnFusionPass());
  pm.addPass(createDeltaAttnChunkingPass());
  pm.addPass(createLookaheadSparseAttnExpandPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

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

  // ── Phase 0 production spine — Graph IR → upstream linalg ──────────────────
  ::mlir::registerPass([]() { return createTesseraToLinalgPass(); });

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
        addGraphIRPreLoweringPasses(pm);
        pm.addPass(createDistributionLoweringPass());
        // 2026-06-17: layout legality now runs in the named pipelines (was
        // standalone) — early, so unknown-layout / producer-consumer-mismatch /
        // scale-without-layout violations surface with the other structural
        // diagnostics before lowering.
        pm.addPass(createLayoutLegalityPass());
        // Sprint V6b (2026-05-22): re-check symbolic-dim equality
        // AFTER DistributionLoweringPass so a downstream pass that
        // accidentally breaks a `where D = H * Dh` clause fails
        // here with a stable diagnostic (SYMDIM_*).  Ops without
        // dim-name annotations are skipped silently.
        pm.addPass(createSymbolicDimEqualityPass());
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

  // ── attention-family plan — Ling/Kimi/Lightning/Delta pass slots ────────
  ::mlir::registerPass([]() { return createHybridAttnExpandPass(); });
  ::mlir::registerPass([]() { return createLightningAttnFusionPass(); });
  ::mlir::registerPass([]() { return createDeltaAttnChunkingPass(); });
  ::mlir::registerPass([]() { return createLookaheadSparseAttnExpandPass(); });
  ::mlir::registerPass([]() { return createLookaheadSparsePrefetchPass(); });
  ::mlir::registerPass([]() { return createMSAExpandPass(); });

  // ── Stage 13 — RL policy-loss compiler visibility/decomposition ─────────
  ::mlir::registerPass([]() { return createRLLossDecomposePass(); });
  ::mlir::registerPass([]() { return createVarlenSdpaDecomposePass(); });

  // ── Sprint V2 (2026-05-22) — Layout legality skeleton ─────────────────
  // Closes the "no LayoutLegalityPass" gap in SHAPE_SYSTEM.md §11.2.
  // First rule: tessera.cast with `tessera.layout` attribute outside
  // the canonical accept-set emits LAYOUT_LEGALITY_UNKNOWN_LAYOUT and
  // signals pass failure.  Registered standalone for now; inserted into
  // x86 / GPU pipelines in a follow-up sprint.
  ::mlir::registerPass([]() { return createLayoutLegalityPass(); });

  // ── Sprint V5 (2026-05-22) — Symbolic dim equality verifier ───────────
  // Closes the 4th MLIR-verifier gap in SHAPE_SYSTEM.md §11.2.
  // Reads `tessera.dim_bindings` + `tessera.dim_sizes` on each
  // func.func and validates the equations + per-op dim-name contracts
  // for reshape / transpose / matmul.  Registered standalone; inserted
  // after DistributionLoweringPass in named pipelines in V2.
  ::mlir::registerPass([]() { return createSymbolicDimEqualityPass(); });

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
        addGraphIRPreLoweringPasses(pm);
        pm.addPass(createDistributionLoweringPass());
        // 2026-06-17: layout legality in the named pipeline (see lowerToX86).
        pm.addPass(createLayoutLegalityPass());
        // Sprint V6b (2026-05-22): symbolic-dim equality recheck
        // after distribution lowering (see lowerToX86 comment).
        pm.addPass(createSymbolicDimEqualityPass());
        pm.addPass(createTileIRLoweringPass());
        pm.addPass(createWarpSpecializationPass());
        pm.addPass(createAsyncCopyLoweringPass());
        pm.addPass(createNVWGMMALoweringPass());
        pm.addPass(createNVTMADescriptorPass());
        pm.addPass(createNVFlashAttnKernelEmitterPass());
      });

  // ── Sprint G-5 (2026-05-11) — NVIDIATargetPipeline ─────────────────────
  //
  // CUDA 13.2 Update 1 pinned variant of `tessera-lower-to-gpu`.  The pass
  // order is identical (see normative reference above); this alias adds:
  //
  //   * Toolchain pin recorded as the pipeline description (lit fixtures
  //     and `tessera-mlir` introspection can verify the pin via the
  //     `--help` text).
  //   * Hardware-free target contract: every pass below emits IR that
  //     CUDA 13.2 U1 `nvcc -ptx -arch=sm_90a` accepts — verified by
  //     Lane-2 compile-only checks (Sprint G-6/G-8).
  //   * Pre-canonical attention-family fusion (SwiGLU / MLA / NSA / hybrid
  //     / Lightning / Delta) so backend lowering sees the fused ops.
  //   * Final NVPTX descriptor + flash-attn kernel emission.
  //
  // Aliases registered:
  //   - `tessera-nvidia-pipeline-sm90`   → SM_90 (Hopper)  : WGMMA + TMA
  //   - `tessera-nvidia-pipeline-sm100`  → SM_100 (Blackwell): + TCGEN05 + TMEM
  //   - `tessera-nvidia-pipeline-sm120`  → SM_120 (Rubin)  : same chain
  //   - `tessera-nvidia-pipeline`        → default = SM_90 chain
  //
  // All four aliases share the same pass list today — the per-SM
  // dispatching happens inside `createLowerTileToNVIDIAPass(sm)` in the
  // NVIDIA backend (`tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp`).
  // When SM_100/SM_120 add post-WGMMA passes (TCGEN05 / TMEM), they go
  // here under the corresponding alias.

  auto buildCUDA13Pipeline = [](OpPassManager &pm) {
    addGraphIRPreLoweringPasses(pm);
    pm.addPass(createDistributionLoweringPass());
    // 2026-06-17: layout legality in the named pipeline (see lowerToX86).
    pm.addPass(createLayoutLegalityPass());
    // Sprint V6b (2026-05-22): symbolic-dim equality recheck.
    pm.addPass(createSymbolicDimEqualityPass());
    pm.addPass(createTileIRLoweringPass());
    pm.addPass(createWarpSpecializationPass());
    pm.addPass(createAsyncCopyLoweringPass());
    pm.addPass(createNVWGMMALoweringPass());
    pm.addPass(createNVTMADescriptorPass());
    pm.addPass(createNVFlashAttnKernelEmitterPass());
  };

  ::mlir::PassPipelineRegistration<>
    nvidiaPipeline("tessera-nvidia-pipeline",
                   "Sprint G-5: NVIDIATargetPipeline (CUDA 13.2 U1, default SM_90) — "
                   "WarpSpec → AsyncCopy → WGMMA → TMA → NVPTXLowering. "
                   "Toolchain pin: nvcc 13.2 U1, PTX ISA 8.6, NCCL 2.22.",
                   buildCUDA13Pipeline);

  ::mlir::PassPipelineRegistration<>
    nvidiaPipelineSM90("tessera-nvidia-pipeline-sm90",
                       "Sprint G-5: NVIDIATargetPipeline pinned to SM_90 (Hopper) "
                       "under CUDA 13.2 U1.  Emits WGMMA + TMA + mbarrier paths.",
                       buildCUDA13Pipeline);

  ::mlir::PassPipelineRegistration<>
    nvidiaPipelineSM100("tessera-nvidia-pipeline-sm100",
                        "Sprint G-5: NVIDIATargetPipeline pinned to SM_100 (Blackwell) "
                        "under CUDA 13.2 U1.  Emits TCGEN05 / TMEM / block-scaled MMA "
                        "paths via the WGMMA lowering's sm=100 mode.",
                        buildCUDA13Pipeline);

  ::mlir::PassPipelineRegistration<>
    nvidiaPipelineSM120("tessera-nvidia-pipeline-sm120",
                        "Sprint G-5: NVIDIATargetPipeline pinned to SM_120 (Rubin) "
                        "under CUDA 13.2 U1 (preliminary intrinsic set).",
                        buildCUDA13Pipeline);
}
} // namespace tessera
