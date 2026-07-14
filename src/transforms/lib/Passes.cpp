
#include "Tessera/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace tessera {

// Opt-in knob for the named lowering pipelines: run LayoutAssignmentPass
// (seed kernel layouts → propagate through pointwise → insert
// `tessera.cast{layout}` markers) immediately before LayoutLegalityPass, so the
// two-sided layout contract (assign + verify) executes inside the pipeline.
//
// Default OFF on purpose. The assignment inserts same-type `tessera.cast`
// markers that are deliberately preserved from canonicalization, and no backend
// consumes them yet (rank-2 CPU JIT is layout-agnostic; Apple GPU is hand-MSL).
// Leaving it off keeps the *executing* x86/GPU lowering byte-identical; turning
// it on (e.g. `tessera-lower-to-x86{assign-layouts=true}`) exercises and
// verifies the assignment half end-to-end. When a layout-sensitive backend
// lands, this flips to default-on and the markers become real reorders.
struct TesseraLoweringPipelineOptions
    : public PassPipelineOptions<TesseraLoweringPipelineOptions> {
  Option<bool> assignLayouts{
      *this, "assign-layouts",
      llvm::cl::desc("Run LayoutAssignmentPass before layout legality "
                     "(default false; no backend consumes the cast{layout} "
                     "markers yet, so the default lowering path is unchanged)."),
      llvm::cl::init(false)};
  // C4 (2026-06-23): the compute/storage dtype legalize split. Opt-in (default
  // false) so the executing path stays byte-identical until a low-precision
  // backend consumes the storage_packed marker; compute-legalize runs before
  // IRContractLegality (so reduced-precision ops gain a wide accum and pass the
  // contract), storage-legalize runs terminally.
  Option<bool> legalizeDtypes{
      *this, "legalize-dtypes",
      llvm::cl::desc("Run compute-legalize (early) + storage-legalize "
                     "(terminal) — Decision #15a as pass ordering (default "
                     "false; additive annotations, executing path unchanged)."),
      llvm::cl::init(false)};
};

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

  // ── CF0 — control-flow target guard (standalone + wired into the non-Apple
  // pipelines below). Standalone form names the target via the `target` option.
  ::mlir::registerPass([]() { return createControlFlowTargetGuardPass(); });

  // ── CF2 — control-flow → scf lowering (control_for → scf.for).
  ::mlir::registerPass([]() { return createLowerControlFlowToSCFPass(); });

  // ── CF4a — decode the control_for op-list payload into a real @body func.
  ::mlir::registerPass([]() { return createMaterializeControlPayloadPass(); });

  // Full Phase 2 lowering chain: Graph IR → x86 CPU calls.
  //
  // Pass order (normative — matches docs/spec/LOWERING_PIPELINE_SPEC.md §2.1):
  //   1. tessera-effect-annotation     — annotate tessera.effect on func.func
  //   2. tessera-canonicalize          — fuse/simplify Graph IR patterns
  //   3. tessera-distribution-lowering — tessera.shard → schedule.mesh.*
  //   4. tessera-tiling                — tessera.matmul → scf.for tile loops
  //   5. tessera-tile-to-x86           — tiled matmul → func.call @tessera_x86_*
  ::mlir::PassPipelineRegistration<TesseraLoweringPipelineOptions>
    lowerToX86("tessera-lower-to-x86",
               "Full Phase 2 lowering chain to x86 AMX/AVX-512 backend",
      [](OpPassManager &pm, const TesseraLoweringPipelineOptions &opts) {
        addGraphIRPreLoweringPasses(pm);
        // CF2: lower tessera.control_for → scf.for FIRST, so a lowerable loop
        // becomes a portable scf.for and never reaches the guard below; the
        // executable-payload form is skipped and still guarded.
        pm.addPass(createLowerControlFlowToSCFPass());
        // CF0: x86 has no device control-flow lowering — reject the control_*
        // ops CF2 left with a stable diagnostic instead of a confusing
        // downstream failure.
        pm.addPass(createControlFlowTargetGuardPass("x86"));
        pm.addPass(createDistributionLoweringPass());
        // 2026-06-22: optional layout *assignment* runs just before legality so
        // the verifier validates the assignment + inserted cast{layout} markers.
        // Opt-in (default off) — see TesseraLoweringPipelineOptions.
        if (opts.assignLayouts)
          pm.addPass(createLayoutAssignmentPass());
        // 2026-06-17: layout legality now runs in the named pipelines (was
        // standalone) — early, so unknown-layout / producer-consumer-mismatch /
        // scale-without-layout violations surface with the other structural
        // diagnostics before lowering.
        pm.addPass(createLayoutLegalityPass());
        // C4 (2026-06-23): compute-legalize before the contract check so
        // reduced-precision storage gains a wide accum and passes #15a (gated).
        if (opts.legalizeDtypes)
          pm.addPass(createComputeLegalizePass());
        // 2026-06-19: dtype / aliasing / buffer-binding contracts (Decision
        // #15a) alongside layout legality — same early placement.
        pm.addPass(createIRContractLegalityPass());
        // Sprint V6b (2026-05-22): re-check symbolic-dim equality
        // AFTER DistributionLoweringPass so a downstream pass that
        // accidentally breaks a `where D = H * Dh` clause fails
        // here with a stable diagnostic (SYMDIM_*).  Ops without
        // dim-name annotations are skipped silently.
        pm.addPass(createSymbolicDimEqualityPass());
        pm.addPass(createTilingPass());
        pm.addPass(createTileToX86Pass());
        // C4 terminal storage-legalize — pack sub-byte storage last (gated).
        if (opts.legalizeDtypes)
          pm.addPass(createStorageLegalizePass());
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
  // 2026-06-23: real pipeline-stage partitioning + 1F1B schedule proof.
  ::mlir::registerPass([]() { return createPipelineStagePartitionPass(); });
  ::mlir::registerPass([]() { return createPipelineScheduleLegalityPass(); });

  // Pipeline-parallel lowering: partition into stages → insert send/recv SSA
  // rewrites → prove the 1F1B schedule. The single alias that drives the layer
  // from an unpartitioned function to a verified 1F1B pipeline.
  ::mlir::PassPipelineRegistration<>
    pipelinePP("tessera-pipeline",
               "Pipeline-parallel: stage partition -> send/recv insertion -> "
               "1F1B schedule legality",
      [](OpPassManager &pm) {
        pm.addPass(createPipelineStagePartitionPass());
        pm.addPass(createPipelineStageInsertionPass());
        pm.addPass(createPipelineScheduleLegalityPass());
      });

  // ── Phase F4 autodiff (reverse-mode via AdjointInterface) ──────────────────
  // ODS scaffold: src/compiler/ir/include/Tessera/AdjointInterface.td
  // Pass body: src/transforms/lib/AutodiffPass.cpp
  // Registers as `--tessera-autodiff` for `tessera-opt`. Until tablegen on
  // the .td runs, the pass body is a registered no-op (Python-tape autodiff
  // remains the production path). See docs/spec/AUTODIFF_SPEC.md §Phase F4.
  ::mlir::registerPass([]() { return createAutodiffPass(); });
  // Phase 2 — paired forward/backward autodiff (separate @f__bwd function).
  ::mlir::registerPass([]() { return createAutodiffPairedPass(); });

  // ── Phase 8.4.8 SwiGLU fusion (Stage 2b of SwiGLU Performance Plan) ───────
  // Matches the 3-op SwiGLU chain at the Graph IR layer and emits
  // `tessera.swiglu_fused`. Runs ahead of backend-specific lowering so
  // each backend gets the fused op as input (longest-fusion-first matches
  // Apple GPU pipeline ordering — see `apple_backend.md`).
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
  // 2026-06-17: layout assignment (seed → propagate → insert cast{layout}),
  // verified by LayoutLegalityPass. Standalone for now (no backend consumes the
  // assignments yet); registered as --tessera-layout-assignment.
  ::mlir::registerPass([]() { return createLayoutAssignmentPass(); });
  // 2026-07-08: Tile IR global buffer assignment/reuse (Workstream H / W3) —
  // disjoint-live-range shared buffers share a reuse group (tile.buffer_group).
  // Assignment half of shared-memory planning; verified by
  // TileBarrierReuseLegalityPass. Registered as --tessera-tile-buffer-reuse.
  ::mlir::registerPass([]() { return createTileBufferReusePass(); });
  // 2026-07-08: realize the reuse plan into a concrete SMEM/TMEM arena
  // (tile.smem_offset / tile.tmem_offset + arena bytes) — the first consumer of
  // tile.buffer_group. Registered as --tessera-tile-buffer-arena.
  ::mlir::registerPass([]() { return createTileBufferArenaPass(); });
  // 2026-06-19: dtype / aliasing / buffer-binding contracts (Decision #15a).
  // LayoutLegalityPass's sibling for the remaining contract families;
  // registered standalone as --tessera-ir-contracts.
  ::mlir::registerPass([]() { return createIRContractLegalityPass(); });
  // 2026-06-23: C2 — barriers as a layout-reuse correctness property (TIRx
  // review). Standalone as --tessera-tile-barrier-reuse-legality.
  ::mlir::registerPass([]() { return createTileBarrierReuseLegalityPass(); });
  // 2026-06-23: C3 — typed-barrier + pipeline-state cross-op legality (phase
  // asymmetry + barrier-kind consistency). --tessera-tile-pipeline-legality.
  ::mlir::registerPass([]() { return createTilePipelineLegalityPass(); });
  // 2026-06-23: C4 — compute/storage dtype legalize split (Decision #15a as
  // pass ordering). --tessera-compute-legalize (early) / --tessera-storage-
  // legalize (terminal).
  ::mlir::registerPass([]() { return createComputeLegalizePass(); });
  ::mlir::registerPass([]() { return createStorageLegalizePass(); });
  // 2026-06-23: the first real consumer of the C4 packing markers.
  ::mlir::registerPass([]() { return createStoragePackConsumePass(); });
  // 2026-06-23: C6 — warp-spec structural diagnostics (init placement,
  // collective-in-branch, loop-count agreement, TMA visibility fence).
  // --tessera-warpspec-legality.
  ::mlir::registerPass([]() { return createWarpSpecLegalityPass(); });

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

  // ── Phase F2 (IR form) activation rematerialization ─────────────────────────
  // Clones `tessera.recompute`-tagged pure activations to their backward
  // consumers (activation checkpointing at the IR level). The Graph-IR
  // counterpart of the tessera.autodiff.rematerialize Python surface.
  ::mlir::registerPass(
      []() { return createActivationRematerializationPass(); });

  // Full reverse-mode autodiff pipeline: forward → autodiff → rematerialize →
  // collective insertion. Remat runs before collective insertion so the
  // recomputed activations live inside the backward region the collectives
  // then synchronise.
  ::mlir::PassPipelineRegistration<>
    autodiffPipeline("tessera-autodiff-pipeline",
                     "Phase F4+F2+F5 — reverse-mode autodiff with rematerialization "
                     "and adjoint collective insertion",
      [](OpPassManager &pm) {
        pm.addPass(createAutodiffPass());
        pm.addPass(createActivationRematerializationPass());
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
  ::mlir::PassPipelineRegistration<TesseraLoweringPipelineOptions>
    lowerToGPU("tessera-lower-to-gpu",
               "Full Phase 3 lowering chain to NVIDIA SM_90 GPU backend",
      [](OpPassManager &pm, const TesseraLoweringPipelineOptions &opts) {
        addGraphIRPreLoweringPasses(pm);
        // CF2: lower region-structured forms to SCF first. Executable payload
        // forms remain Graph ops until TileIRLowering gives them a typed Tile
        // carrier; only unsupported residual Graph forms are guarded below.
        pm.addPass(createLowerControlFlowToSCFPass());
        pm.addPass(createDistributionLoweringPass());
        // 2026-06-22: optional layout assignment (see lowerToX86 / opts).
        if (opts.assignLayouts)
          pm.addPass(createLayoutAssignmentPass());
        // 2026-06-17: layout legality in the named pipeline (see lowerToX86).
        pm.addPass(createLayoutLegalityPass());
        // C4 (2026-06-23): compute-legalize before the contract check (gated).
        if (opts.legalizeDtypes)
          pm.addPass(createComputeLegalizePass());
        // 2026-06-19: dtype / aliasing / buffer-binding contracts (Decision
        // #15a) alongside layout legality — matches lowerToX86 / CUDA13.
        pm.addPass(createIRContractLegalityPass());
        // Sprint V6b (2026-05-22): symbolic-dim equality recheck
        // after distribution lowering (see lowerToX86 comment).
        pm.addPass(createSymbolicDimEqualityPass());
        pm.addPass(createTileIRLoweringPass());
        pm.addPass(createControlFlowTargetGuardPass("nvidia_sm90"));
        pm.addPass(createWarpSpecializationPass());
        // C2/C3/C6 (2026-06-23): warp-spec legality gates run on the markers
        // WarpSpecialization now emits — phase asymmetry + barrier-kind (C3),
        // reuse hazards (C2), and structural invariants (C6). Pure checks.
        pm.addPass(createTilePipelineLegalityPass());
        pm.addPass(createWarpSpecLegalityPass());
        pm.addPass(createTileBarrierReuseLegalityPass());
        pm.addPass(createAsyncCopyLoweringPass());
        pm.addPass(createNVWGMMALoweringPass());
        pm.addPass(createNVTMADescriptorPass());
        // C3/C6 again — now over the typed #tile.barrier markers
        // NVTMADescriptor emits (kind consistency + arrival-count == init-count).
        pm.addPass(createTilePipelineLegalityPass());
        pm.addPass(createWarpSpecLegalityPass());
        pm.addPass(createNVFlashAttnKernelEmitterPass());
        // C4 terminal storage-legalize (gated).
        if (opts.legalizeDtypes)
          pm.addPass(createStorageLegalizePass());
      });

  // ── Sprint G-5 (2026-05-11) — NVIDIATargetPipeline ─────────────────────
  //
  // CUDA 13.3 pinned variant of `tessera-lower-to-gpu`.  The pass
  // order is identical (see normative reference above); this alias adds:
  //
  //   * Toolchain pin recorded as the pipeline description (lit fixtures
  //     and `tessera-mlir` introspection can verify the pin via the
  //     `--help` text).
  //   * Hardware-free target contract: every pass below emits IR that
  //     CUDA 13.3 `nvcc -ptx -arch=sm_90a` accepts — verified by
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

  auto buildCUDA13Pipeline = [](OpPassManager &pm,
                                const TesseraLoweringPipelineOptions &opts) {
    addGraphIRPreLoweringPasses(pm);
    // CF2: region-structured forms become SCF. Executable payload forms remain
    // Graph ops until TileIRLowering converts them to typed Tile carriers.
    pm.addPass(createLowerControlFlowToSCFPass());
    pm.addPass(createDistributionLoweringPass());
    // 2026-06-22: optional layout assignment (see lowerToX86 / opts).
    if (opts.assignLayouts)
      pm.addPass(createLayoutAssignmentPass());
    // 2026-06-17: layout legality in the named pipeline (see lowerToX86).
    pm.addPass(createLayoutLegalityPass());
    // C4 (2026-06-23): compute-legalize before the contract check (gated).
    if (opts.legalizeDtypes)
      pm.addPass(createComputeLegalizePass());
    // 2026-06-19: dtype / aliasing / buffer-binding contracts (Decision #15a).
    pm.addPass(createIRContractLegalityPass());
    // Sprint V6b (2026-05-22): symbolic-dim equality recheck.
    pm.addPass(createSymbolicDimEqualityPass());
    pm.addPass(createTileIRLoweringPass());
    // Decision #21 still rejects any Graph control form outside the Tile
    // envelope, but no longer rejects supported NVIDIA payload contracts.
    pm.addPass(createControlFlowTargetGuardPass("nvidia_sm90"));
    pm.addPass(createWarpSpecializationPass());
    // C2/C3/C6 (2026-06-23): warp-spec legality gates on the WarpSpec markers.
    pm.addPass(createTilePipelineLegalityPass());
    pm.addPass(createWarpSpecLegalityPass());
    pm.addPass(createTileBarrierReuseLegalityPass());
    pm.addPass(createAsyncCopyLoweringPass());
    pm.addPass(createNVWGMMALoweringPass());
    pm.addPass(createNVTMADescriptorPass());
    // C3/C6 again — over the typed #tile.barrier markers (kind + arrival-count).
    pm.addPass(createTilePipelineLegalityPass());
    pm.addPass(createWarpSpecLegalityPass());
    pm.addPass(createNVFlashAttnKernelEmitterPass());
    // C4 terminal storage-legalize (gated).
    if (opts.legalizeDtypes)
      pm.addPass(createStorageLegalizePass());
  };

  ::mlir::PassPipelineRegistration<TesseraLoweringPipelineOptions>
    nvidiaPipeline("tessera-nvidia-pipeline",
                   "Sprint G-5: NVIDIATargetPipeline (CUDA 13.3, default SM_90) — "
                   "WarpSpec → AsyncCopy → WGMMA → TMA → NVPTXLowering. "
                   "Toolchain pin: nvcc 13.3, PTX ISA 9.3, NCCL 2.22.",
                   buildCUDA13Pipeline);

  ::mlir::PassPipelineRegistration<TesseraLoweringPipelineOptions>
    nvidiaPipelineSM90("tessera-nvidia-pipeline-sm90",
                       "Sprint G-5: NVIDIATargetPipeline pinned to SM_90 (Hopper) "
                       "under CUDA 13.3.  Emits WGMMA + TMA + mbarrier paths.",
                       buildCUDA13Pipeline);

  ::mlir::PassPipelineRegistration<TesseraLoweringPipelineOptions>
    nvidiaPipelineSM100("tessera-nvidia-pipeline-sm100",
                        "Sprint G-5: NVIDIATargetPipeline pinned to SM_100 (Blackwell) "
                        "under CUDA 13.3.  Emits TCGEN05 / TMEM / block-scaled MMA "
                        "paths via the WGMMA lowering's sm=100 mode.",
                        buildCUDA13Pipeline);

  ::mlir::PassPipelineRegistration<TesseraLoweringPipelineOptions>
    nvidiaPipelineSM120("tessera-nvidia-pipeline-sm120",
                        "Sprint G-5: NVIDIATargetPipeline pinned to SM_120 (Rubin) "
                        "under CUDA 13.3 (preliminary intrinsic set).",
                        buildCUDA13Pipeline);
}
} // namespace tessera
