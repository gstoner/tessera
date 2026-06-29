
#pragma once
#include "mlir/Pass/Pass.h"

namespace tessera {

// Phase 1 passes
std::unique_ptr<mlir::Pass> createCanonicalizeTesseraIRPass();
std::unique_ptr<mlir::Pass> createVerifyTesseraIRPass();
std::unique_ptr<mlir::Pass> createMigrateTesseraIRPass();

// Phase 2 lowering chain
//
// Pipeline order (normative — see docs/spec/LOWERING_PIPELINE_SPEC.md §2.1):
//   1. tessera-effect-annotation     — annotate tessera.effect on func.func
//   2. tessera-canonicalize          — fuse/simplify Graph IR patterns
//   3. tessera-distribution-lowering — tessera.shard → schedule.mesh.*
//   4. tessera-tiling                — tessera.matmul → scf.for tile loops
//   5. tessera-tile-to-x86           — tiled matmul → func.call @tessera_x86_*
//
// Run the whole chain with: -tessera-lower-to-x86

// DistributionLoweringPass — converts tessera.shard argument attributes into
// schedule.mesh.define + schedule.mesh.region ops wrapping the function body.
// Options:
//   --mesh-axes  comma-separated axis names (e.g. "dp,tp")
//   --mesh-sizes comma-separated axis sizes (e.g. "4,4")
std::unique_ptr<mlir::Pass> createDistributionLoweringPass();

// EffectAnnotationPass — infers semantic effects from the function body and
// attaches tessera.effect = "pure"|"random"|"movement"|"state"|"collective"|
// "memory"|"io" to each func.func. Signals failure if a func annotated "pure"
// contains any higher effect.
std::unique_ptr<mlir::Pass> createEffectAnnotationPass();

// TilingPass — tiles tessera.matmul ops (inside schedule.mesh.region bodies)
// into scf.for loop nests over M and N tiles using tensor.extract/insert_slice.
// Options:
//   --tile-m  tile size along M dimension (default 16)
//   --tile-n  tile size along N dimension (default 16)
std::unique_ptr<mlir::Pass> createTilingPass();
// Apple Value Target IR sprint 5: value-mode tiling preserves static rank-2 f32
// matmul/gemm as a single tile op (for the Accelerate GEMM value call) instead
// of tiling to scf.for. Used by the apple_cpu `-full` value pipeline only.
std::unique_ptr<mlir::Pass> createTilingPass(bool valueMode);

// TileToX86Pass — replaces tiled tessera.matmul ops (static bf16 operands)
// with calls to the tessera_x86_backend C functions via func.call with raw
// i64 pointer arguments.  Handles fused_epilogue bias/gelu variants.
// Options:
//   --prefer-amx  prefer AMX over AVX-512 when both are available (default true)
std::unique_ptr<mlir::Pass> createTileToX86Pass();

// ── Phase 0 production spine — Graph IR → upstream linalg ─────────────────
// Lowers total elementwise Tessera Graph IR ops (currently tessera.add) to
// linalg.generic on tensors so the standard bufferize → llvm → ExecutionEngine
// pipeline can produce executable code. The shared front-half every production
// target inherits. See docs/spec/PRODUCTION_COMPILER_PLAN.md.
std::unique_ptr<mlir::Pass> createTesseraToLinalgPass();

// ── Phase 3 passes — GPU backend + FA-4 Tile IR ───────────────────────────
//
// Full GPU lowering pipeline (SM_90 FlashAttention)
// (normative — see docs/spec/LOWERING_PIPELINE_SPEC.md §2.2):
//   1. tessera-effect-annotation     — annotate tessera.effect on func.func
//   2. tessera-canonicalize          — fuse/simplify Graph IR patterns
//   3. tessera-distribution-lowering — tessera.shard → schedule.mesh.*
//   4. tessera-tile-ir-lowering      — schedule.mesh.region → tile.* + attn.*
//   5. tessera-warp-specialization   — warp role assignment + queue barriers
//   6. tessera-async-copy-lowering   — tile.async_copy → TMA / cp.async
//   7. tessera-nvwgmma-lowering      — tile.mma → wgmma.mma_async PTX
//   8. tessera-nvtma-descriptor      — TMA descriptor hoisting + mbarrier init
//   9. tessera-nvflash-attn-emitter  — FA-4 kernel finalisation
//
// Run the whole chain with: -tessera-lower-to-gpu

// TileIRLoweringPass — lowers tessera.flash_attn / tessera.matmul (inside
// schedule.mesh.region) to FA-4 Tile IR ops: tile.async_copy, tile.mma,
// tessera.attn.scaled_dot_product, tessera.attn.online_softmax, etc.
// Options:
//   --tile-q   Q tile rows (default 64)
//   --tile-kv  KV tile cols (default 64)
//   --sm       target SM version (default 90)
std::unique_ptr<mlir::Pass> createTileIRLoweringPass();

// WarpSpecializationPass — assigns producer/consumer warp roles to tile IR ops
// inside schedule.mesh.region bodies and inserts tessera.queue barriers.
std::unique_ptr<mlir::Pass> createWarpSpecializationPass();

// AsyncCopyLoweringPass — lowers tile.async_copy + tile.wait_async to
// tile.tma.* (SM≥90) or tile.cp_async.* (SM<90).
// Options:
//   --sm  target SM version (default 90)
std::unique_ptr<mlir::Pass> createAsyncCopyLoweringPass();

// NVWGMMALoweringPass — lowers tile.mma to wgmma.mma_async PTX inline asm
// for SM_90+ or nvgpu.mma.sync WMMA for SM<90.
// Options:
//   --sm  target SM version (default 90)
std::unique_ptr<mlir::Pass> createNVWGMMALoweringPass();

// NVTMADescriptorPass — hoists tile.tma.descriptor ops to the kernel
// preamble, deduplicates them, and assigns unique mbarrier slot indices.
std::unique_ptr<mlir::Pass> createNVTMADescriptorPass();

// NVFlashAttnKernelEmitterPass — finalises the FA-2 FlashAttention kernel:
// resolves scale sentinels, emits mbarrier arrive/wait PTX, attaches CUDA
// launch bounds, annotates shared memory budget.
// Options:
//   --sm       target SM version (default 90)
//   --tile-q   Q tile rows (default 64)
//   --tile-kv  KV tile cols (default 64)
//   --warps    warps per CTA (default 4)
std::unique_ptr<mlir::Pass> createNVFlashAttnKernelEmitterPass();

// ── Phase 4 passes — Distributed training collectives + pipeline ──────────
//
// Pipeline order (after Phase 2/3 distribution + effect annotation):
//   1. tessera-gpu-collective-insertion — insert reduce_scatter/all_gather
//      at DP/TP mesh boundaries (reads tessera.weight_sharding + tessera.effect)
//   2. tessera-pipeline-stage-insertion — 1F1B stage split; insert send/recv
//      at PP stage boundaries (reads tessera.pipeline_plan on module)

// GPUCollectiveInsertionPass — inserts collective.reduce_scatter at
// data-parallel gradient boundaries and collective.all_gather at tensor-
// parallel output boundaries.  Must run after EffectAnnotationPass.
// Options:
//   --dp-axis  mesh axis for data parallelism (default "dp")
//   --tp-axis  mesh axis for tensor parallelism (default "tp")
std::unique_ptr<mlir::Pass> createGPUCollectiveInsertionPass();

// PipelineStageInsertionPass — partitions the IR into 1F1B pipeline stages
// and inserts tessera.pipeline.send / tessera.pipeline.recv ops at boundaries.
// Options:
//   --num-stages         pipeline stage count (overrides module attr)
//   --num-micro-batches  micro-batch count    (overrides module attr)
//   --interleaved        use interleaved 1F1B (default false)
std::unique_ptr<mlir::Pass> createPipelineStageInsertionPass();

// ── Phase F4 — Reverse-mode autodiff via AdjointInterface ─────────────────
//
// Walks `func.func`s annotated with `tessera.autodiff = "reverse"` in
// reverse program order and emits adjoint ops via the
// `Tessera_AdjointInterface` op trait.
//
// ODS:  src/compiler/ir/include/Tessera/AdjointInterface.td
// Body: src/transforms/lib/AutodiffPass.cpp
// Spec: docs/spec/AUTODIFF_SPEC.md §Phase F4
//
// Until tablegen on the .td runs and produces `AdjointInterface.h.inc`,
// the pass is registered as a no-op so opting into `--tessera-autodiff` in
// a pipeline doesn't break the build. The Python numpy-tape autodiff
// (`python/tessera/autodiff/`) remains the production path until then.
std::unique_ptr<mlir::Pass> createAutodiffPass();

// AdjointCollectiveInsertionPass — Phase F5. Runs **after** AutodiffPass.
// For each function argument with both a recorded cotangent (set by
// AutodiffPass via `tessera.autodiff.arg_cotangents`) and a sharding
// declaration in `tessera.weight_sharding`, plans the appropriate
// distributed-gradient collective:
//   * "dp"-sharded         → reduce_scatter on dpAxis
//   * "tp"-sharded         → all_gather on tpAxis (when the consumer needs full grad)
//   * "replicated"         → all_reduce
// The chosen plan is recorded as a per-arg `tessera.adjoint_collective_plan`
// attribute. Real op insertion follows once AutodiffPass's multi-output
// rewrite step lands.
//
// Options:
//   --dp-axis  (default "dp")
//   --tp-axis  (default "tp")
std::unique_ptr<mlir::Pass> createAdjointCollectiveInsertionPass();

// ── Phase 8.4.8 SwiGLU fusion (Stage 2b of SwiGLU Performance Plan) ───────
//
// Recognizes the 3-op SwiGLU chain
//   %gate   = tessera.matmul(%x, %W_gate)
//   %up     = tessera.matmul(%x, %W_up)
//   %hidden = tessera.silu_mul(%gate, %up)
//   %out    = tessera.matmul(%hidden, %W_down)
// and rewrites it to a single
//   %out    = tessera.swiglu_fused(%x, %W_gate, %W_up, %W_down)
//
// Backends with a fused MLP-block kernel (Apple GPU MSL — Stage 3, NVIDIA
// WGMMA epilogue, ROCm MFMA epilogue) lower the fused op directly.
std::unique_ptr<mlir::Pass> createSwigluFusionPass();

// ── attention_variants_plan, MLA-1 — DeepSeek MLA decode fusion ─────────
//
// Recognizes the chain
//   %c = tessera.latent_kv_compress(%x, %W_dkv)
//   %K = tessera.latent_kv_expand_k(%c, %W_uk)
//   %V = tessera.latent_kv_expand_v(%c, %W_uv)
//   %O = tessera.flash_attn(%Q, %K, %V)
// and rewrites it to
//   %O = tessera.mla_decode_fused(%x, %W_dkv, %W_uk, %W_uv, %Q)
// so that backends with a FlashMLA-style absorb-K kernel never have to
// materialize the full K / V matrices.
std::unique_ptr<mlir::Pass> createMLAFusionPass();

// ── attention_variants_plan, NSA-4 — DeepSeek NSA fusion ────────────────
//
// Recognizes the three-branch NSA shape (sliding_window + compressed
// _blocks + top_k_blocks all sharing the same Q) and rewrites to
//   %O = tessera.native_sparse_attn_fused(%Q, %K, %V, %gate_logits)
// for backends with a fused NSA kernel.
std::unique_ptr<mlir::Pass> createNativeSparseAttnFusionPass();

// ── attention-family plan — linear/delta/hybrid lowering slots ───────────
//
// These target-agnostic passes establish stable pipeline positions for
// Lightning Attention fusion, chunked Delta/Kimi scan lowering, and named
// hybrid-policy expansion. Current implementations are conservative
// compiler-visibility passes: they preserve SSA/dataflow and attach stable
// reasoning-family metadata, without claiming backend execution.
std::unique_ptr<mlir::Pass> createLightningAttnFusionPass();
std::unique_ptr<mlir::Pass> createDeltaAttnChunkingPass();
std::unique_ptr<mlir::Pass> createHybridAttnExpandPass();
std::unique_ptr<mlir::Pass> createLookaheadSparseAttnExpandPass();
std::unique_ptr<mlir::Pass> createLookaheadSparsePrefetchPass();
std::unique_ptr<mlir::Pass> createMSAExpandPass();

// Stage 13 — RL policy-loss compiler visibility / decomposition pass.
// PPO receives a stable primitive-form decomposition marker when it is inside
// the supported compiler envelope. GRPO/CISPO are marked compiler-visible but
// remain non-executable until they decompose or gain a runtime proof.
std::unique_ptr<mlir::Pass> createRLLossDecomposePass();

// varlen_sdpa → per-block flash_attn (Cosmos-3 two-way flat attention). Static
// decomposition when cu_seqlens are constant; runtime-lowering annotation +
// op preservation otherwise.
std::unique_ptr<mlir::Pass> createVarlenSdpaDecomposePass();

// ── Sprint V2 (2026-05-22) — LayoutLegalityPass skeleton ─────────────────
//
// Closes the "no LayoutLegalityPass" item in SHAPE_SYSTEM.md §11.2.
// First rule: `tessera.cast` ops whose `tessera.layout` attribute names
// a layout outside the canonical accept-set (row_major / col_major /
// nhwc / nchw / bhsd / tile / bsr / packed) emit
// `LAYOUT_LEGALITY_UNKNOWN_LAYOUT` and fail the pass.
//
// Future rules (declared in `LayoutLegalityPass.cpp` as comment-only
// placeholders): producer/consumer accept-set mismatches without an
// intervening cast; identity-cascade folding.
//
// Registered standalone as `--tessera-layout-legality`.  Inserted into
// the named lowering pipelines in a follow-up sprint once the rule
// surface is large enough.
std::unique_ptr<mlir::Pass> createLayoutLegalityPass();

// ── 2026-06-17 — LayoutAssignmentPass (the assignment half) ──────────────
// Seed kernel-producer layouts (matmul→row_major, flash_attn→bhsd,
// conv2d_nhwc→nhwc), propagate through pointwise ops, and insert
// `tessera.cast{tessera.layout=...}` markers at consumer accept-set boundaries.
// Paired with LayoutLegalityPass as its verifier. Registered standalone as
// `--tessera-layout-assignment`. v1 assignments are IR metadata; no backend
// consumes them yet (an IR-completeness milestone).
std::unique_ptr<mlir::Pass> createLayoutAssignmentPass();

// ── 2026-06-19 — IRContractLegalityPass (dtype / aliasing / buffer-binding) ──
// LayoutLegalityPass's sibling for the three remaining contract families in
// COMPILER_AUDIT's "Layout and binding contracts are uneven" item: dtype
// (numeric_policy storage/accum coupling + TF32-as-storage + unknown storage,
// Decision #15a), aliasing (tessera.inplace requires an in-range tessera.aliases),
// and buffer-binding (tessera.buffer_role accept-set + no conflicting role per
// tessera.binding). Registered standalone as `--tessera-ir-contracts`.
std::unique_ptr<mlir::Pass> createIRContractLegalityPass();

// ── CF0 — ControlFlowTargetGuardPass ─────────────────────────────────────
// Rejects tessera.control_{for,if,while,scan} on any backend without a
// control-flow lowering (everything but apple_gpu today) with a stable
// CONTROL_FLOW_UNSUPPORTED_ON_TARGET diagnostic (Decision #21). `target` names
// the backend in the message only. Wired into the non-Apple lowering pipelines;
// CF3/CF4 replace it with executable CUDA/ROCm control-flow kernels. See
// docs/spec/CONTROL_FLOW_CONTRACT.md §5. Standalone:
// `--tessera-control-flow-target-guard=target=<name>`.
std::unique_ptr<mlir::Pass>
createControlFlowTargetGuardPass(llvm::StringRef target = "this backend");

// ── CF2 — LowerControlFlowToSCFPass ──────────────────────────────────────
// Lowers the Graph IR bounded loop tessera.control_for to a standard scf.for
// carrying state in iter_args (body kept as a func.call) — the portable,
// hardware-free first step of the CUDA/ROCm control-flow path so the loop
// codegens as one wrapper, not one launch per iteration. The legacy all-carried
// form becomes a multi-iter_args scf.for (where pytree carries fold in).
// control_if/while → CF2b. See docs/spec/CONTROL_FLOW_CONTRACT.md. Standalone:
// `--tessera-control-flow-to-scf`.
std::unique_ptr<mlir::Pass> createLowerControlFlowToSCFPass();

// ── 2026-06-23 — TileBarrierReuseLegalityPass (C2, TIRx review) ──────────
// "Barriers are a layout-reuse correctness property." For a buffer carrying the
// C1 #tile.layout attribute, two writes to overlapping storage-axis (m/tlane/
// tcol) footprints with no intervening barrier op emit
// TILE_BARRIER_REUSE_MISSING_BARRIER. Registered standalone as
// `--tessera-tile-barrier-reuse-legality`; the acceptance gate for the typed-
// barrier + warp-spec reuse work (C3).
std::unique_ptr<mlir::Pass> createTileBarrierReuseLegalityPass();

// ── 2026-06-23 — TilePipelineLegalityPass (C3, TIRx review) ──────────────
// Cross-op companion to the #tile.pipeline_state / #tile.barrier verifiers:
// producer phase=1 / consumer phase=0 asymmetry (TILE_PIPELINE_PHASE_ASYMMETRY)
// + per-barrier-id kind consistency (TILE_PIPELINE_BARRIER_KIND_MISMATCH).
// Registered standalone as `--tessera-tile-pipeline-legality`.
std::unique_ptr<mlir::Pass> createTilePipelineLegalityPass();

// ── 2026-06-23 — C4: compute/storage dtype legalize split (TIRx review) ──
// Operationalizes Decision #15a as pass ordering. compute-legalize (early):
// reduced-precision storage without an accumulator gets numeric_policy.accum
// (fp32, or int32 for int4/int8). storage-legalize (terminal): sub-byte /
// block-scaled storage gets tessera.storage_packed + tessera.storage_container.
// Registered as --tessera-compute-legalize / --tessera-storage-legalize.
std::unique_ptr<mlir::Pass> createComputeLegalizePass();
std::unique_ptr<mlir::Pass> createStorageLegalizePass();
// 2026-06-23: the first real *consumer* of the C4 packing markers — reads
// tessera.storage_packed/storage_container + numeric_policy.storage and emits a
// concrete tessera.storage_pack = {logical, container, factor} descriptor for a
// backend's packed load/store (HF Target IR; the packed codegen + the
// legalize-dtypes default-flip are the hardware-gated tail).
// --tessera-storage-pack-consume; emits DTYPE_PACK_BAD_WIDTHS.
std::unique_ptr<mlir::Pass> createStoragePackConsumePass();

// ── 2026-06-23 — C6: warp-spec diagnostics (TIRx review) ─────────────────
// Structural warp-specialization invariants from the "Debugging Warp-Specialized
// Kernels" appendix, complementing C3's phase asymmetry: barrier-init placement
// (WARPSPEC_INIT_UNDER_GUARD), collectives outside divergent branches
// (WARPSPEC_COLLECTIVE_IN_DIVERGENT_BRANCH), producer/consumer loop-count
// agreement (WARPSPEC_LOOP_COUNT_DISAGREE), and TMA-store visibility fences
// (WARPSPEC_MISSING_VISIBILITY_FENCE). --tessera-warpspec-legality.
std::unique_ptr<mlir::Pass> createWarpSpecLegalityPass();

// ── 2026-06-23 — pipeline-parallel layer: real stage partitioning + 1F1B proof ──
// PipelineStagePartition: cost-balanced, program-order-monotonic partition of
// each function into num_stages (emits tessera.pp_stage) — the "true stage
// partitioning" the insertion pass previously required an external tagger for.
// PipelineScheduleLegality: the 1F1B schedule proof — micro-batch fill
// (Decision #17), no empty stage, forward-adjacent send/recv pairing
// (PP_MICRO_BATCHES_TOO_FEW / PP_EMPTY_STAGE / PP_SEND_WITHOUT_RECV /
// PP_RECV_WITHOUT_SEND). Registered standalone; chained in the `tessera-pipeline`
// pipeline (partition → stage-insertion → schedule-legality).
std::unique_ptr<mlir::Pass> createPipelineStagePartitionPass();
std::unique_ptr<mlir::Pass> createPipelineScheduleLegalityPass();

// ── Sprint V5 (2026-05-22) — SymbolicDimEqualityPass ─────────────────────
//
// Closes the 4th and final MLIR-verifier gap in SHAPE_SYSTEM.md §11.2:
// "No MLIR-level pass that re-checks symbolic dim equality after
// lowering."  Reads function-level `tessera.dim_bindings` (equations
// like "D = H * Dh") + `tessera.dim_sizes` (symbol → i64) and
// validates the equations.  Walks `tessera.reshape`, `tessera.transpose`,
// `tessera.matmul` ops carrying `tessera.dim_names_*` attributes and
// checks the local permutation / product / contracting-dim contracts.
//
// Stable diagnostic codes (for SHAPE_SYSTEM §11 cross-linking):
//   SYMDIM_BINDING_VIOLATION
//   SYMDIM_RESHAPE_VIOLATION
//   SYMDIM_TRANSPOSE_VIOLATION
//   SYMDIM_MATMUL_CONTRACT_VIOLATION
//
// Registered as `--tessera-symdim-equality`.  Ops without dim-name
// attributes are skipped (best-effort verifier, not a hard requirement).
std::unique_ptr<mlir::Pass> createSymbolicDimEqualityPass();

void registerTesseraPasses();

} // namespace tessera
