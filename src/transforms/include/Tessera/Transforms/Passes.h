
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

// TileToX86Pass — replaces tiled tessera.matmul ops (static bf16 operands)
// with calls to the tessera_x86_backend C functions via func.call with raw
// i64 pointer arguments.  Handles fused_epilogue bias/gelu variants.
// Options:
//   --prefer-amx  prefer AMX over AVX-512 when both are available (default true)
std::unique_ptr<mlir::Pass> createTileToX86Pass();

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
// tessera.tma.* (SM≥90) or tessera.cp_async.* (SM<90).
// Options:
//   --sm  target SM version (default 90)
std::unique_ptr<mlir::Pass> createAsyncCopyLoweringPass();

// NVWGMMALoweringPass — lowers tile.mma to wgmma.mma_async PTX inline asm
// for SM_90+ or nvgpu.mma.sync WMMA for SM<90.
// Options:
//   --sm  target SM version (default 90)
std::unique_ptr<mlir::Pass> createNVWGMMALoweringPass();

// NVTMADescriptorPass — hoists tessera.tma.descriptor ops to the kernel
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

void registerTesseraPasses();

} // namespace tessera
