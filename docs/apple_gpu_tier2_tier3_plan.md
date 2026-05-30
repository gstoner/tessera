# Apple GPU — Tier 2 / Tier 3 implementation plan

> Status: planning + in-progress. Tier-1 (activations / norms / SwiGLU gate via
> the MetalPerformanceShadersGraph lane) and the f16/bf16 fused-chain matrix are
> shipped (see [`apple_gpu_overview.md`](apple_gpu_overview.md)). This doc tracks
> the next two tiers. **Batched matmul (`bmm`) is the keystone — it is being
> implemented first** because every Tier-2 attention/projection item depends on it.

## Grounding facts (verified in-tree)

- **All Tier-2 attention ops already have ODS ops**: `flash_attn`, `gqa_attention`,
  `mqa_attention`, `multi_head_attention`, `mla_decode`, `mla_decode_fused`,
  `matmul`. Not yet registered: `gemm`, `batched_gemm`, `qkv_projection`,
  `linear_general`.
- **`flash_attn_f32(Q,K,V,O, B, Sq, Sk, D, scale, causal)` is already batched over
  `B`** — so GQA/MQA is a *localized* extension (KV-head grouping), not a new kernel.
- **General batched matmul is the one true gap** on the GPU side (only flash_attn
  is batched today; `tessera.matmul` is rank-2 / MPS).
- Lowering-pass scaffold already exists for the attention family (FlashAttn,
  LinearAttn, MLADecodeFusion, NativeSparseAttn). Most Tier-2 work is therefore
  *runtime kernels + dispatchers*, not new IR.
- Runtime execution is **Python-metadata-driven** (the per-op `op_name` dispatch
  loop), so an op runs on GPU once it has: (1) a `.mm` C ABI symbol, (2) a
  `runtime.py` dispatcher + ctypes wrapper + envelope entry, (3) `driver.py`
  gating. An MLIR lowering pass + ODS op is only needed for the compile-time /
  lit artifact path.

## Tier 2 — high value, medium effort (dependency-ordered)

| Order | Op(s) | Approach | Effort |
|---|---|---|---|
| **1 ✅** | **`bmm` (batched / rank-3 matmul)** — keystone | **DONE** (`tessera_apple_gpu_bmm_{f32,f16}` + bf16 host-upcast, with a `b_broadcast` flag; reuses the cached-graph infra + buffer pool; rank-4+ folds to batch). `tests/unit/test_apple_gpu_bmm.py`. | Med · low risk |
| 2 ✅ | **`qkv_projection`, `linear_general`** | **DONE** — `runtime.py` dispatchers route `tessera.linear_general` (last-axis `x@W(+bias)`) and `tessera.qkv_projection` (`x@W` then split-3) through the matmul/`bmm` lane; `_APPLE_GPU_PROJECTION_OPS` gating → `metal_runtime`. `tests/unit/test_apple_gpu_projections.py`. | Low |
| 3 ✅ | **`multi_head_attention`** | **DONE (via bmm composition)** — a full MHA block composes from the Tier-2 ops: `qkv_projection → bmm(Q,Kᵀ)·scale → softmax → bmm(_,V) → linear_general`, batch = B·H (no per-head loop), and head_dim is unbounded (no flash_attn ≤256 limit). `tests/unit/test_apple_gpu_batched_mha.py`. A single fused batched `matmul_softmax_matmul` kernel remains a perf follow-up. | Med |
| 4 | **`gqa_attention`, `mqa_attention`** | flash_attn is already batched over `B`; **GQA/MQA Phase-1 (repeat-KV → flash_attn) is already implemented in `nn.functional`**. Remaining: native KV-group indexing in the MSL kernel (bandwidth win, Ph2). | Med (Ph1 done) |
| 5 | **`mla_decode` / `mla_decode_fused`** | Promote the host-reference `mla_decode_f32` symbol to a real kernel built on `bmm` (latent up-proj / absorb-K) + the existing `rope` kernel (decoupled rope) + batched attention. Phase: explicit-KV → compressed-KV → decoupled-rope. | Med–High (last) |

## Tier 3 — re-scoped around MPSGraph (the lever)

MPSGraph nodes cover most of what the original framing assumed needed bespoke MSL.

| Item | Mapping | Effort | Recommendation |
|---|---|---|---|
| **Reductions** (`sum`/`mean`/`var`/`std`/`amax`/`amin`/`prod`/`argmax`/`argmin`/`cumsum`/`cumprod`) | Extend the MPSGraph lane with an op-coded reduce runner — the lane already uses `meanOfTensor:axes:`, `reductionSumWithTensor:axes:`, `reductionMaximumWithTensor:`. `cumsum`→`cumulativeSumWithTensor:axis:`, `argmax`→`reductionArgMaximumWithTensor:axis:`. No N limit; reuses the cached-graph infra. | Low–Med | **Do it** (high reuse) |
| **`dropout`, `rng_normal`, `rng_uniform`** | (a) MPSGraph random nodes — quick but **won't bit-match the CPU Philox stream** (breaks Decision #18); (b) hand-written Philox MSL for bit-exactness. | Med / Low | **Defer** (training-side); MSL if pursued |
| **`conv2d` / `conv3d`** | MPSGraph `convolution2DWithSourceTensor:weightsTensor:descriptor:` (full stride/pad/dilation); conv3d → im2col + `bmm` fallback. | Med / High | **Defer** unless vision models enter scope |

## Cross-cutting

- **MPSGraph is the recurring lever** — extend the existing op-coded lane rather
  than write bespoke MSL for reductions/conv. Exception: RNG, where
  bit-reproducibility argues for hand-written Philox MSL.
- **`bmm` first** — low risk (MPS/MPSGraph native), unblocks all of Tier 2.
- **Reuse the proven seams**: the 4-artifact pattern (kernel source + C ABI symbol
  + lowering pass + Python dispatcher), the MPSGraph cached-graph infra + buffer
  pool, the null-handling convention, and the per-op driver gating sets.
- **Testing**: each primitive gets a focused, numpy-validated lane test with a
  `metal_runtime` gate; the attention family gets a model-shaped E2E like
  `test_apple_gpu_llama_decoder_layer.py` (e.g. a GQA decoder layer).

## Suggested first sprint

`bmm` (f32/f16/bf16) → `qkv_projection` / `linear_general` → MPSGraph reductions.
Three low/medium, low-risk items that unblock the attention family and the
reduction long-tail, with no RNG/conv complexity.
