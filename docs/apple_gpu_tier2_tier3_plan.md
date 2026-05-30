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
| **1** | **`bmm` (batched / rank-3 matmul)** — keystone | MPSGraph `matrixMultiplication` handles leading batch dims **and broadcasting** (needed for GQA `[B,H,M,K] @ [B,1,K,N]`). New `tessera_apple_gpu_bmm_{f32,f16,bf16}` symbols with a `b_broadcast` flag; reuse the cached-graph infra + buffer pool. Keep the rank-2 MPS fast path. | Med · low risk |
| 2 | **`qkv_projection`, `linear_general`** | Thin dispatchers: reshape contracted axes → `bmm`/`matmul` → reshape. `qkv_projection` = one fused proj + host split. | Low |
| 3 | **`multi_head_attention`** | Extend the existing `matmul_softmax_matmul` fused kernel with a **batch dim** (per-head = batch) → one dispatch instead of the MLA-proof's per-head loop. | Med |
| 4 | **`gqa_attention`, `mqa_attention`** | flash_attn is already batched over `B`; add a **`num_kv_heads`/`group_size`** param so query head `h` reads KV group `h // (H/G)` (MQA = 1 KV head). **Ph1:** broadcast KV via `bmm` (correctness-first). **Ph2:** native KV-group indexing in the MSL kernel (bandwidth win). | Med (Ph1 Low) |
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
