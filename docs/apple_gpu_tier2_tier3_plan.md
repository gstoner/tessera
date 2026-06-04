# Apple GPU — Tier 2 / Tier 3 implementation plan

> Status: planning + in-progress. Tier-1 (activations / norms / SwiGLU gate via
> the MetalPerformanceShadersGraph lane) and the f16/bf16 fused-chain matrix are
> shipped (see [`apple_backend.md`](apple_backend.md)). This doc tracks
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
| 3 ✅ | **`multi_head_attention`** | **DONE** — composes via bmm (`tests/unit/test_apple_gpu_batched_mha.py`) **and** the fused single-dispatch `bsmm` kernel below (`tests/unit/test_apple_gpu_fused_attention.py`). Both batch = B·H, head_dim unbounded. | Med |
| 3b ✅ | **fused batched `matmul_softmax_matmul`** | **DONE** — `tessera_apple_gpu_mpsgraph_bsmm_{f32,f16}`: `softmax((Q@Kᵀ)·scale)@V` per batch in one MPSGraph dispatch (vs the 3-call compose). | Med |
| 4 ✅ | **`gqa_attention`, `mqa_attention`** | **DONE** — Ph1 (repeat-KV → flash_attn) already in `nn.functional`; **Ph2 native KV-group indexing landed** (`flash_attn_gqa_f32` reads KV group `h//(H/G)` with no repeated KV — the bandwidth win); **Ph3 native f16/bf16** (`flash_attn_gqa_f16` native `half` I/O + fp32 accum; `flash_attn_gqa_bf16` fp32-conversion kernel; uint16 ABI, no host f32 round-trip). `tests/unit/test_apple_gpu_gqa.py`. | Med (done) |
| 5 ✅ | **`mla_decode`** | **DONE — compressed-KV + decoupled-RoPE.** (a) *Compressed-KV*: `mla_decode_f32` promoted to an on-GPU cached MPSGraph fusing `c = X@Wdkv` → `K = c@Wuk` / `V = c@Wuv` → `softmax((Q@Kᵀ)·scale)@V` (B·S_q folded to rows, KV shared across batch). `tests/unit/test_apple_gpu_mla_decode_gpu.py` (6) + 30 existing MLA tests. (b) *Decoupled-RoPE* (DeepSeek-style): new `mla_decode_rope_f32` — each head's q/k splits into no-position (`dn`) + RoPE (`dr`) parts with the key-RoPE **shared across heads**; RoPE (switchable interleaved/half) + `[nope;rope]` concat assembled on host, the resulting standard MHA (head_dim `dn+dr`) runs on-GPU via the fused `bsmm` kernel. Explicit per-head K; host reference fallback. `tests/unit/test_apple_gpu_mla_decoupled_rope.py` (12, both styles) + `benchmarks/apple_gpu/benchmark_mla_decode.py`. (c) *Weight absorption* (**the real bandwidth win**): new `mla_absorb_decode_f32` — the up-projection weights absorb into the query/output (`q_abs = q_nope@Wukᵀ`, `s_nope = q_abs@c_kvᵀ`, `ctx = attn@c_kv`, `O = ctx@Wuv`) so attention runs directly against the cached latent and **per-head K/V are never materialized**; the KV cache stores only `c_kv [Skv,Dl]` + shared `k_rope [Skv,dr]` (~8.9× smaller for DeepSeek-V2 dims). One cached MPSGraph; mathematically identical to the explicit path (cross-checked in tests). `tests/unit/test_apple_gpu_mla_weight_absorb.py` (14, incl. incremental KV-cache decode) + `benchmarks/apple_gpu/benchmark_mla_absorb.py`. **Native f16/bf16 landed:** `mla_absorb_decode_{f16,bf16}` — f16 carries f16 I/O on-GPU (half the cache-read bandwidth) with fp32 accumulation via `mpsg_up`; bf16 via host fp32 round-trip; the Python dispatcher routes by `q_nope.dtype`. `tests/unit/test_apple_gpu_mla_absorb_dtypes.py` (8, both styles). (d) *Paged-cache wiring* (production serving): `tessera.cache.MLAPagedDecoder` bundles two `LatentKVCacheHandle`s (compressed latent + shared rope key slice) and drives the absorbed GPU kernel through a real prefill→step decode loop, with absolute-position tracking so sliding-window/eviction stays RoPE-correct and a numpy fallback for portability. `tests/unit/test_mla_paged_decoder.py` (9). (e) *Multi-sequence block-table batching* (vLLM-style paged attention): `tessera.cache.MLABlockPagedCache` manages many concurrent sequences over a single physical block pool — per-sequence block tables, on-demand block allocation from a free list, free-on-finish page reuse (no external fragmentation), non-contiguous block-table indirection, and a ragged `decode_batch`. Cross-checked against `MLAPagedDecoder`. `tests/unit/test_mla_block_paged_cache.py` (11). **Same-length `B>1` batching landed:** `decode_batch` groups sequences by cached length (shared RoPE positions) and dispatches each group in a single `B = group_size` kernel call (`absorb_decode_batch`) instead of looping — cross-checked against per-sequence decode. **Full f16/bf16 dtype matrix complete:** all three MLA decode kernels now ship `{f32,f16,bf16}` — `mla_decode_{f16,bf16}` (compressed-KV) and `mla_decode_rope_{f16,bf16}` (explicit-K, via the dtype-generic `bsmm`) joined the absorbed kernel; f16 native I/O + fp32 accumulation, bf16 host round-trip, dispatchers route by input dtype. `tests/unit/test_apple_gpu_mla_secondary_dtypes.py` (9). | Med–High |

## Tier 3 — re-scoped around MPSGraph (the lever)

MPSGraph nodes cover most of what the original framing assumed needed bespoke MSL.

| Item | Mapping | Effort | Recommendation |
|---|---|---|---|
| **Reductions** (`sum`/`mean`/`var`/`std`/`amax`/`amin`/`prod`/`argmax`/`argmin`/`cumsum`/`cumprod`) | **DONE** — `tessera_apple_gpu_mpsgraph_{reduce,argreduce,scan}_f32`; `runtime.py` normalizes arbitrary axis/keepdims/ddof by folding reduced axes to the last dim. `tests/unit/test_apple_gpu_reductions.py` (51). | Low–Med | **Done** |
| **`dropout`, `rng_normal`, `rng_uniform`** | (a) MPSGraph random nodes — quick but **won't bit-match the CPU Philox stream** (breaks Decision #18); (b) hand-written Philox MSL for bit-exactness. **Inference sampler shipped** (separate from training RNG): `tessera_apple_gpu_gumbel_argmax_f32` + `runtime._apple_gpu_gumbel_sample` — Gumbel-max categorical draw `argmax(logits/T + g)` with the Gumbel noise taken from the canonical Philox stream (so it's deterministic / reproducible / **#18-safe**, no on-GPU RNG), supporting temperature / greedy / top-k / top-p. The per-row vocab argmax runs on-GPU; **honest benchmark finding (`benchmark_gumbel_sampler.py`): it is upload-bound today — host numpy argmax is faster until the logits stay GPU-resident (fully-fused decode) or a Philox-MSL noise generator removes the noise upload.** `tests/unit/test_apple_gpu_gumbel_sampler.py` (10, incl. a 20k-sample distribution-convergence check). | Med / Low | **Defer** (training-side); MSL if pursued |
| **`conv2d`** | **DONE** — `tessera_apple_gpu_conv2d_{f32,f16}` via MPSGraph `convolution2DWithSourceTensor:weightsTensor:descriptor:` (NHWC source / HWIO weights, full stride/pad/dilation/groups, optional bias, fp32 internal accumulation; bf16 via host fp32 round-trip). Wired into the metadata op-loop + `_APPLE_GPU_CONV_OPS` envelope + driver gating; reference fallback in the stub. `tests/unit/test_apple_gpu_conv2d.py` (13). | Med | **Done** |
| **`conv3d`** | **DONE** — MPSGraph has no 3-D conv node, so `tessera_apple_gpu_conv3d_{f32,f16}` lower to im2col + a single GPU MPSGraph **batched matmul** (batch = groups, fp32 accumulation): patches gathered to a per-group `[groups, rows, K]` column matrix, weights regrouped to `[groups, K, Cout/groups]`, the dominant GEMM on-GPU; bias + scatter on the host. NDHWC source / DHWIO weights, full stride/pad/dilation/groups, optional bias; bf16 via host fp32 round-trip. Wired into the metadata op-loop + `_APPLE_GPU_CONV_OPS` envelope + driver gating; reference fallback in the stub. `tests/unit/test_apple_gpu_conv3d.py` (12). | High | **Done** |

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
