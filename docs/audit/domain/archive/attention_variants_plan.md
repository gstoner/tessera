---
status: Normative (development plan)
classification: Audit / Plan
authority: Sequenced plan for the four attention-variant primitives
last_updated: 2026-05-09
---

# Attention Variants — Capability Audit + Build Plan

This document evaluates Tessera's coverage of four attention variants
that show up in current research / production deployments, then sizes
the build-out for the gaps.

| Variant | Python op? | Apple GPU? | NVIDIA GPU? | Verdict |
|---------|------------|------------|-------------|---------|
| **Flash Attention** | ✅ `ops.flash_attn` | ✅ MSL kernel (Phase 8.4.1) | 🟡 IR + Tile IR (FA-4 dialect ready); execution gated on Phase G | Largely there; just needs Phase G to be real on H100 |
| **Flash Linear Attention** | ✅ `ops.linear_attn` + `_state` + `power_attn` + `retention` | ✅ MSL kernel (LA-2 — D_qk*D_v ≤ 256 envelope) | 🟡 ODS shipped; Hopper kernel = Phase G | **LA-1 + LA-2 + LA-4 landed 2026-05-09; 30 unit tests** |
| **Flash MLA Decoding** | ✅ Theme 5 + `mla_decode_fused` (MLA-1) | ✅ runtime call + host reference (MLA-2) | 🟡 ODS shipped; FlashMLA absorb-K kernel = Phase G | **MLA-1 + MLA-2 landed 2026-05-09; Schedule IR fusion + Apple GPU symbol; 2 unit tests + 2 lit fixtures** |
| **Native Sparse Attention** | ✅ 3 branch primitives + `compress_blocks` + `nn.NativeSparseAttention` | ✅ runtime call + host reference (NSA-5) | 🟡 ODS shipped; fused NSA kernel = Phase G | **NSA-1 → NSA-5 landed 2026-05-09; 15 unit tests + 1 lit fixture** |

The rest of this document spells out what each variant requires, what
exists today, what's missing, and an executable build plan.

---

## Variant 1 — Flash Attention

### Demand

Standard FlashAttention (FA-1 / FA-2 / FA-3) is the workhorse op for
every example using `tessera.ops.flash_attn` (`Diffusion_LLM`,
`Jet_nemotron`, `MultiHeadAttention` Module, etc.).

### What Tessera has today

**Python op surface:** ✅
- `ops.flash_attn(Q, K, V, *, scale=None, causal=False, cache=None,
  dropout_p=0.0, params=None, deterministic=None, seed=None)` ships at
  `python/tessera/__init__.py:551`.
- VJP registered at `python/tessera/autodiff/vjp.py:408`
  (`vjp_flash_attn`).
- `nn.MultiHeadAttention` + `nn.MultiHeadCrossAttention` Modules
  consume it.
- `nn.functional.flash_attention` alias for torch-port style.

**Graph IR ODS:** ✅
- `Tessera_FlashAttnOp` at `src/compiler/ir/TesseraOps.td:186` —
  `(q, k_or_cache, v, head_dim, numeric_policy, dropout_p, causal)`.

**FA-4 Tile IR ODS:** ✅
- `Tessera_Attn_Dialect` at
  `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td`.
- Ops: `attn.lse_save`, `attn.lse_load`,
  `attn.scaled_dot_product_tile`, `attn.online_softmax_update`,
  `attn.lse_finalize`, `attn.dropout_mask`, `attn.causal_mask`.
- AttnOps verifier at `lib/Dialect/Attn/AttnOps.cpp`.

**FA-4 Tile lowering passes:** ✅
- `WarpSpecializationPass.cpp` — producer/consumer warp roles +
  `tessera.queue` barriers.
- `AsyncCopyLoweringPass.cpp` — `tile.async_copy` → TMA / cp.async.
- `NVFlashAttnKernelEmitter.cpp` — finalizes FA-2 kernel: scale
  sentinels, mbarrier arrive/wait PTX, launch bounds, smem budget.
- `NVWGMMALoweringPass.cpp` — Hopper WGMMA `wgmma.mma_async` PTX.
- `NVTMADescriptorPass.cpp` — TMA descriptor hoist + dedup.

**Apple GPU runtime:** ✅
- `tessera_apple_gpu_flash_attn_{f32,f16,bf16}` MSL kernels
  (Phase 8.4.1 + 8.4.4.2).
- Cap `head_dim ≤ 256`; one row per threadgroup; native `half` for
  f16, fp32-conversion for bf16.

### Gaps

| Gap | What's missing | Phase |
|-----|----------------|-------|
| **NVIDIA execution** | The whole FA-4 lowering chain produces correct IR, but there's no built tessera-opt against H100/Hopper PTX → cuBin pipeline. End-to-end execution isn't wired. | **Phase G** |
| **FA-3 features** | Warpgroup-level async + producer/consumer pipelining (FA-3 is the Hopper-specific variant of FA-2). The FA-4 dialect is structurally ready; the pipeline ordering is the missing wire-up. | Phase G |
| **GQA / MQA** | `flash_attn` accepts `(B, H, S, D)` Q + same-shape K/V. No first-class support for grouped-query attention (`num_q_heads ≠ num_kv_heads`). | Item below |
| **Sliding window** | No first-class kwarg or attention mask primitive. Workable today via `ops.masked_fill` (Theme 9), but the FA-4 path doesn't fuse the window check into the score-tile loop. | Item below |
| **Sink tokens / streaming** | Same as sliding window — `examples/advanced/long_context_attention/` plans this but no compiler primitive. | Item below |
| **fp8 attention** | No fp8 path through the FA kernel; just fp16/bf16/fp32. | Phase G |

### Recommended fixes

**FA-1.1 — `flash_attn(*, num_kv_heads=None)` GQA/MQA support** (S, ~150 LOC)
- Accept `K` / `V` with `(B, num_kv_heads, S, D)` instead of the full
  `num_heads`. Repeat-broadcast inside the op (numpy reference) or
  emit a `tile.broadcast` in FA-4 lowering.
- VJP straightforward: same chain rule; sum-reduce gradient back into
  the smaller KV head dim.
- Tests: per-head correctness against the equivalent
  per-group-replicated reference.

**FA-1.2 — `flash_attn(*, window_size=None, sink_tokens=0)` sliding-
window + sink** (M, ~250 LOC)
- New mask kind in the FA-4 Attn dialect: `attn.window_mask` op
  alongside `attn.causal_mask`.
- Python op surface: kwargs `window_size: int | None`,
  `sink_tokens: int = 0`. When set, the kernel only attends to the
  most recent `window_size` keys, plus the first `sink_tokens` if
  provided (StreamingLLM pattern).
- Python reference: clamp the score matrix beyond the window /
  before sink to `-inf` then reuse softmax.
- Tests: window correctness, sink-token mass preservation, gradient
  through both regions.

**FA-1.3 — Phase G execution wire-up** (covered by Phase G in the
execution roadmap; not duplicating the plan here).

---

## Variant 2 — Flash Linear Attention — ✅ landed 2026-05-09 (Phase G items deferred)

### What it is

Linear attention rewrites attention as a recurrent state update:

    S_t = S_{t-1} + φ(K_t)^T V_t                  (state)
    O_t = φ(Q_t) @ S_t                            (output)

with feature map `φ` (typically `elu(x) + 1` or polynomial). FlashLA
adds:
- **Chunked parallel form** (training) — fold chunks of size C
  through `S` with per-chunk parallelism.
- **Causal recurrent form** (decoding) — single-token streaming
  update against running `S`.
- **Mixed-precision** — fp16 / bf16 with fp32 state accumulator.
- **Decay variants** — RetNet (decay_g), GLA, Mamba2-style selective
  decay; GroundedSSM-style position-dependent gates.

### What Tessera has today

❌ **Nothing.** No `ops.linear_attn`, no causal-state primitive.

The closest adjacent code is `examples/advanced/power_retention/` —
it has its own (third-party-style) MLIR dialect with
`power.attention` + `power.retention` ops, but it's:
- Not in `src/compiler/`
- A scaffold (Python pkg is `def version(): return ...`)
- Not registered with `tessera-opt`

The `ops.online_softmax_state` and `ops.depthwise_conv1d` (with
streaming state buffer) prove the underlying state-threading
mechanism is in place; linear attention follows the same pattern.

### Build plan — `ops.linear_attn`

**Phase A — Python op surface** (M, ~350 LOC + ~200 LOC tests)

- `ops.linear_attn(Q, K, V, *, feature_map="elu", state=None,
  chunk_size=None, decay=None, causal=True)` returns
  `(O, state_out)`.

  | Arg | Meaning |
  |-----|---------|
  | `Q, K, V` | `(B, H, S, D)` queries, keys, values |
  | `feature_map` | `"elu"` (Performer-style `elu(x)+1`) / `"relu"` (positive) / `"identity"` (RetNet) / `"polynomial_2"` (squared) — registered table |
  | `state` | optional `(B, H, D_qk, D_v)` recurrent state from a prior chunk. `None` = fresh start |
  | `chunk_size` | training: parallel-over-chunks form. `None` = pure recurrent |
  | `decay` | optional `(B, H, S)` per-token decay (RetNet/GLA/Mamba2-selective) — folded into the state update |
  | `causal` | `True` (default); non-causal linear attn folds to a single matmul |

- VJP: standard chain rule through the recurrent state. Backward
  produces `dQ`, `dK`, `dV`, optional `d_state_in`.

- Numpy-reference path: exactly the recurrence above; gives
  correctness reference for the kernel-side optimizations.

**Phase B — Apple GPU MSL kernels** (M, ~400 LOC + tests; mirrors the
flash_attn pattern)

- `tessera_apple_gpu_linear_attn_{f32,f16,bf16}` — chunk-parallel +
  recurrent variants. Per-thread loop over the state update, one
  threadgroup per `(B, H)` row.
- Lowering pass `LinearAttnToAppleGPU.cpp` mirroring
  `FlashAttnToAppleGPU.cpp`.

**Phase C — NVIDIA Tile IR ops** (L, ~600 LOC; gated on Phase G to be
real)

- New ops in `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td`:
  - `attn.feature_map_apply` (φ over a Q/K tile)
  - `attn.linear_attn_state_update` (S += φ(K)^T V, optionally
    decayed)
  - `attn.linear_attn_query` (O = φ(Q) @ S)
- Reuse `attn.lse_save`/`load` machinery for chunked training.
- Hopper kernel emitter: warp-specialize feature map + matmul into
  separate warps (matches FA-4 producer/consumer pattern).

**Phase D — Promote `power_retention` example dialect into
`src/compiler/codegen/Tessera_Power_Backend/`** (S, ~150 LOC; only
after Phase A unblocks the Python surface). This makes the existing
retention research kernels addressable by `@jit(target=...)`.

**Acceptance criteria**

- Recurrent form: `linear_attn(Q, K, V, feature_map="elu",
  causal=True)` matches the explicit `S_t = S_{t-1} + φ(K_t)^T V_t`
  reference at fp64.
- Chunked form: same numerical result regardless of `chunk_size`
  (chunked-parallel must be bit-equivalent to recurrent at fp64).
- Decay: `linear_attn(..., decay=g)` matches the exponential-decay
  reference (RetNet equation 4) at fp64.
- VJP: numerical-Jacobian agreement to 1e-5 at fp64.

**Estimated runway:** Python op surface + tests = 1–2 weeks; Apple
GPU MSL = 1 week; NVIDIA Tile IR ops gated on Phase G.

---

## Variant 3 — Flash MLA Decoding — ✅ MLA-1 + MLA-2 landed 2026-05-09

### What it is

DeepSeek's Multi-Latent Attention with FlashMLA — a Hopper-tuned
kernel that fuses:

1. Latent compress: `c = x @ W_dkv` (the "down" projection)
2. Cache the latent (paged storage at `(seq, latent_dim)` instead of
   `(seq, num_heads, head_dim)`)
3. Absorbed-K decoding: at score time, compute `Q @ (W_uk^T) @ c^T`
   without ever materializing the full K matrix — `W_uk` is absorbed
   into the score accumulator.
4. Same trick for V: `softmax(scores) @ c @ W_uv`.

The KV cache memory drop is ~93% (reported by DeepSeek-V3) because
`latent_dim` ≪ `num_heads * head_dim`.

### What Tessera has today

**Python op surface (Theme 5, landed 2026-05-09):** ✅
- `ops.latent_kv_compress(x, W_dkv)` — distinct op_name anchors the
  FlashMLA target match.
- `ops.latent_kv_expand_k(c, W_uk)` / `latent_kv_expand_v(c, W_uv)` —
  the eventually-absorbed expansion.
- `ops.rope_split` / `ops.rope_merge` — decoupled-RoPE pattern.
- `tessera.cache.LatentKVCacheHandle(latent_dim, max_seq, ...)` —
  paged latent storage, sliding-window via `auto_evict=True`.
- 20 unit tests in `test_mla_primitives.py`.

### Gaps

| Gap | What's missing | Phase |
|-----|----------------|-------|
| **Absorb-K target pass** | The `ops.latent_kv_expand_k` op exists, but no IR pass *fuses* it with the score-matrix computation so K is never materialized. The current path materializes the full `K = c @ W_uk` then runs flash_attn — defeats the memory win. | Item below |
| **Decoding kernel** | The expansion is per-decode-step; today's `ops.flash_attn` is a per-call op, no streaming K/V from a `LatentKVCacheHandle`. The "decoding" in FlashMLA Decoding refers to the per-token loop that reuses the latent state. | Item below |
| **NVIDIA Hopper FlashMLA kernel** | The reference path runs on numpy (correctness only). The actual win — 93% memory drop, 660 TFLOPS reported — needs the absorb-K fusion + Hopper-specific WGMMA layout. | **Phase G** |
| **Per-backend (Apple GPU)** | Could ship an Apple GPU MSL kernel that does the absorb at the per-thread-stack-array level — similar to the existing 3-op `matmul→softmax→matmul` fusion. | Item below |

### Build plan

**MLA-1 — Schedule IR `mla_decode_fused` pattern recognizer** (M,
~200 LOC + lit fixture)
- New rewrite in `src/transforms/lib/MLAFusionPass.cpp` mirroring
  `SwigluFusionPass.cpp` (already shipped for SwiGLU).
- Pattern: `latent_kv_compress → cache_append (state) →
  latent_kv_expand_k → flash_attn → ...` — collapse to
  `tessera.mla_decode_fused`.
- ODS: new `Tessera_MLADecodeFusedOp` in `TesseraOps.td` carrying
  `(x, W_dkv, W_uk, W_uv, latent_cache, ...)`.
- Lit fixture under `tests/tessera-ir/phase8/mla_decode_fusion.mlir`.

**MLA-2 — Apple GPU MSL kernel** (M, ~350 LOC + tests; mirrors the
SwiGLU 3-op chain pattern)
- `tessera_apple_gpu_mla_decode_f32` — fused MSL kernel that:
  1. Loads `Q` (B*H, D_q) into thread registers.
  2. Loads the latent cache slice into threadgroup memory.
  3. Computes `Q @ (W_uk^T)` → `Q_proj` (compute the projection
     once per query).
  4. Accumulates `score = Q_proj @ c^T` per cached token.
  5. Online softmax + accumulate `score @ c @ W_uv` per token.
- New pass `MLADecodeFusionToAppleGPU.cpp`.
- Cap `latent_dim ≤ 512`, `num_heads * head_dim ≤ 1024` for the
  per-thread-stack-array path.

**MLA-3 — NVIDIA Hopper FlashMLA kernel** (XL, Phase G material)
- New target dialect `Tessera_FlashMLA_NVIDIA` mirroring the existing
  FA-4 Tile IR.
- Absorbed-K layout: store `W_uk` in shared memory once per kernel
  launch, broadcast across all warps.
- WGMMA pipelining of `Q @ W_uk^T @ c^T` as a fused triple-product.
- Decoding-mode: persistent CTA scheduler for autoregressive
  decoding loop (one CTA = one batch element across multiple decode
  steps).

**Acceptance criteria (MLA-1 + MLA-2)**

- A `@jit(target="apple_gpu")` function whose body is the MLA pattern
  produces the fused MSL kernel call (verified via lit + unit test).
- End-to-end output matches the explicit numpy reference at rtol=1e-4.
- Memory profile shows the latent cache, *not* the full K/V — the
  goal is to validate this even on the Apple GPU path before Phase G.

**Estimated runway:** MLA-1 + MLA-2 = 2 weeks. MLA-3 (Hopper) is
Phase G material.

---

## Variant 4 — Native Sparse Attention — ✅ NSA-1 → NSA-5 landed 2026-05-09

### What it is

DeepSeek's "Native Sparse Attention" (NSA, paper Jan 2025) — a
hardware-aware sparse attention pattern that's *natively* trainable
end-to-end (unlike post-hoc sparsification like H2O). The structure:

1. **Sliding-window branch** — every Q attends to the most recent
   `W` keys (dense local context).
2. **Compressed-block branch** — chunks of `C` consecutive keys are
   summarized via a per-chunk linear projection `c_chunk = K_chunk @
   W_compress`; queries attend to the chunk summaries.
3. **Selected-block branch** — top-k blocks are selected per query
   via a learnable scoring head, and the queries attend to the full
   tokens of those selected blocks.
4. **Gating** — a learned per-query gate decides how much weight goes
   to each branch (sliding / compressed / selected).

The "native" part: all three branches are jointly trainable, and the
selection step is differentiable (straight-through or Gumbel-softmax
on the top-k). Reported: ~2-3× decode speedup at long context with
no quality regression.

### What Tessera has today

❌ **Nothing.** No `ops.native_sparse_attn`, no block-selection
primitive, no compressed-block summary op.

Adjacent infrastructure that helps:
- `ops.gather` (Theme 9, 2026-05-09) — for selected-block lookups.
- `ops.masked_fill` (Theme 9) — for sliding-window mask.
- `ops.softmax` + numerical-stability machinery — reusable.
- Sparse linalg ops (`spmm_coo`, `spmm_csr`, `sddmm`, `bsmm`,
  `segment_reduce`) — could back the compressed-block branch.

### Build plan

**NSA-1 — Branch primitives** (M, ~400 LOC + ~250 LOC tests)

Three new ops in `python/tessera/__init__.py`:

```python
ops.attn_sliding_window(Q, K, V, *, window_size, causal=True)
ops.attn_compressed_blocks(Q, K_chunks_compressed, V_chunks_compressed)
ops.attn_top_k_blocks(Q, K, V, *, scores, top_k, block_size)
```

Each has:
- Numpy reference (clear correctness baseline).
- VJP — straightforward except for `attn_top_k_blocks` whose
  selection is non-differentiable. Use straight-through estimator
  (gradient flows through the selected blocks; the `argmax` is
  treated as identity for the backward).
- Tests against the explicit numpy reference.

**NSA-2 — Block compression op** (S, ~120 LOC)

```python
ops.compress_blocks(K, V, *, block_size, W_compress) → (K_c, V_c)
```

Chunk K/V into `block_size`-sized groups, project each group through
`W_compress` to a single summary token. VJP via standard chain rule.

**NSA-3 — `nn.NativeSparseAttention` Module** (S, ~150 LOC + ~80
LOC tests)

Wraps the three branches + gating:

```python
class NativeSparseAttention(Module):
    def __init__(self, dim, num_heads, window_size=512, block_size=64,
                 num_compressed=128, top_k=16): ...
    def forward(self, x):
        Q, K, V = qkv_projection(x)
        # Branch 1: sliding window
        out_w = ops.attn_sliding_window(Q, K, V, window_size=...)
        # Branch 2: compressed
        K_c, V_c = ops.compress_blocks(K, V, block_size=..., W_compress=self.W_compress)
        out_c = ops.attn_compressed_blocks(Q, K_c, V_c)
        # Branch 3: top-k selected
        block_scores = self.scorer(Q, K_c)
        out_s = ops.attn_top_k_blocks(Q, K, V, scores=block_scores,
                                       top_k=..., block_size=...)
        # Gating
        gate = sigmoid(self.gate_proj(x))   # (B, S, 3)
        return gate[..., 0:1] * out_w + gate[..., 1:2] * out_c + gate[..., 2:3] * out_s
```

**NSA-4 — Schedule IR fusion** (M, ~250 LOC + lit fixture)

The three-branch shape is amenable to Schedule IR fusion: a single
`tessera.native_sparse_attn_fused` op carrying all the branch
parameters can be matched and lowered to a backend-specific kernel.
Mirrors the SwiGLU + MLA fusion patterns.

**NSA-5 — Apple GPU MSL kernel** (M, ~400 LOC; can ship after
NSA-1+2+3 unblock the Python surface)

One MSL kernel that does all three branches in a single dispatch —
reuses the threadgroup memory for the sliding window, computes
compressed scores in registers, and uses `simdgroup_*` reductions
for the top-k selection. Cap `block_size ≤ 64`, `num_blocks ≤ 256`
for the per-thread-stack-array variant.

**NSA-6 — NVIDIA Hopper kernel (Phase G)**

Block-sparse `wgmma` patterns for the selected-block branch are
already covered by NVIDIA's existing block-sparse cuBLAS / cuSparse
support; Tessera's role is to emit the right `tile.bsmm` calls.

### Acceptance criteria

- `attn_sliding_window` matches a numpy `flash_attn` with explicit
  `masked_fill` window mask.
- `attn_compressed_blocks` matches a numpy reference where queries
  see the per-chunk averaged tokens.
- `attn_top_k_blocks` selection: backward gradient flows only through
  the selected blocks (verified by zeroing one selected block and
  confirming its gradient drops to zero).
- `NativeSparseAttention` Module trains a small toy LM (1-layer,
  64-token sequence) — gating learns to favor the sliding-window
  branch on local-dependency tasks and the selected-block branch on
  long-range dependency tasks.

**Estimated runway:** NSA-1 through NSA-3 = 2 weeks. NSA-4 + NSA-5 =
2 more weeks. NSA-6 is Phase G material.

---

## Phase summary (at-a-glance)

| # | Item | Scope | Wks | Independent? | Demand |
|---|------|-------|-----|--------------|--------|
| FA-1.1 | `flash_attn(num_kv_heads=...)` GQA/MQA | S | 0.5 | ✅ | High — every modern LLM (Llama, Mistral, GPT-OSS) uses GQA |
| FA-1.2 | `flash_attn(window_size=, sink_tokens=)` | M | 1 | ✅ | Medium — `long_context_attention` example needs it |
| FA-1.3 | Phase G NVIDIA execution | XL | 4–8 | Phase G | Highest — unblocks every GPU example |
| LA-1 | `ops.linear_attn` Python op + VJP | M | 1–2 | ✅ | Growing — RetNet / Mamba2-linear / Lightning-attention research |
| LA-2 | Apple GPU MSL kernel for linear_attn | M | 1 | depends on LA-1 | Same |
| LA-3 | NVIDIA Tile IR + Hopper kernel | L | 3–4 | Phase G | Same |
| LA-4 | Promote `power_retention` dialect | S | 0.5 | depends on LA-1 | Niche today |
| MLA-1 | Schedule IR `mla_decode_fused` recognizer | M | 1 | ✅ | Theme 5 already shipped Python ops; this is the IR-side win |
| MLA-2 | Apple GPU MSL MLA decode kernel | M | 1 | depends on MLA-1 | Demonstrates absorb-K on a real (Apple) backend |
| MLA-3 | Hopper FlashMLA kernel | XL | 4–6 | Phase G | DeepSeek-V3-style serving |
| NSA-1 | Three branch primitives | M | 1.5 | ✅ | Cutting-edge — if it lives up to the paper, the whole industry switches |
| NSA-2 | `compress_blocks` op | S | 0.5 | depends on NSA-1 | Same |
| NSA-3 | `nn.NativeSparseAttention` Module | S | 0.5 | depends on NSA-1+2 | Same |
| NSA-4 | Schedule IR fusion + lit fixture | M | 1 | depends on NSA-3 | Same |
| NSA-5 | Apple GPU MSL kernel | M | 1.5 | depends on NSA-3+4 | Same |
| NSA-6 | NVIDIA Hopper kernel | L | 3–4 | Phase G | Same |

**Recommended ordering:**

1. **Easy wins (1–2 weeks):** FA-1.1 (GQA) + FA-1.2 (sliding window
   + sink). Both unblock real example code that's already in-tree.
2. **Linear attention (3–4 weeks):** LA-1 + LA-2 + LA-4. Adds a
   research-relevant op family (RetNet, Mamba2-linear, Lightning) that
   Tessera currently has zero coverage of.
3. **MLA decode fusion (2 weeks):** MLA-1 + MLA-2. Demonstrates
   absorb-K end-to-end on Apple GPU before Phase G ships the Hopper
   kernel.
4. **Native Sparse Attention (4–5 weeks):** NSA-1 → NSA-5. Largest
   single new feature surface; do this when there's real research
   intent to use it.
5. **Phase G items in parallel** (FA-1.3, LA-3, MLA-3, NSA-6) —
   gated on the Phase G NVIDIA execution unblock that's already on
   the execution roadmap.

**Total addressable runway:** ~12–15 weeks of focused work end-to-end
across all four variants (excluding Phase G GPU kernels). The
recommended near-term sprint (FA-1.1 + FA-1.2) is ~1.5 weeks and
visibly improves coverage.

---

## Cross-references

- `docs/audit/roadmap/ROADMAP_AUDIT.md` — Phase G is the long pole for
  all native GPU kernel paths
- `docs/audit/coverage/COVERAGE_AUDIT.md` — Theme 5 (MLA
  primitives) is shipped; this plan extends it with the absorb-K
  fusion piece
- `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td` —
  the FA-4 Tile IR ODS that variants 2 + 3 + 4 will all extend
- `examples/advanced/long_context_attention/` — sliding-window /
  sink-token example that FA-1.2 directly unblocks
- `examples/advanced/mla/flashmla_tessera.md` — DeepSeek MLA design
  notes; MLA-3 follows this
- `examples/advanced/power_retention/include/tessera/power/TesseraPower.td` —
  existing third-party-style dialect for retention (`Power_AttnOp` +
  `Power_RetentionOp`); LA-4 promotes this to first-class
