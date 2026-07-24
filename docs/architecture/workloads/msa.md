---
classification: Architecture / Workload
authority: MSA workload design and implementation guide
last_updated: 2026-07-13
---

# MSA — MiniMax Sparse Attention

MiniMax Sparse Attention ([MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA),
arXiv:2606.13392) is a blockwise sparse attention built on Grouped Query
Attention (GQA) for ultra-long context. A lightweight **Index Branch** scores
key–value blocks and independently selects a Top-`k` subset **for each GQA
group**; the **Main Branch** then performs *exact* block-sparse attention over
only the selected blocks. On a 109B model the authors report attention compute
reduced 28.4× at 1M context, and a co-designed H800 kernel delivers 14.2×
prefill / 7.6× decode wall-clock speedups.

This page documents Tessera's implementation status, the shape contract, the
public API, and how MSA differs from the existing
[`deepseek_sparse_attention`](../../CANONICAL_API.md) (NSA) path.

> **Status (2026-06-13).** Phase 0 (contract) + Phase 1 (reference numpy API) +
> Phase 2 (Graph IR op contract) + Phase 3 (Apple GPU host-select lane) landed.
> The reference path runs on CPU and is dense-equivalent when
> `top_k == num_blocks`. The three MSA Graph IR ops are ODS-registered with
> verifiers and a compiler-visibility pass (`tessera-opt` builds clean;
> lit-proven). **`@jit(target="apple_gpu")` executes MSA on the GPU today**: the
> Index Branch scoring + Top-k block selection run on the host (data-dependent,
> like `attn_top_k_blocks`), and the exact block-sparse Main Branch runs on Metal
> (two GPU bmms + GPU softmax), validated bit-for-bit against the numpy reference
> at fp32 tolerance. **Phase 6** then landed a **native fused block-sparse flash
> MSL kernel** (`tessera_apple_gpu_msa_block_sparse_f32`): one Metal dispatch,
> KV-outer over the selected blocks, online softmax — no host gather, no MPS-bmm
> round-trips. It is the default `@jit(target="apple_gpu")` path for `D ≤ 256`
> (composed bmm path remains the fallback), and its latency scales ~linearly with
> `top_k` — a measured ~10.6× wall-clock reduction at `top_k=1` (6% coverage) vs
> the dense-equivalent on this M1 Max. The exp-free Top-k *selection* still runs
> on the host (a small data-dependent step). The CUDA/NVIDIA path now has a
> Python Graph→Schedule→Tile→Target artifact lowering for the KV-outer
> selected-block contract, while the native CUDA/H800/Blackwell kernel remains
> future work. **Phase 4**
> (`nn.MinimaxSparseAttention` GQA layer + `from_gqa` conversion) and **Phase 5**
> (the `examples/attention/minimax_sparse_attention.py` dense-vs-sparse example +
> theoretical-compute benchmark) landed. All six phases (0–5) are complete on the
> reference + Apple GPU host-select surface; the native fused kernel and CUDA
> path remain the only open frontier.

---

## 1. Shape contract

| Tensor | Shape | Notes |
|---|---|---|
| `Q` | `(B, Hq, Sq, D)` | query heads |
| `K`, `V` | `(B, Hkv, Sk, D)` | KV heads; `Dv` may differ for `V` |
| GQA constraint | `Hq % Hkv == 0` | group size `g = Hq // Hkv`; query head `h` ∈ group `h // g` |
| block constraint | `Sk` may be non-divisible in the reference/runtime path | K/V are padded internally; static MLIR verifier paths may still require known-divisible blocks |

**Block metadata:** `block_size`, `top_k` (`1 ≤ top_k ≤ num_blocks`),
`force_local_block` (default `True`), `causal` (default `True`).

**Selection granularity:** per query token, **per GQA group** (shared by every
query head in the group — *not* per head), block-level KV ids. Query heads in a
group are mean-pooled into one group query for scoring; each KV block is
summarized by its mean key. Selection is **exp-free** (raw dot products) and
**deterministic** (ties broken by ascending block id). The query's own
("local") block is always selected when `force_local_block`.

**Selected-block layout contract:** `msa_select_blocks(scores, ...)` consumes
`scores` shaped `(B, Hkv, Sq, num_blocks)` and produces `block_ids` shaped
`(B, Hkv, Sq, top_k)` with `i64` element type. The first three dimensions must
match the score tensor exactly, and the final dimension must equal the
`top_k` attribute. This is the schedule-facing layout consumed by sparse main
branch planning: each `(B, Hkv, Sq)` row owns a sorted Top-`k` block-id list,
and query heads map into that selected row through the GQA group id
`h // (Hq / Hkv)`.

---

## 2. How MSA differs from `deepseek_sparse_attention` (NSA)

| | NSA (`deepseek_sparse_attention`) | MSA |
|---|---|---|
| Structure | **Blend** of 3 branches via a learned gate | Index Branch + **single exact** sparse Main Branch |
| Branches | sliding window + compressed blocks + top-k blocks | top-k block selection only |
| Selection | per head | **per GQA group** (shared across the group's query heads) |
| Scoring | per-branch | one lightweight exp-free index score per (group, query, block) |
| Local context | sliding-window branch | forced-local block inside the selected set |
| Output | gated sum of 3 branch outputs | exact attention over the union of selected blocks |

MSA is deliberately streamlined: there is no gate and no compressed/sliding
blend — just *select blocks, then attend exactly over them*.

---

## 3. Public API

```python
import numpy as np, tessera as ts
from tessera.nn.functional import minimax_sparse_attention

B, Hq, Hkv, Sq, Sk, D = 1, 8, 2, 64, 64, 16
Q = np.random.randn(B, Hq, Sq, D)
K = np.random.randn(B, Hkv, Sk, D)
V = np.random.randn(B, Hkv, Sk, D)

# High-level wrapper (nn.functional):
O = minimax_sparse_attention(Q, K, V, block_size=16, top_k=2)   # (B, Hq, Sq, D)

# Low-level ops (the three MSA primitives):
scores = ts.ops.msa_index_scores(Q, K, block_size=16)            # (B, Hkv, Sq, num_blocks)
ids    = ts.ops.msa_select_blocks(scores, top_k=2, block_size=16)# (B, Hkv, Sq, top_k) int64
O      = ts.ops.msa_sparse_attention(Q, K, V, block_size=16, top_k=2)

# Debug metadata (selected ids, coverage fraction, local-block hit rate):
O, dbg = ts.ops.msa_sparse_attention(Q, K, V, block_size=16, top_k=2, return_debug=True)
```

Canonical names (one per concept):

| Concept | Canonical name |
|---------|----------------|
| Index Branch block scoring | `tessera.ops.msa_index_scores` |
| Top-k block selection | `tessera.ops.msa_select_blocks` |
| Exact block-sparse attention | `tessera.ops.msa_sparse_attention` |
| High-level GQA wrapper | `tessera.nn.functional.minimax_sparse_attention` |

Autodiff: `msa_index_scores` (smooth) and `msa_sparse_attention` (exact Main
Branch) have VJP+JVP registered; `msa_select_blocks` is a hard top-k and is
correctly `not_applicable` for autodiff.

---

## 4. What's proven (Phase 1)

| Property | Test |
|---|---|
| Output / score / selection shapes | `tests/unit/test_msa.py` |
| GQA grouping (`Hq % Hkv`, group-shared selection) | `tests/unit/test_msa.py` |
| Causal masking (no future-token leakage) | `tests/unit/test_msa.py` |
| Forced-local-block always selected | `tests/unit/test_msa.py` |
| Deterministic top-k (repeatable, sorted) | `tests/unit/test_msa.py` |
| **Dense-equivalence: `top_k == num_blocks` == dense GQA attention** | `tests/unit/test_msa.py` |
| VJP vs finite differences (top_k==num_blocks, smooth regime) | `tests/unit/test_msa.py` |
| op_catalog + primitive_coverage registration | `tests/unit/test_msa.py` |

The dense-equivalence anchor is the key correctness guarantee: when every block
is selected, MSA collapses to ordinary (causal) GQA attention bit-for-bit,
independent of the (approximate, mean-pooled) index scores.

---

## 5. Phase 2 — Graph IR op contract (landed)

Three ODS ops in the `tessera` dialect (`src/compiler/ir/TesseraOps.td`), mirroring
the reasoning-family pattern (cf. `deepseek_sparse_attention`):

| Op | Operands | Key attributes | Result |
|---|---|---|---|
| `tessera.msa_index_scores` | `q`, `k` | `block_size`, opt `scale` | `scores` |
| `tessera.msa_select_blocks` | `scores` | `top_k`, `block_size`, `force_local_block`, `causal` | `block_ids` |
| `tessera.msa_sparse_attention` | `q`, `k`, `v` | `block_size`, `top_k`, `force_local_block`, `causal`, opt `scale` | `o` |

`force_local_block` is an ODS attribute — a **semantic** the verifier and any
downstream pass can rely on, not a lowering heuristic. Verifiers
(`TesseraOps.cpp`) enforce GQA divisibility (`Hq % Hkv == 0`), KV-block
divisibility (`Sk % block_size == 0`), and `top_k <= num_blocks`, on top of the
shared `verifyAttentionQKV` shape/dtype checks. A compiler-visibility pass
`-tessera-msa-expand` (`AttentionFamilyPasses.cpp`) tags the ops
`tessera.reasoning.family = "minimax_sparse"` + per-op variant and is wired into
the Apple reasoning-attention prologue. Lit:
`tests/tessera-ir/phase8/msa_visibility.mlir` (visibility) +
`tests/tessera-ir/phase3/msa_verifier.mlir` (1 positive + 3 negative verifier
cases). Schedule/Tile IR (selected-block layout, reverse sparse worklist,
partial-output/logsumexp buffers, combine) is carried into Phase 3 lowering.

## 6. Phase 3 — Apple GPU host-select lane (landed)

`tessera.msa_sparse_attention` joins the Apple GPU `sparse_attn` lane (alongside
`deepseek_sparse_attention` / `attn_top_k_blocks`) in
`compiler/apple_gpu_envelope.py`. Under `@jit(target="apple_gpu")` the runtime
dispatcher (`runtime.py::_apple_gpu_dispatch_sparse_attn`) runs:

1. **Host** (data-dependent, reusing the reference ops so the math is identical):
   Index Branch scoring (`msa_index_scores`) + Top-k block selection
   (`msa_select_blocks`); per-`(b, group, query)` deduped, causally-valid token
   gather; a per-row additive mask folds token-level causal masking + block
   dedup so the GPU result matches the reference bit-for-bit.
2. **GPU** (the attention FLOPs): `A = Qb @ Kbᵀ` → `+mask` → `softmax(A)` →
   `O = P @ Vb`, via two GPU bmms + the GPU softmax lane.

The single-source envelope auto-feeds the driver gate, the C++
`isAppleGpuRuntimeOp` mirror (`apple_runtime_ops.inc`, regenerated by
`scripts/generate_apple_runtime_ops_table.py`), and the dashboards. Guard:
`tests/unit/test_msa_apple_gpu.py` (skips off-Darwin) — direct-dispatch and
end-to-end `@jit` paths validated vs the reference at fp32 tolerance, incl. GQA
and dense-equivalence (`top_k == num_blocks`).

## 7. Phase 3 CUDA/NVIDIA lowering target (artifact path landed)

The CUDA/NVIDIA artifact path lowers the selected-block layout to a KV-outer
sparse attention target, not to a monolithic opaque model op. Graph IR
`tessera.msa_sparse_attention` becomes `schedule.attn.kv_outer_sparse`, then
Tile IR `tessera_attn.msa_kv_outer_sparse`, then an `artifact_only` NVIDIA
Target IR kernel contract named `msa_kv_outer_sparse`. The contract carries
`block_ids` shaped `(B,Hkv,Sq,top_k)`, `gqa_group_size = Hq / Hkv`, explicit
`mode = "prefill" | "decode"` metadata, and `kv_traversal = "kv_outer"`.
Dense-equivalence remains the first oracle: when `top_k == num_blocks`, the
sparse target must match dense GQA.

See [MSA CUDA detail](msa-cuda-phase3.md). The executable contract guard is
`tests/unit/test_msa_kv_outer_schedule.py`; there is no C++ MLIR lowering for
this Python-authored artifact contract yet.

**Execution-mode reality.** The program reports `execution_mode = "metal_runtime"`
(`compiler_path = apple_gpu_mps`) — identical to the rest of the sparse-attn lane:
the exact attention runs on Metal; block selection is host-side. The three
conceptual modes — *reference* (CPU numpy), *host_select_gpu_attention* (today's
Apple GPU lane), *native_sparse* (a single fused exp-free-Top-k + KV-outer
kernel) — are a semantic distinction; only the first two are realized. A new
top-level `execution_mode` value for `native_sparse` is deferred to when that
kernel exists (it would extend the single-source execution matrix).

## 7. Phase 4 — Model integration (landed)

`tessera.nn.MinimaxSparseAttention` — a stateful GQA block-sparse attention
layer: grouped-query Q/K/V projections (`W_k`/`W_v` narrower than `W_q`) →
`ops.msa_sparse_attention` → output projection. `MinimaxSparseAttention.from_gqa(
embed_dim, num_heads, num_kv_heads, seq_len, block_size, sparsity, dense=…)`
converts a dense GQA/full-attention config into an MSA layer: `top_k =
ceil(sparsity * num_blocks)`, and `dense=True` makes it exactly dense GQA (the
warmup/training setting before annealing sparsity). The reference Index Branch is
parameter-free (mean-pooled), so there are no separate index-projection
dimensions to configure. Guard: `tests/unit/test_msa.py` (module + `from_gqa`).

## 8. Phase 5 — Example + benchmark (landed)

`examples/attention/minimax_sparse_attention.py` (CPU reference) compares dense
GQA vs sparse MSA on the same Q/K/V — printing the dense-equivalence anchor
(`top_k == num_blocks` reproduces dense GQA to ~1e-15 fp64), selected-block
coverage / local-block-hit stats, and a long-context **theoretical**
attention-compute reduction sweep (`attention_flops` / `long_context_table`,
accounting for the Index Branch overhead). It is explicit that the reference
path realizes **no wall-clock speedup** — the native fused sparse kernel is
future. Guard: `tests/unit/test_example_minimax_sparse_attention.py`.

## 9. Phase 6 — native fused block-sparse MSL kernel (landed)

`tessera_apple_gpu_msa_block_sparse_f32` (`apple_gpu_runtime.mm`, non-Darwin
parity in `apple_gpu_runtime_stub.cpp`) is the native Main Branch: **one MSL
dispatch**, one thread per `(batch, query_head, query_row)` streaming **KV-outer
over the host-selected blocks only**, with flash-style online softmax and
intra-block causal masking — never materializing a gathered KV set or an
`(Sq, Sk)` score matrix. GQA-aware (head `h` reads KV group `h / (Hq/Hkv)` and
its block ids). The runtime routes `tessera.msa_sparse_attention` through it for
`Dv == Dqk == D ≤ 256` (the composed bmm path is the fallback otherwise);
selection (Index Branch Top-k) stays host-side — a small data-dependent step,
like `attn_top_k_blocks`.

**Measured (M1 Max, S=1024, B·Hq=8, D=64, block_size=64, num_blocks=16):** native
latency scales ~linearly with `top_k` — **10.6× at top_k=1** (6% coverage), 5.2×
at top_k=2, 1.0× at top_k=16 — versus the dense-equivalent (same kernel, all
blocks). So the proven *theoretical* compute reduction is now a **measured
wall-clock** reduction on this machine. Caveats: this compares native-sparse vs
native-dense in the same kernel (not vs a hand-tuned dense baseline), and the
kernel is naive/untiled (the *scaling* is the proven property, not the absolute
TFLOPs). Guard: `tests/unit/test_msa_apple_gpu.py` (native symbol + direct-ABI vs
reference, GQA, dense-equivalence).

### Phase 6 follow-on — fp16 native variant (landed)

`tessera_apple_gpu_msa_block_sparse_f16` — half I/O (half the buffer footprint),
fp32 accumulators (the standard flash-attn mixed-precision pattern). The runtime
routes f16 `tessera.msa_sparse_attention` through it; **bf16 routes through the
f32 kernel** at the boundary (no native MSL bf16 type), matching the rest of the
Apple lane. Validated vs the f32 reference within f16 tolerance (~1e-3).

**Honest perf note:** f16 is *not* faster than f32 on this naive kernel — at
S=1024/top_k=8 it measures slightly slower (≈23 ms vs ≈20 ms). The kernel is
**compute-bound** (one thread per query, serial dot products), so halving the I/O
width buys nothing and the per-element half→float conversion costs a little. The
f16 win here is **memory** (half the resident KV → longer context fits in unified
memory) + dtype parity for real models; the *bandwidth* win awaits a
tiled/vectorized kernel. Guard: `tests/unit/test_msa_apple_gpu.py` (f16 symbol +
f16-dispatch vs f32-reference).

### Phase 6 follow-on — tiled (simdgroup-cooperative) kernel (landed)

`tessera_apple_gpu_msa_block_sparse_tiled_f32` — the fix for the compute-bound
bottleneck. Instead of one thread per query doing a serial D-length dot per key,
a **32-lane SIMD-group** is assigned per query: each key's Q·K dot is computed
cooperatively (each lane strides over D, then `simd_sum` reduces), and the fp32
online-softmax accumulator `o[]` is partitioned across lanes (lane owns
`d ∈ {lane, lane+32, …}`). `m`/`l` are scalar and identical on every lane (the
reduced score is uniform), so no barriers are needed. Correctness is identical to
the scalar kernel (validated ~1e-7, incl. `D=40` non-multiple-of-32 striding); it
is the default `@jit(target="apple_gpu")` f32 path (the C side falls back to the
scalar kernel when simd reduction is absent).

**Measured (M1 Max, S=1024, D=64):** **3.9–5.2× faster than the scalar kernel** —
top_k=4: 9.7 → 2.5 ms; top_k=8: 18.5 → 4.4 ms; top_k=16 (dense-equiv): 34.4 →
6.6 ms. Composed with sparsity, MSA at top_k=4 (2.5 ms) is **~14×** the scalar
dense path (34.4 ms). Guard: `tests/unit/test_msa_apple_gpu.py` (tiled symbol,
direct-ABI vs reference, strided-owner head dims).

### Phase 6 follow-on — tiled fp16 (landed)

`tessera_apple_gpu_msa_block_sparse_tiled_f16` combines both wins: the 32-lane
cooperative dot **and** half I/O. The runtime prefers it for f16 inputs (falling
back to the f16 scalar kernel, then the fp32-conversion reference). Validated
within f16 tolerance (~1e-3, incl. `D=40`). **Measured (M1 Max, S=1024, D=64):**
now **marginally faster than tiled-f32** (1.03–1.07×) — the bandwidth halving
*helps* once the dot is cooperative, exactly the opposite of f16 on the *scalar*
kernel (where it lost). The win is modest because per-token reduction/softmax
overhead remains; a fully vectorized (half4-load) kernel would show more. Plus
half the resident KV → longer context fits.

### Phase 6 follow-on — GPU-resident selection, huge-batch regime (landed)

The earlier note called GPU-side Top-k an anti-optimization — **and it is, at the
shapes this implementation usually targets** (host `argpartition` ~1 ms ≪ the GPU
attention). But at very large `B·Hkv·Sq·num_blocks` the *Index-Branch scoring
matmul* `q_grp·K_cᵀ` (`O(B·Hkv·Sq·D·num_blocks)`) dominates on host. So MSA now
runs the **whole Index Branch on the GPU** above a FLOP threshold
(`TESSERA_MSA_GPU_SELECT_FLOPS`, default 64M): cheap group-mean-query /
block-mean-key reductions on host, the scoring matmul through the **GPU bmm**, and
a GPU **`tessera_apple_gpu_msa_select_blocks_f32`** top-k kernel → `block_ids`
(local + top-k′ causally-valid past blocks; `-1` fillers, masked downstream).
Below the threshold the host path stays (no regression). End-to-end validated vs
the reference (~1e-7); **measured 2.42× faster Index Branch** at
B2·Hq16·Hkv2·Sq4096·D64·nb64 (14.6 → 6.0 ms). The fully-fused per-query kernel was
*not* built — it would redundantly recompute the block mean-keys in every thread;
computing `K_c` once and routing the matmul through the bmm is the right shape.
Guard: `tests/unit/test_msa_apple_gpu.py` (select symbol, forced-threshold
end-to-end, small-shape gating).

## 11. Open frontier

> The cross-backend work (ROCm already executes MSA; x86 MSA is the one CPU gap;
> the CUDA kernel + `attn_bias`/DFlash seams) is sequenced in the consolidated
> [`attention-family.md`](attention-family.md).

The remaining item is the **native** CUDA KV-outer kernel plus a real
H800/Blackwell speedup proof — **hardware-gated on NVIDIA**. The compiler
artifact path now carries the selected-block KV-outer schedule to NVIDIA Target
IR; this Mac still cannot prove native NVIDIA execution. The Apple-GPU surface
is now: native fused kernel (sparsity → 10.6×) → tiled f32/f16 (3.9–5.2× over
scalar) → GPU-resident Index Branch (2.4× at scale), all measured on this M1
Max.
