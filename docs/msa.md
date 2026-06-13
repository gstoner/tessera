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
[`deepseek_sparse_attention`](CANONICAL_API.md) (NSA) path.

> **Status (2026-06-13).** Phase 0 (contract) + Phase 1 (reference numpy API) +
> Phase 2 (Graph IR op contract) landed. The reference path runs on CPU and is
> dense-equivalent when `top_k == num_blocks`. The three MSA Graph IR ops are
> ODS-registered with verifiers and a compiler-visibility pass (`tessera-opt`
> builds clean; lit-proven). Native target speedups are **not** claimed yet: the
> Apple-GPU host-select path (Phase 3) reuses the existing `flash_attn` lane for
> exact attention over host-selected blocks, and the CUDA KV-outer sparse kernel
> is lit-fixture-only on non-NVIDIA hardware. The Schedule/Tile IR sparse
> worklist (selected-block layout, reverse index, partial-output/logsumexp
> buffers, combine phase) is carried into Phase 3 with the lowering. Phases 3–5
> (target lowering, model integration, example/benchmark) are tracked.

---

## 1. Shape contract

| Tensor | Shape | Notes |
|---|---|---|
| `Q` | `(B, Hq, Sq, D)` | query heads |
| `K`, `V` | `(B, Hkv, Sk, D)` | KV heads; `Dv` may differ for `V` |
| GQA constraint | `Hq % Hkv == 0` | group size `g = Hq // Hkv`; query head `h` ∈ group `h // g` |
| block constraint | `Sk % block_size == 0` | `num_blocks = Sk // block_size` |

**Block metadata:** `block_size`, `top_k` (`1 ≤ top_k ≤ num_blocks`),
`force_local_block` (default `True`), `causal` (default `True`).

**Selection granularity:** per query token, **per GQA group** (shared by every
query head in the group — *not* per head), block-level KV ids. Query heads in a
group are mean-pooled into one group query for scoring; each KV block is
summarized by its mean key. Selection is **exp-free** (raw dot products) and
**deterministic** (ties broken by ascending block id). The query's own
("local") block is always selected when `force_local_block`.

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

## 6. Roadmap (Phases 3–5, not yet built)

- **Phase 3 — Target runtime:** Apple GPU host-select + existing `flash_attn`
  lane first (real execution here); CUDA exp-free top-k + KV-outer sparse kernel
  lit-only on this Mac. Runtime metadata records `execution_mode ∈
  {reference, host_select_gpu_attention, native_sparse}`.
- **Phase 4 — Model integration:** `NativeSparseAttention(mode="msa")` or
  equivalent; long-context config wiring; GQA/full → MSA conversion helpers.
- **Phase 5 — Example + benchmark:** a planned MSA example under examples/attention/
  (minimax_sparse_attention.py — dense vs MSA + selected-block stats / FLOP
  reduction); long-context synthetic benchmark separating *theoretical compute*
  from *runtime speedup*.
