#!/usr/bin/env python3
"""MiniMax Sparse Attention (MSA) — dense-vs-sparse comparison + FLOP benchmark.

MSA (arXiv:2606.13392) is a GQA blockwise sparse attention: a lightweight Index
Branch scores KV blocks and selects Top-k of them per GQA group; the Main Branch
runs *exact* attention over only the selected blocks. This example:

  1. Compares dense GQA vs sparse MSA on the same Q/K/V, and shows the
     dense-equivalence anchor — MSA with ``top_k == num_blocks`` reproduces dense
     GQA attention bit-for-bit (within fp tolerance).
  2. Prints selected-block statistics (coverage, local-block-hit).
  3. Reports the **theoretical** per-token attention-compute reduction
     (dense vs MSA), separated from any runtime-speedup claim.
  4. Runs a long-context synthetic sweep of that theoretical reduction.

Runs on the CPU reference path — no accelerator required. The reference numpy
path does selection + gather on the host and is *not* a speedup; the native
fused sparse kernel (the paper's exp-free Top-k + KV-outer attention) is future
work, so this example reports theoretical compute, never wall-clock speedup.
See docs/msa.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import tessera as ts


# ── reference dense GQA (the equivalence target) ─────────────────────────────

def dense_gqa(Q, K, V, *, causal=True, scale=None):
    """Plain dense grouped-query attention reference."""
    B, Hq, Sq, D = Q.shape
    Hkv, Sk, Dv = K.shape[1], K.shape[2], V.shape[-1]
    g = Hq // Hkv
    scale = (1.0 / math.sqrt(D)) if scale is None else scale
    out = np.zeros((B, Hq, Sq, Dv), dtype=np.float64)
    for b in range(B):
        for h in range(Hq):
            grp = h // g
            s = (Q[b, h] @ K[b, grp].T) * scale
            if causal:
                fut = np.arange(Sk)[None, :] > np.arange(Sq)[:, None]
                s = np.where(fut, -np.inf, s)
            e = np.exp(s - s.max(axis=-1, keepdims=True))
            w = e / e.sum(axis=-1, keepdims=True)
            out[b, h] = w @ V[b, grp]
    return out


# ── dense-vs-sparse comparison ───────────────────────────────────────────────

@dataclass
class Comparison:
    seq_len: int
    block_size: int
    num_blocks: int
    top_k: int
    coverage: float            # mean fraction of causally-valid KV blocks attended
    local_block_hit: float     # fraction of rows whose own block was selected
    dense_equiv_max_abs_err: float  # MSA(top_k=num_blocks) vs dense GQA
    sparse_vs_dense_max_abs_err: float  # how far sparse MSA drifts from dense (informational)


def compare(*, B=1, Hq=4, Hkv=2, seq_len=64, head_dim=16, block_size=8,
            top_k=2, causal=True, seed=0) -> Comparison:
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(B, Hq, seq_len, head_dim)).astype(np.float64)
    K = rng.normal(size=(B, Hkv, seq_len, head_dim)).astype(np.float64)
    V = rng.normal(size=(B, Hkv, seq_len, head_dim)).astype(np.float64)
    num_blocks = seq_len // block_size

    # Sparse MSA (with debug stats) + dense MSA (top_k == num_blocks).
    sparse, dbg = ts.ops.msa_sparse_attention(
        Q, K, V, block_size=block_size, top_k=top_k, causal=causal, return_debug=True
    )
    dense_msa = ts.ops.msa_sparse_attention(
        Q, K, V, block_size=block_size, top_k=num_blocks, causal=causal
    )
    ref = dense_gqa(Q, K, V, causal=causal)

    return Comparison(
        seq_len=seq_len,
        block_size=block_size,
        num_blocks=num_blocks,
        top_k=top_k,
        coverage=float(dbg["coverage"]),
        local_block_hit=float(dbg["local_block_hit"]),
        dense_equiv_max_abs_err=float(np.max(np.abs(dense_msa - ref))),
        sparse_vs_dense_max_abs_err=float(np.max(np.abs(np.asarray(sparse) - ref))),
    )


# ── theoretical attention-compute accounting ─────────────────────────────────

def attention_flops(*, seq_len, head_dim, block_size, top_k, Hq, Hkv):
    """Per-token (multiply-add) attention FLOPs for dense GQA vs MSA.

    Dense main branch attends all ``seq_len`` keys: ~2 * seq_len * head_dim per
    query head. MSA attends ``top_k * block_size`` selected keys + a lightweight
    Index Branch that scores ``num_blocks`` block summaries per GQA group
    (~num_blocks * head_dim per query, shared across the group's heads).
    Returns dense/msa per-token FLOPs and the reduction factor.
    """
    num_blocks = seq_len // block_size
    g = Hq // Hkv
    # Dense: per query head, scores + output over all keys.
    dense = Hq * 2 * seq_len * head_dim
    # MSA main branch: per query head, over the selected tokens only.
    sel_tokens = top_k * block_size
    msa_main = Hq * 2 * sel_tokens * head_dim
    # Index branch: per GQA group (shared by g heads), score num_blocks summaries.
    msa_index = Hkv * num_blocks * head_dim
    msa = msa_main + msa_index
    return {
        "num_blocks": num_blocks,
        "selected_tokens": sel_tokens,
        "dense_flops_per_token": dense,
        "msa_flops_per_token": msa,
        "reduction_factor": dense / msa,
    }


def long_context_table(*, head_dim=128, block_size=128, sparsity=0.1,
                       Hq=64, Hkv=8, seq_lens=(8192, 32768, 131072, 1048576)):
    """Theoretical attention-compute reduction across context lengths.

    top_k = max(1, ceil(sparsity * num_blocks)). Returns a list of dict rows.
    This is *compute*, not wall-clock — the reference path realizes no speedup;
    the native sparse kernel is future work.
    """
    rows = []
    for S in seq_lens:
        num_blocks = S // block_size
        top_k = max(1, math.ceil(sparsity * num_blocks))
        f = attention_flops(seq_len=S, head_dim=head_dim, block_size=block_size,
                            top_k=top_k, Hq=Hq, Hkv=Hkv)
        rows.append({"seq_len": S, "top_k": top_k, **f})
    return rows


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("MiniMax Sparse Attention (MSA) — dense vs sparse")
    print("=" * 60)

    c = compare(seq_len=64, block_size=8, top_k=2)
    print(f"shape: seq_len={c.seq_len}, block_size={c.block_size}, "
          f"num_blocks={c.num_blocks}, top_k={c.top_k}")
    print(f"  selected-block coverage : {c.coverage:.3f}  "
          f"(~top_k/num_blocks = {c.top_k}/{c.num_blocks} = {c.top_k / c.num_blocks:.3f})")
    print(f"  local-block hit rate    : {c.local_block_hit:.3f}")
    print(f"  dense-equivalence error : {c.dense_equiv_max_abs_err:.2e}  "
          f"(MSA top_k==num_blocks vs dense GQA — should be ~0)")
    print(f"  sparse drift vs dense   : {c.sparse_vs_dense_max_abs_err:.2e}  "
          f"(informational: sparse approximates dense)")

    print()
    print("Theoretical attention-compute reduction (NOT wall-clock speedup):")
    print(f"  {'seq_len':>10} {'top_k':>7} {'num_blocks':>11} {'reduction':>10}")
    for r in long_context_table():
        print(f"  {r['seq_len']:>10} {r['top_k']:>7} {r['num_blocks']:>11} "
              f"{r['reduction_factor']:>9.1f}x")
    print()
    print("Reference path runs selection + gather on the host (no speedup). The "
          "native fused sparse kernel\n(exp-free Top-k + KV-outer attention) is "
          "future work — see docs/msa.md.")


if __name__ == "__main__":
    main()
