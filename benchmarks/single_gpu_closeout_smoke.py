"""Tiny benchmark smoke rows for single-GPU closeout audit evidence.

These rows are deliberately small and reference-runnable.  Their job is to keep
the support-table benchmark axis grounded in an executable harness for the
single-GPU compiler/domain rows that are otherwise easy to lose in larger model
benchmarks.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

import numpy as np

import tessera as ts
from tessera import complex as tc
from tessera.cache import KVCacheHandle
from tessera.nn import varlen as tv
from tessera.stdlib import quant


CLOSEOUT_SMOKE_OPS: tuple[str, ...] = (
    "attn_compressed_blocks",
    "attn_local_window_2d",
    "attn_top_k_blocks",
    "linear_attn_state",
    "lookahead_sparse_attention",
    "msa_sparse_attention",
    "memory_index_score",
    "msa_index_scores",
    "varlen_sdpa",
    "score_combine",
    "dynamic_slice",
    "masked_categorical",
    "slice",
    "cast",
    "chunk",
    "rope_split",
    "split",
    "unpack",
    "dequant_matmul",
    "kv_cache_read",
    "complex_abs",
    "complex_arg",
    "complex_conjugate",
    "complex_div",
    "complex_exp",
    "complex_log",
    "complex_mul",
    "complex_pow",
    "complex_sqrt",
    "mobius",
    "stereographic",
)


def _attention_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.linspace(-0.5, 0.5, 1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
    k = np.linspace(0.25, 0.75, 1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
    v = np.linspace(-0.75, 0.25, 1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
    return q, k, v


def _complex_inputs() -> tuple[tc.ComplexScalar, tc.ComplexScalar]:
    z = tc.from_numpy(np.array([1.0 + 0.5j, -0.25 + 0.75j], dtype=np.complex64))
    w = tc.from_numpy(np.array([0.75 - 0.25j, 0.5 + 0.125j], dtype=np.complex64))
    return z, w


def _consume(value: Any) -> float:
    if hasattr(value, "to_numpy"):
        return _consume(value.to_numpy())
    if isinstance(value, dict):
        return sum(_consume(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_consume(v) for v in value)
    arr = np.asarray(value)
    if arr.dtype.kind == "c":
        arr = np.abs(arr)
    if arr.dtype == object:
        return float(len(arr.reshape(-1)))
    finite = np.nan_to_num(arr.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return float(finite.sum())


def _cases() -> dict[str, Callable[[], Any]]:
    q, k, v = _attention_inputs()
    x = np.arange(16, dtype=np.float32).reshape(4, 4)
    z, w = _complex_inputs()

    def attn_compressed_blocks() -> Any:
        k_c = k.reshape(1, 2, 2, 2, 4).mean(axis=3)
        v_c = v.reshape(1, 2, 2, 2, 4).mean(axis=3)
        return ts.ops.attn_compressed_blocks(q, k_c, v_c)

    def attn_local_window_2d() -> Any:
        q2 = q.reshape(1, 2, 2, 2, 4)
        return ts.ops.attn_local_window_2d(q2, q2, q2, window=(1, 1))

    def attn_top_k_blocks() -> Any:
        scores = np.matmul(q, np.swapaxes(k.reshape(1, 2, 2, 2, 4).mean(axis=3), -1, -2))
        return ts.ops.attn_top_k_blocks(q, k, v, scores=scores, top_k=1, block_size=2)

    def linear_attn_state() -> Any:
        return ts.ops.linear_attn_state(q, k, v, causal=True)

    def lookahead_sparse_attention() -> Any:
        return ts.ops.lookahead_sparse_attention(
            q, k, v, window_size=2, block_size=2, threshold=0.0, causal=True
        )

    def msa_sparse_attention() -> Any:
        return ts.ops.msa_sparse_attention(q, k, v, block_size=2, top_k=1, causal=True)

    def memory_index_score() -> Any:
        keys = k.reshape(1, 2, 2, 2, 4).mean(axis=3)
        return ts.ops.memory_index_score(keys, q)

    def msa_index_scores() -> Any:
        return ts.ops.msa_index_scores(q, k, block_size=2)

    def varlen_sdpa() -> Any:
        qv = q.reshape(2, 4, 4)
        kv = k.reshape(2, 4, 4)
        vv = v.reshape(2, 4, 4)
        cu = tv.cu_seqlens_from_lengths([2, 2])
        return ts.ops.varlen_sdpa(qv, kv, vv, cu_seqlens_q=cu, cu_seqlens_k=cu, causal=True)

    def score_combine() -> Any:
        return ts.ops.score_combine(q, v, gamma=0.25)

    def dequant_matmul() -> Any:
        w_q = quant.quantize_weight(
            np.linspace(-0.25, 0.25, 16, dtype=np.float32).reshape(4, 4),
            dtype="int4",
            group_size=4,
        )
        return ts.ops.dequant_matmul(np.eye(4, dtype=np.float32), w_q)

    def kv_cache_read() -> Any:
        cache = KVCacheHandle(num_heads=2, head_dim=4, max_seq=8)
        cache = ts.ops.kv_cache_append(cache, k.reshape(4, 2, 4), v.reshape(4, 2, 4))
        return ts.ops.kv_cache_read(cache, 1, 3)

    return {
        "attn_compressed_blocks": attn_compressed_blocks,
        "attn_local_window_2d": attn_local_window_2d,
        "attn_top_k_blocks": attn_top_k_blocks,
        "linear_attn_state": linear_attn_state,
        "lookahead_sparse_attention": lookahead_sparse_attention,
        "msa_sparse_attention": msa_sparse_attention,
        "memory_index_score": memory_index_score,
        "msa_index_scores": msa_index_scores,
        "varlen_sdpa": varlen_sdpa,
        "score_combine": score_combine,
        "dynamic_slice": lambda: ts.ops.dynamic_slice(x, (1, 1), (2, 2)),
        "masked_categorical": lambda: ts.ops.masked_categorical(
            np.array([[0.1, 0.7, -0.2]], dtype=np.float32),
            np.array([[True, True, False]]),
        ),
        "slice": lambda: ts.ops.slice(x, (0, 1), (3, 2)),
        "cast": lambda: ts.ops.cast(x, "fp16"),
        "chunk": lambda: ts.ops.chunk(x, 2, axis=0),
        "rope_split": lambda: ts.ops.rope_split(np.zeros((2, 8), dtype=np.float32), rope_dim=4),
        "split": lambda: ts.ops.split(x, 2, axis=1),
        "unpack": lambda: ts.ops.unpack(ts.ops.pack(x, "row_major")),
        "dequant_matmul": dequant_matmul,
        "kv_cache_read": kv_cache_read,
        "complex_abs": lambda: tc.complex_abs(z),
        "complex_arg": lambda: tc.complex_arg(z),
        "complex_conjugate": lambda: tc.complex_conjugate(z),
        "complex_div": lambda: tc.complex_div(z, w),
        "complex_exp": lambda: tc.complex_exp(z),
        "complex_log": lambda: tc.complex_log(z),
        "complex_mul": lambda: tc.complex_mul(z, w),
        "complex_pow": lambda: tc.complex_pow(z, w),
        "complex_sqrt": lambda: tc.complex_sqrt(z),
        "mobius": lambda: tc.mobius(z, 1.0 + 0.0j, 0.25 + 0.0j, 0.1 + 0.0j, 1.0 + 0.0j),
        "stereographic": lambda: tc.stereographic(
            np.array([[0.0, 0.0, -1.0], [0.5, 0.0, 0.0]], dtype=np.float32)
        ),
    }


def run_smoke(*, reps: int = 3) -> list[dict[str, Any]]:
    """Run the closeout smoke benchmark and return one row per op."""
    cases = _cases()
    rows: list[dict[str, Any]] = []
    for op_name in CLOSEOUT_SMOKE_OPS:
        fn = cases[op_name]
        _consume(fn())
        start = time.perf_counter()
        checksum = 0.0
        for _ in range(max(1, int(reps))):
            checksum += _consume(fn())
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(1, int(reps))
        rows.append({
            "op": op_name,
            "latency_ms": elapsed_ms,
            "checksum": checksum,
            "ok": True,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = run_smoke(reps=args.reps)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            print(f"{row['op']:28s} {row['latency_ms']:9.4f} ms")


if __name__ == "__main__":
    main()
