"""Benchmark-coverage source for the support-table ``bench`` axis.

Before this module the bench axis was a hard-coded GA/EBM frozenset in
``audit.py`` — it modelled *every other* runnable benchmark surface (GEMM,
attention, fusion chains, collectives, the Apple hot paths, MegaMoE) as
un-benchmarked, even though those benchmarks exist and several ops carry a
first-class ``benchmark_json`` in ``backend_manifest``.  That made the
``support_table`` bench column silently stale.

This module derives bench coverage from five honest sources, in priority
order:

1. **Manifest-attached** — any ``backend_manifest`` entry that carries a
   ``benchmark_json``.  This is read live, so a new benchmarked kernel can
   never go stale here again (the gap that motivated this module).
2. **GA/EBM inventory** — the GA + native-EBM primitives benchmarked by
   ``benchmarks/apple_gpu/benchmark_ga_ebm.py`` (no manifest ``benchmark_json``;
   they benchmark through that harness's row catalog).
3. **Explicit real-op map** — surfaces that benchmark a *callable op* without a
   manifest ``benchmark_json`` (GEMM alias, collectives, MHA, MegaMoE overlap).
4. **Operator-benchmark coverage** — representative ops from the active
   ``Tessera_Operator_Benchmarks`` groups.
5. **Single-GPU closeout smoke** — tiny runnable rows for compiler/domain
   primitives whose benchmark evidence should not depend on large model
   harnesses or backend-specific hardware being present in CI.

**Fused-chain names are deliberately excluded.**  ``matmul_softmax`` /
``matmul_gelu`` / ``matmul_rmsnorm`` / ``matmul_softmax_matmul`` / ``swiglu``
are *benchmark-only aliases* of a (matmul + epilogue) chain — they are not
callable ``tessera.ops.*`` names (absent from ``OP_SPECS``) and have no
standalone runtime symbol, so they must NOT become support-table rows.  Their
constituent ops (matmul / softmax / gelu / rmsnorm) carry the bench coverage.
This decision is enforced by ``tests/unit/test_benchmark_coverage.py``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from .op_catalog import OP_SPECS

_GA_EBM_BENCH = "benchmarks/apple_gpu/benchmark_ga_ebm.py"
_SINGLE_GPU_CLOSEOUT_SMOKE = "benchmarks/single_gpu_closeout_smoke.py"

# GA + native-EBM primitives benchmarked by benchmark_ga_ebm.py.  These have no
# manifest ``benchmark_json`` — the harness owns its own row catalog.
_GA_EBM_BENCH_OPS: dict[str, str] = {
    op: _GA_EBM_BENCH
    for op in (
        *[
            f"clifford_{n}"
            for n in (
                "geometric_product", "wedge", "left_contraction", "inner",
                "grade_projection", "reverse", "grade_involution", "conjugate",
                "norm", "exp", "log", "rotor_sandwich", "hodge_star",
                "ext_deriv", "vec_deriv", "codiff", "integral",
            )
        ],
        "ebm_inner_step", "ebm_refinement", "ebm_langevin_step",
        "ebm_decode_init", "ebm_bivector_langevin", "ebm_sphere_langevin",
        "ebm_self_verify", "ebm_energy", "ebm_energy_quadratic",
        "ebm_partition_exact",
        # Fused-chain GA row emitted by the GA/EBM benchmark harness.
        "clifford_norm_squared",
    )
}

# Surfaces that benchmark a callable op without a manifest ``benchmark_json``.
# Every key MUST be a real op in ``op_catalog.OP_SPECS`` (the drift gate
# enforces this) so the bench axis never invents a phantom support-table row.
_EXPLICIT_BENCH_OPS: dict[str, str] = {
    "gemm": "benchmarks/benchmark_gemm.py",
    "all_reduce": "benchmarks/benchmark_collective.py",
    "all_gather": "benchmarks/benchmark_collective.py",
    "reduce_scatter": "benchmarks/benchmark_collective.py",
    "all_to_all": "benchmarks/benchmark_collective.py",
    "multi_head_attention": "benchmarks/benchmark_attention.py",
}

_SINGLE_GPU_CLOSEOUT_SMOKE_OPS: dict[str, str] = {
    op: _SINGLE_GPU_CLOSEOUT_SMOKE
    for op in (
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
}

_STRUCTURAL_BENCH_ALIASES: dict[str, str] = {
    # Structural wrappers covered by the same benchmark group as their
    # canonical movement primitive. Only aliases whose canonical op already has
    # benchmark evidence are listed here; slice/cast/cat/where remain visible
    # benchmark gaps until their own smoke rows land.
    "index_select": "gather",
    "memory_index_select": "gather",
    "memory_index_select_ste": "gather",
    "msa_select_blocks": "gather",
    "permute": "transpose",
    "rearrange": "transpose",
    "take": "gather",
}

#: Fused-chain benchmark aliases — benchmarked but NOT callable ops, so they are
#: intentionally absent from the support table.  Kept here so the decision is
#: greppable and the drift gate can assert they never leak into ``OP_SPECS``.
FUSED_CHAIN_BENCH_ALIASES: frozenset[str] = frozenset({
    "matmul_softmax", "matmul_gelu", "matmul_rmsnorm",
    "matmul_softmax_matmul", "swiglu",
})


@lru_cache(maxsize=1)
def _manifest_attached() -> dict[str, str]:
    """Ops whose backend_manifest entry carries a ``benchmark_json``."""
    from . import backend_manifest as _bm

    out: dict[str, str] = {}
    for op_name, entries in _bm.all_manifests().items():
        for entry in entries:
            bj = getattr(entry, "benchmark_json", None)
            if bj:
                out[op_name] = bj
                break
    return out


@lru_cache(maxsize=1)
def _operator_benchmarked() -> dict[str, str]:
    """Representative callable ops covered by Tessera_Operator_Benchmarks."""
    from . import operator_benchmarks_coverage as _obc

    out: dict[str, str] = {}
    for row in _obc.COVERAGE_ROWS:
        if row.coverage_status not in {"direct", "grouped"}:
            continue
        source = (
            "benchmarks/Tessera_Operator_Benchmarks/"
            f"scripts/configs/quick_sweep.yaml#{row.opbench_group}"
        )
        for op_name in row.representative_ops:
            if op_name in OP_SPECS:
                out[op_name] = source
    return out


def benchmark_source_for(op_name: str) -> Optional[str]:
    """Return the benchmark file covering ``op_name`` or ``None``."""
    alias = _STRUCTURAL_BENCH_ALIASES.get(op_name)
    if alias is not None:
        return benchmark_source_for(alias)
    return (
        _manifest_attached().get(op_name)
        or _GA_EBM_BENCH_OPS.get(op_name)
        or _EXPLICIT_BENCH_OPS.get(op_name)
        or _operator_benchmarked().get(op_name)
        or _SINGLE_GPU_CLOSEOUT_SMOKE_OPS.get(op_name)
    )


def benchmarked_ops() -> frozenset[str]:
    """Every op with benchmark coverage today (the support-table bench set)."""
    return (
        frozenset(_manifest_attached())
        | frozenset(_GA_EBM_BENCH_OPS)
        | frozenset(_EXPLICIT_BENCH_OPS)
        | frozenset(_operator_benchmarked())
        | frozenset(_SINGLE_GPU_CLOSEOUT_SMOKE_OPS)
        | frozenset(_STRUCTURAL_BENCH_ALIASES)
    )


__all__ = [
    "benchmark_source_for",
    "benchmarked_ops",
    "FUSED_CHAIN_BENCH_ALIASES",
]
