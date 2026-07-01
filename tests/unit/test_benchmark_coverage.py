"""Drift gate for the support-table bench-coverage source.

`benchmark_coverage.py` replaced a hard-coded GA/EBM-only frozenset in
`audit.py`. These tests lock the three contracts that keep the bench axis
honest:

  1. manifest-attached coverage is LIVE — every op with a `benchmark_json` in
     `backend_manifest` shows up (the staleness the module was built to fix);
  2. the explicit map names only REAL ops (no phantom support-table rows);
  3. fused-chain aliases never leak into `OP_SPECS` (they are benchmark-only).
"""

from tessera.compiler import backend_manifest as bm
from tessera.compiler import benchmark_coverage as bc
from tessera.compiler.op_catalog import OP_SPECS


def test_manifest_attached_benchmarks_are_live():
    # Every op whose manifest entry carries a benchmark_json is reported as
    # benchmarked — this is the regression the module exists to prevent.
    attached = {
        op
        for op, entries in bm.all_manifests().items()
        for e in entries
        if getattr(e, "benchmark_json", None)
    }
    assert attached, "expected at least one manifest-attached benchmark"
    assert attached <= bc.benchmarked_ops()
    for op in attached:
        assert bc.benchmark_source_for(op) is not None
    # the MoE blind-spot fix must be reflected here
    assert {"grouped_gemm", "moe_swiglu_block", "matmul"} <= bc.benchmarked_ops()


def test_explicit_map_names_only_real_ops():
    # Adding a non-OP_SPECS name here would invent a phantom support-table row.
    for op in bc._EXPLICIT_BENCH_OPS:
        assert op in OP_SPECS, f"explicit bench op {op!r} is not a real OP_SPECS op"


def test_single_gpu_closeout_smoke_names_only_real_ops():
    for op in bc._SINGLE_GPU_CLOSEOUT_SMOKE_OPS:
        assert op in OP_SPECS, f"closeout smoke op {op!r} is not a real OP_SPECS op"
        assert bc.benchmark_source_for(op) == "benchmarks/single_gpu_closeout_smoke.py"


def test_operator_benchmark_representatives_feed_bench_axis():
    expected = {
        "add", "mul", "relu", "sigmoid", "tanh",
        "reduce", "sum",
        "softmax", "layer_norm",
        "transpose", "gather",
        "conv2d", "flash_attn",
    }
    assert expected <= bc.benchmarked_ops()
    opbench_sourced = expected - {"conv2d", "flash_attn", "softmax"}
    for op in opbench_sourced:
        assert "Tessera_Operator_Benchmarks" in (bc.benchmark_source_for(op) or "")


def test_fused_chain_aliases_are_benchmark_only():
    # matmul_softmax / matmul_gelu / ... are NOT callable ops; they must never
    # become support-table rows. Their constituents carry the coverage.
    for alias in bc.FUSED_CHAIN_BENCH_ALIASES:
        assert alias not in OP_SPECS, f"fused alias {alias!r} leaked into OP_SPECS"
        assert alias not in bc.benchmarked_ops()


def test_bench_axis_is_no_longer_ga_ebm_only():
    # The headline gap: the bench set must now include non-GA/EBM surfaces.
    ops = bc.benchmarked_ops()
    non_ga_ebm = {o for o in ops if not (o.startswith("clifford_") or o.startswith("ebm_"))}
    assert {"matmul", "flash_attn", "all_reduce", "gemm"} <= non_ga_ebm


def test_ga_ebm_alias_rows_are_covered_by_ga_ebm_harness():
    assert bc.benchmark_source_for("ebm_energy_quadratic") == (
        "benchmarks/apple_gpu/benchmark_ga_ebm.py"
    )
    assert bc.benchmark_source_for("clifford_norm_squared") == (
        "benchmarks/apple_gpu/benchmark_ga_ebm.py"
    )


def test_source_priority_manifest_over_explicit():
    # matmul is both manifest-attached and a GEMM-family op; the manifest
    # baseline wins (manifest-attached is the strongest proof).
    assert bc.benchmark_source_for("matmul") == "benchmarks/baselines/apple_gpu_hot_paths.json"
    assert bc.benchmark_source_for("gemm") == "benchmarks/benchmark_gemm.py"


def test_structural_benchmark_aliases_reuse_canonical_sources():
    for alias, canonical in {
        "index_select": "gather",
        "memory_index_select": "gather",
        "memory_index_select_ste": "gather",
        "msa_select_blocks": "gather",
        "permute": "transpose",
        "rearrange": "transpose",
        "take": "gather",
    }.items():
        assert bc.benchmark_source_for(alias) == bc.benchmark_source_for(canonical)
