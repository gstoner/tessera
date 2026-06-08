"""CI-stable harness for ``benchmarks/apple_gpu/benchmark_ga_ebm.py``.

Drives the GA + EBM end-to-end benchmark with a CI-friendly rep count
and deterministic seeds, then validates:

  - Report envelope: counts for GA + EBM (split native vs python_ref)
    + workload rows; compile_time_ms separated from per-row dispatch;
    manifest-driven `apple_gpu_status` per EBM row.
  - GA stack walk: 17 fused MSL kernels, each carrying the symbol
    that `clifford_manifest_for` resolves.
  - Native EBM stack walk: 6 fused kernels (inner_step, refinement,
    langevin_step, decode_init, bivector_langevin, sphere_langevin),
    each carrying the manifest-resolved symbol.
  - Python-ref EBM rows: 8 entries (incl. inner_step alongside its
    native counterpart for speedup comparison).
  - Workload mode: 4 rows total (GA pipeline + EBT-tiny, each in
    apple_gpu + python_ref variants); namespace=workload; mode is
    `fused_chain` / `reference_chain`; correctness vs Python reference.
  - Timing methodology: p10/p50/p90/min/max present; p10 ≤ p50 ≤ p90;
    median == latency_ms; compile_time stays in the envelope.
  - Determinism: re-runs reproduce ok-verdicts + correctness errors.
  - Manifest cross-check: native ops report `apple_gpu=fused`,
    everything else `apple_gpu=planned`.
  - CLI flags: --workloads-only / --primitives-only filter correctly.

Graceful non-Darwin: GA + native EBM + workload-apple_gpu rows skip;
the 8 Python-ref EBM rows + 2 Python-ref workload rows still emit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "benchmarks" / "apple_gpu"
sys.path.insert(0, str(BENCH_DIR))
import benchmark_ga_ebm as bench  # noqa: E402

from tessera.compiler import backend_manifest as bm  # noqa: E402


EXPECTED_GA_OPS = {
    "clifford_reverse", "clifford_grade_involution", "clifford_conjugate",
    "clifford_hodge_star", "clifford_exp", "clifford_log",
    "clifford_norm",
    "clifford_geometric_product", "clifford_wedge",
    "clifford_left_contraction", "clifford_rotor_sandwich",
    "clifford_inner", "clifford_grade_projection",
    "clifford_ext_deriv", "clifford_vec_deriv", "clifford_codiff",
    "clifford_integral",
}

# EBM ops that ship native fused MSL kernels on Apple GPU.
# Native EBM ops the benchmark emits as per-primitive rows (one
# `apple_gpu` row per op).  `ebm_ebt_tiny` ships a fused MSL kernel
# but is used only through the EBT-tiny workload — not as a separate
# per-primitive row — so it lives in `ALL_NATIVE_EBM_MANIFEST_KEYS`
# below rather than this set.
EXPECTED_NATIVE_EBM_OPS = {
    "ebm_inner_step", "ebm_refinement", "ebm_langevin_step",
    "ebm_decode_init", "ebm_bivector_langevin", "ebm_sphere_langevin",
    "ebm_self_verify", "ebm_energy",
    # 2026-05-18 9/9 closure — stable logsumexp on a precomputed
    # energies array.
    "ebm_partition_exact",
}

# All EBM ops that have a fused MSL kernel in the manifest — this set
# is the source-of-truth for the manifest-completeness gate and
# includes ``ebm_ebt_tiny`` (the workload-only optimization).
ALL_NATIVE_EBM_MANIFEST_KEYS = (
    EXPECTED_NATIVE_EBM_OPS
    | {"ebm_ebt_tiny"}
    # M6 Step 4 (2026-05-18): on-device Philox variant of langevin_step.
    | {"ebm_langevin_step_philox"}
    # S-series #3 (2026-06-02): manifold Langevin steps now ship real fused
    # Apple GPU kernels (sphere: dedicated MSL; bivector: reuses the affine
    # ebm_langevin_step kernel on grade-2 coeffs).
    | {"ebm_sphere_langevin_step", "ebm_bivector_langevin_step"}
)

# EBM ops that still have no native dispatch.  Empty after the 9/9
# closure — every EBM primitive now ships a fused MSL kernel.
EXPECTED_PYTHON_ONLY_EBM_OPS = set()

# Every EBM op gets a python_ref row EXCEPT `ebm_refinement` — that's
# the multi-iteration native wrapper around `inner_step`, with no
# distinct Python entry-point (callers would just call `inner_step`
# in a loop). The Python comparison for refinement-style work is the
# `ebt_tiny_refinement` workload row.
EXPECTED_EBM_PYTHON_ROWS = (
    (EXPECTED_NATIVE_EBM_OPS - {"ebm_refinement"})
    | EXPECTED_PYTHON_ONLY_EBM_OPS
)

EXPECTED_WORKLOAD_OPS = {
    "ga_feature_pipeline",
    "ebt_tiny_refinement",
    # 2026-05-17 fused GA+EBM workload — exp → rotor_sandwich → ebt_tiny.
    "rotor_conditioned_ebt",
}

REQUIRED_ROW_FIELDS = {
    "backend", "namespace", "op", "shape", "dtype", "mode", "reps",
    "latency_ms", "stdev_ms",
    "p10_ms", "p50_ms", "p90_ms", "min_ms", "max_ms",
    "max_abs_err", "tolerance", "ok",
    "device", "tessera_version",
    "variant_kind", "compiler_path", "executor", "runtime_status",
    "execution_kind",
}

REQUIRED_ENVELOPE_FIELDS = {
    "runs", "ga_primitives_count", "ebm_paths_count",
    "ebm_native_apple_gpu_count", "native_ebm_ops", "workload_count",
    "composite_count", "ebt_sweep_count", "ebt_sweep_summary",
    "jit_bridge_count",
    "compile_time_ms", "skipped_apple_gpu",
    "device", "tessera_version", "reps",
}


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> dict:
    """Build the GA/EBM benchmark report once for all assertions."""
    tmp_dir = tmp_path_factory.mktemp("ga_ebm_bench")
    return bench.run_report(reps=bench.DEFAULT_REPS_CI, tmp_dir=tmp_dir)


def _rows(report: dict, predicate) -> list[dict]:
    return [r for r in report["runs"] if predicate(r)]


def _ga_rows(report):
    return _rows(report, lambda r: r["namespace"] == "ga")


def _ebm_rows(report):
    return _rows(report, lambda r: r["namespace"] == "ebm")


def _workload_rows(report):
    return _rows(report, lambda r: r["namespace"] == "workload")


def _ebm_native_rows(report):
    return _rows(report, lambda r: r["namespace"] == "ebm"
                                 and r["backend"] == "apple_gpu")


def _ebm_python_rows(report):
    return _rows(report, lambda r: r["namespace"] == "ebm"
                                 and r["backend"] == "python_ref")


def _workload_native_rows(report):
    return _rows(report, lambda r: r["namespace"] == "workload"
                                 and r["backend"] == "apple_gpu")


def _workload_python_rows(report):
    return _rows(report, lambda r: r["namespace"] == "workload"
                                 and r["backend"] == "python_ref")


def _composite_rows(report):
    return _rows(report, lambda r: r["namespace"] == "composite")


def _apple_gpu_available(report: dict) -> bool:
    return report.get("skipped_apple_gpu") is None


# ---------------------------------------------------------------------------
# Envelope checks (cross-platform)
# ---------------------------------------------------------------------------

def test_report_envelope_fields_present(report: dict) -> None:
    missing = REQUIRED_ENVELOPE_FIELDS - report.keys()
    assert not missing, f"envelope missing fields: {missing}"
    assert isinstance(report["compile_time_ms"], (int, float))
    assert report["compile_time_ms"] >= 0.0
    assert report["reps"] == bench.DEFAULT_REPS_CI


def test_every_row_has_required_schema_fields(report: dict) -> None:
    for row in report["runs"]:
        missing = REQUIRED_ROW_FIELDS - row.keys()
        assert not missing, f"row {row['op']} missing: {missing}"


def test_ebm_value_target_row_distinguishes_executor() -> None:
    def dispatch():
        return None

    row = bench.run_ebm_apple_value_path(
        "ebm_energy", "B=2,D=3/value_ir", dispatch, 0.0,
        "tessera_apple_gpu_ebm_energy_quadratic_value_f32",
        tolerance=1.0e-6, reps=1, device="test-device",
        version="test-version")
    missing = REQUIRED_ROW_FIELDS - row.keys()
    assert not missing
    assert row["backend"] == "apple_gpu_value_target_ir"
    assert row["mode"] == "value_target_ir"
    assert row["executor"] == "apple_gpu_value_target_ir"
    assert row["runtime_status"] == "success"
    assert row["execution_kind"] == "native_gpu"
    assert row["variant_kind"] == "apple_gpu_value_target_ir"
    assert row["compiler_path"] == "apple_value_target_ir"
    assert row["symbol"] == (
        "tessera_apple_gpu_ebm_energy_quadratic_value_f32")


def test_value_target_executor_claim_requires_numerical_success() -> None:
    """Stage 16F: value rows cannot hard-code native_gpu/success/executor.

    A value dispatch whose numerical comparison fails may still be represented
    as a row, but it must not carry the executable claim triple.
    """
    row = bench.run_ebm_apple_value_path(
        "ebm_energy", "B=2,D=3/value_ir", lambda: None, 1.0,
        "tessera_apple_gpu_ebm_energy_quadratic_value_f32",
        tolerance=1.0e-6, reps=1, device="test-device",
        version="test-version")
    assert row["variant_kind"] == "apple_gpu_value_target_ir"
    assert row["ok"] is False
    assert row["executor"] is None
    assert row["runtime_status"] == "numerical_mismatch"
    assert row["execution_kind"] == "unknown"


def test_compiler_visible_reference_row_never_claims_gpu_executor() -> None:
    row = bench.run_compiler_visible_reference_path(
        "ebm", "ebm_energy", "B=2,D=3/value_ir", lambda: None, 0.0,
        "tessera_apple_gpu_ebm_energy_quadratic_value_f32",
        tolerance=1.0e-6, reps=1, device="test-device",
        version="test-version")
    assert row["variant_kind"] == "compiler_visible_reference"
    assert row["compiler_path"] == "apple_value_target_ir"
    assert row["executor"] == "python_reference"
    assert row["runtime_status"] == "reference"
    assert row["execution_kind"] == "reference_cpu"


def test_stage16f_value_claims_are_row_kind_scoped(report: dict) -> None:
    allowed_native_triple = {
        "executor": "apple_gpu_value_target_ir",
        "runtime_status": "success",
        "execution_kind": "native_gpu",
    }
    assert any(r["variant_kind"] == "compiler_visible_reference"
               for r in report["runs"])
    for row in report["runs"]:
        carries_value_executor = row.get("executor") == allowed_native_triple["executor"]
        carries_native_success = (
            row.get("runtime_status") == allowed_native_triple["runtime_status"]
            and row.get("execution_kind") == allowed_native_triple["execution_kind"]
        )
        if carries_value_executor or carries_native_success:
            assert row["variant_kind"] == "apple_gpu_value_target_ir", row
            assert row["backend"] == "apple_gpu_value_target_ir", row
            assert row["compiler_path"] == "apple_value_target_ir", row
            assert row["ok"] is True, row
            assert row["max_abs_err"] <= row["tolerance"], row
        elif row["variant_kind"] == "compiler_visible_reference":
            assert row["executor"] == "python_reference"
            assert row["runtime_status"] == "reference"
            assert row["execution_kind"] == "reference_cpu"


def test_stage17_composite_rows_present_and_honest(report: dict) -> None:
    rows = _composite_rows(report)
    assert report["composite_count"] == len(rows) == 3
    ops = {row["op"] for row in rows}
    assert ops == {
        "composite_ebt_tiny_refinement",
        "composite_manifold_ebm",
        "composite_ga_feature_pipeline",
    }
    for row in rows:
        assert row["backend"] == "compiler_visible_reference"
        assert row["variant_kind"] == "compiler_visible_reference"
        assert row["compiler_path"] == "apple_value_target_ir"
        assert row["executor"] == "python_reference"
        assert row["runtime_status"] == "reference"
        assert row["execution_kind"] == "reference_cpu"
        assert row["composite_status"] == "multi_call_value_ir_gated"
        assert row["multi_call_executor"] is None
        assert row["ok"] is True
        assert row["value_call_count"] == len(row["value_calls"])
        assert row["value_call_count"] >= 2
        assert row["symbols"] == [call["symbol"] for call in row["value_calls"]]


def test_stage17_composites_do_not_claim_value_executor(report: dict) -> None:
    for row in _composite_rows(report):
        assert row["executor"] != "apple_gpu_value_target_ir"
        assert row["runtime_status"] != "success"
        assert row["execution_kind"] != "native_gpu"
        assert not any(
            status == "executable_multi_call"
            for status in row["component_value_status"].values()
        )


def test_stage17_ebt_tiny_composite_contract(report: dict) -> None:
    rows = [r for r in _composite_rows(report)
            if r["op"] == "composite_ebt_tiny_refinement"]
    assert len(rows) == 1
    row = rows[0]
    assert row["component_ops"] == [
        "ebm_decode_init",
        "ebm_refinement",
        "ebm_energy_quadratic",
        "ebm_self_verify",
    ]
    assert row["component_value_status"]["ebm_refinement"] == (
        "executable_single_call")
    assert row["component_value_status"]["ebm_energy_quadratic"] == (
        "executable_single_call")
    assert row["component_value_status"]["ebm_decode_init"] == (
        "compiler_visible_gated")
    assert row["component_value_status"]["ebm_self_verify"] == (
        "compiler_visible_gated")
    assert row["max_abs_err"] <= row["tolerance"]
    assert len(row["contract_metrics"]["winner_indices"]) == 3


def test_stage17_manifold_composite_contract(report: dict) -> None:
    row = [r for r in _composite_rows(report)
           if r["op"] == "composite_manifold_ebm"][0]
    assert row["component_ops"] == [
        "ebm_sphere_langevin_step",
        "ebm_bivector_langevin_step",
    ]
    metrics = row["contract_metrics"]
    assert metrics["sphere_norm_error"] <= row["tolerance"]
    assert metrics["bivector_grade_leakage"] <= row["tolerance"]


def test_stage17_ga_feature_composite_contract(report: dict) -> None:
    row = [r for r in _composite_rows(report)
           if r["op"] == "composite_ga_feature_pipeline"][0]
    assert row["component_ops"] == [
        "clifford_geometric_product",
        "clifford_grade_projection",
        "clifford_rotor_sandwich",
    ]
    assert row["component_value_status"]["clifford_geometric_product"] == (
        "executable_single_call")
    assert row["component_value_status"]["clifford_grade_projection"] == (
        "compiler_visible_gated")
    assert row["component_value_status"]["clifford_rotor_sandwich"] == (
        "compiler_visible_gated")
    assert row["contract_metrics"]["projected_non_even_max"] <= row["tolerance"]
    assert row["contract_metrics"]["batch"] == bench._BATCH


def test_timing_percentiles_consistent_for_every_row(report: dict) -> None:
    for row in report["runs"]:
        p10, p50, p90 = row["p10_ms"], row["p50_ms"], row["p90_ms"]
        op = row["op"]
        assert p10 <= p50 <= p90, f"{op}: p10/p50/p90 not monotone"
        assert row["min_ms"] <= row["max_ms"]
        assert row["min_ms"] <= p10
        assert row["max_ms"] >= p90
        assert row["latency_ms"] == p50


# ---------------------------------------------------------------------------
# EBM coverage — Python-ref always; native gated on runtime
# ---------------------------------------------------------------------------

def test_python_ref_ebm_rows_always_emitted(report: dict) -> None:
    ops = {r["op"] for r in _ebm_python_rows(report)}
    assert ops == EXPECTED_EBM_PYTHON_ROWS


@pytest.mark.parametrize("op", sorted(EXPECTED_EBM_PYTHON_ROWS))
def test_each_python_ebm_row_correct(report: dict, op: str) -> None:
    row = next(r for r in _ebm_python_rows(report) if r["op"] == op)
    assert row["ok"] is True, f"{op} failed: err={row['max_abs_err']}"
    assert row["backend"] == "python_ref"
    assert row["mode"] == "reference"
    assert row["namespace"] == "ebm"


@pytest.mark.parametrize("op", sorted(EXPECTED_PYTHON_ONLY_EBM_OPS))
def test_python_only_ebm_op_marked_planned_in_manifest(
        report: dict, op: str) -> None:
    """Ops without native dispatch must report apple_gpu_status=planned."""
    row = next(r for r in _ebm_python_rows(report) if r["op"] == op)
    assert row["apple_gpu_status"] == "planned", (
        f"{op}: manifest now reports {row['apple_gpu_status']!r} — "
        f"promote to native if a kernel landed"
    )


@pytest.mark.parametrize("op", sorted(EXPECTED_NATIVE_EBM_OPS))
def test_natively_promoted_ebm_op_marked_fused_in_manifest(op: str) -> None:
    by_target = {e.target: e for e in bm.ebm_manifest_for(op)}
    assert by_target["apple_gpu"].status == "fused"


# ---------------------------------------------------------------------------
# Apple-GPU gates
# ---------------------------------------------------------------------------

def test_apple_gpu_rows_emitted_when_runtime_available(report: dict) -> None:
    if not _apple_gpu_available(report):
        pytest.skip(f"apple_gpu unavailable: {report.get('skipped_apple_gpu')}")
    ga_ops = {r["op"] for r in _ga_rows(report)}
    assert ga_ops == EXPECTED_GA_OPS
    assert report["ga_primitives_count"] == len(EXPECTED_GA_OPS)


def test_six_native_ebm_rows_emitted_when_runtime_available(report: dict) -> None:
    if not _apple_gpu_available(report):
        pytest.skip(f"apple_gpu unavailable: {report.get('skipped_apple_gpu')}")
    native_ops = {r["op"] for r in _ebm_native_rows(report)}
    assert native_ops == EXPECTED_NATIVE_EBM_OPS, (
        f"native set mismatch: have {native_ops}, expected {EXPECTED_NATIVE_EBM_OPS}"
    )
    assert set(report["native_ebm_ops"]) == EXPECTED_NATIVE_EBM_OPS
    assert report["ebm_native_apple_gpu_count"] == len(EXPECTED_NATIVE_EBM_OPS)


def test_non_apple_host_records_skip_reason(report: dict) -> None:
    if _apple_gpu_available(report):
        pytest.skip("apple_gpu runtime is available on this host")
    assert isinstance(report["skipped_apple_gpu"], str)
    assert report["ga_primitives_count"] == 0
    assert report["ebm_native_apple_gpu_count"] == 0


@pytest.mark.parametrize("op", sorted(EXPECTED_GA_OPS))
def test_each_ga_row_passes_correctness_gate(report: dict, op: str) -> None:
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    row = next(r for r in _ga_rows(report) if r["op"] == op)
    assert row["ok"] is True
    assert row["backend"] == "apple_gpu"
    assert row["mode"] == "fused"


@pytest.mark.parametrize("op", sorted(EXPECTED_GA_OPS))
def test_each_ga_row_carries_manifest_resolved_symbol(
        report: dict, op: str) -> None:
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    row = next(r for r in _ga_rows(report) if r["op"] == op)
    assert row["symbol"] == bench._resolve_symbol(op)


@pytest.mark.parametrize("op", sorted(EXPECTED_NATIVE_EBM_OPS))
def test_each_native_ebm_row_passes_correctness_and_manifest_gate(
        report: dict, op: str) -> None:
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    row = next(r for r in _ebm_native_rows(report) if r["op"] == op)
    assert row["ok"] is True, (
        f"{op}: native EBM diverged — err={row['max_abs_err']}"
    )
    assert row["backend"] == "apple_gpu"
    assert row["mode"] == "fused"
    assert row["namespace"] == "ebm"
    assert row["apple_gpu_status"] == "fused"
    # Bridge-trace proof of dispatch: the runner opens a one-shot trace
    # span around the timed dispatch and confirms a route matching this
    # op was recorded.  A False here means the public API silently fell
    # back to numpy and the row was already degraded to python_ref.
    assert row["dispatched_on_gpu"] is True, (
        f"{op}: bridge trace did not record a native dispatch — the "
        f"public API likely fell back to numpy"
    )
    # Reported symbol must equal the one the manifest carries.
    spec = bm._EBM_APPLE_GPU_FUSED[op]
    assert row["symbol"] == spec["symbol"]


# ---------------------------------------------------------------------------
# Workload mode — composite chains, not single primitives
# ---------------------------------------------------------------------------

def test_workload_python_rows_always_emitted(report: dict) -> None:
    """Python-ref workload rows are emitted on every host (graceful skip)."""
    ops = {r["op"] for r in _workload_python_rows(report)}
    assert ops == EXPECTED_WORKLOAD_OPS


def test_workload_apple_gpu_rows_emitted_when_runtime_available(
        report: dict) -> None:
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    ops = {r["op"] for r in _workload_native_rows(report)}
    assert ops == EXPECTED_WORKLOAD_OPS


def test_workload_count_envelope_matches_rows(report: dict) -> None:
    assert report["workload_count"] == len(_workload_rows(report))


@pytest.mark.parametrize("op", sorted(EXPECTED_WORKLOAD_OPS))
def test_each_workload_python_row_correct(report: dict, op: str) -> None:
    row = next(r for r in _workload_python_rows(report) if r["op"] == op)
    assert row["ok"] is True
    assert row["mode"] == "reference_chain"
    assert row["namespace"] == "workload"


@pytest.mark.parametrize("op", sorted(EXPECTED_WORKLOAD_OPS))
def test_each_workload_apple_gpu_row_correct_and_chains_known_symbols(
        report: dict, op: str) -> None:
    """Native workload rows carry the multi-symbol chain in `symbols`."""
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    row = next(r for r in _workload_native_rows(report) if r["op"] == op)
    assert row["ok"] is True, (
        f"{op}: native workload diverged — err={row['max_abs_err']}"
    )
    assert row["backend"] == "apple_gpu"
    assert row["mode"] == "fused_chain"
    assert isinstance(row["symbols"], list)
    assert len(row["symbols"]) >= 1
    # Every symbol referenced should either be an exported runtime symbol
    # (tessera_apple_gpu_*) or a documented host-side helper.
    for sym in row["symbols"]:
        assert sym.startswith("tessera_apple_gpu_") or sym.startswith("ebm.")


def test_ga_feature_pipeline_chains_three_known_msl_kernels(
        report: dict) -> None:
    """The GA workload must reference exp → rotor_sandwich → norm — the
    chain documented in the benchmark module."""
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    row = next(r for r in _workload_native_rows(report)
               if r["op"] == "ga_feature_pipeline")
    syms = row["symbols"]
    assert any("clifford_exp" in s for s in syms)
    assert any("clifford_rotor_sandwich" in s for s in syms)
    assert any("clifford_norm" in s for s in syms)


def test_ebt_tiny_workload_uses_fused_kernel(report: dict) -> None:
    """The EBT-tiny workload must reference the fused single-dispatch
    kernel (after the 2026-05-17 optimization that collapsed
    refinement + self_verify into one MSL dispatch)."""
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    row = next(r for r in _workload_native_rows(report)
               if r["op"] == "ebt_tiny_refinement")
    syms = row["symbols"]
    assert any("ebt_tiny_refinement_argmin_f32" in s for s in syms), (
        f"workload should now use the fused kernel; got symbols={syms}"
    )
    assert any("ebm.ebt_tiny" in s for s in syms)


# ---------------------------------------------------------------------------
# Manifest cross-check
# ---------------------------------------------------------------------------

def test_all_native_ebm_ops_in_fused_manifest_table() -> None:
    """``_EBM_APPLE_GPU_FUSED`` must list every native-EBM op the
    benchmark expects (including the workload-only ``ebm_ebt_tiny``
    fused kernel) — keeps the manifest in lock-step with the runtime."""
    assert set(bm._EBM_APPLE_GPU_FUSED.keys()) == ALL_NATIVE_EBM_MANIFEST_KEYS


def test_manifest_for_routes_ebm_prefix_to_ebm_table() -> None:
    via_router = bm.manifest_for("ebm_inner_step")
    via_table = bm.ebm_manifest_for("ebm_inner_step")
    assert [(e.target, e.status) for e in via_router] == \
           [(e.target, e.status) for e in via_table]


# ---------------------------------------------------------------------------
# CLI filter flags
# ---------------------------------------------------------------------------

def test_primitives_only_flag_drops_workload_rows(tmp_path) -> None:
    r = bench.run_report(reps=bench.DEFAULT_REPS_CI, tmp_dir=tmp_path,
                          include_primitives=True,
                          include_workloads=False)
    assert _workload_rows(r) == []
    assert r["workload_count"] == 0


def test_workloads_only_flag_drops_primitive_rows(tmp_path) -> None:
    r = bench.run_report(reps=bench.DEFAULT_REPS_CI, tmp_dir=tmp_path,
                          include_primitives=False,
                          include_workloads=True)
    assert _ga_rows(r) == []
    assert _ebm_rows(r) == []
    assert r["ga_primitives_count"] == 0
    assert r["ebm_paths_count"] == 0
    assert r["workload_count"] > 0


# ---------------------------------------------------------------------------
# EBT-tiny break-even sweep — opt-in via --ebt-sweep / include_ebt_sweep
# ---------------------------------------------------------------------------

def test_default_report_omits_ebt_sweep(report: dict) -> None:
    """The default report (no `include_ebt_sweep`) emits zero sweep rows
    and a null sweep summary.  Keeps CI snappy."""
    assert report["ebt_sweep_count"] == 0
    assert report["ebt_sweep_summary"] is None


def test_ebt_sweep_emits_pair_per_shape(tmp_path, report: dict) -> None:
    """With sweep enabled, each (B, K, D, T) point yields one
    apple_gpu + one python_ref row when the runtime is available,
    one python_ref-only row when not."""
    r = bench.run_report(reps=bench.DEFAULT_REPS_CI, tmp_dir=tmp_path,
                          include_primitives=False,
                          include_workloads=False,
                          include_ebt_sweep=True,
                          ebt_sweep_points=((4, 8, 6, 4), (8, 16, 32, 4)))
    sweep_rows = [row for row in r["runs"] if row["op"] == "ebt_tiny_sweep"]
    if _apple_gpu_available(report):
        assert len(sweep_rows) == 4   # 2 shapes × {apple_gpu, python_ref}
    else:
        assert len(sweep_rows) == 2   # 2 shapes × python_ref only
    assert r["ebt_sweep_count"] == len(sweep_rows)
    summary = r["ebt_sweep_summary"]
    assert summary is not None
    assert "table" in summary
    assert "first_native_win_shape" in summary


def test_ebt_sweep_summary_shape_pairs_are_speedups(tmp_path) -> None:
    """When both backends emit rows for a shape, the summary entry
    must include a `speedup` field equal to python_ms / native_ms."""
    r = bench.run_report(reps=bench.DEFAULT_REPS_CI, tmp_dir=tmp_path,
                          include_primitives=False,
                          include_workloads=False,
                          include_ebt_sweep=True,
                          ebt_sweep_points=((4, 8, 6, 4),))
    summary = r["ebt_sweep_summary"]
    assert summary is not None
    for entry in summary["table"]:
        if "native_ms" in entry and "python_ms" in entry:
            assert entry["speedup"] == pytest.approx(
                entry["python_ms"] / entry["native_ms"], rel=1e-9)
            assert entry["native_wins"] == (entry["speedup"] >= 1.0)


# ---------------------------------------------------------------------------
# Integration gap — `tessera.ga.inner` + `tessera.ebm.inner_step` route
# through `tessera._apple_gpu_dispatch` rather than benchmark-local ctypes.
# ---------------------------------------------------------------------------

def test_apple_gpu_dispatcher_singleton_caches_runtime(tmp_path) -> None:
    """Two consecutive calls return the same `CDLL` handle (no recompile)."""
    from tessera import _apple_gpu_dispatch as disp
    disp._reset_for_testing()
    h1 = disp.apple_gpu_runtime()
    h2 = disp.apple_gpu_runtime()
    assert h1 is h2  # cached singleton
    if h1 is None:
        # Non-Darwin / missing toolchain — skip reason must be populated.
        assert disp.apple_gpu_skip_reason() is not None


def test_ga_inner_dispatches_through_apple_gpu_when_available() -> None:
    """`tessera.ga.inner` on a batched Cl(3,0) Multivector returns the
    same result as the per-element Python loop reference, proving the
    GPU fast path (when active) doesn't drift from the reference."""
    import numpy as np
    import tessera.ga as ga
    from tessera import _apple_gpu_dispatch as disp

    a = ga.Cl(3, 0)
    rng = np.random.RandomState(99)
    A = rng.randn(16, 8).astype(np.float32)
    B = rng.randn(16, 8).astype(np.float32)
    result = ga.inner(ga.Multivector(A, a), ga.Multivector(B, a))
    ref = np.array([
        float(ga.inner(ga.Multivector(A[i], a), ga.Multivector(B[i], a)))
        for i in range(16)
    ])
    err = float(np.abs(np.asarray(result) - ref).max())
    if disp.apple_gpu_available():
        # GPU fast path active — bit-exact agreement at fp32.
        assert err <= 5e-6
    else:
        # Pure Python path — should still match self.
        assert err <= 1e-6


def test_ebm_inner_step_dispatches_through_apple_gpu_when_available() -> None:
    """`tessera.ebm.inner_step` with f32 inputs and no noise routes
    through the dispatcher; the output matches `y - eta * grad`."""
    import numpy as np
    import tessera.ebm as ebm

    rng = np.random.RandomState(101)
    y = rng.randn(32, 8).astype(np.float32)
    g = rng.randn(32, 8).astype(np.float32)
    out = ebm.inner_step(y, g, eta=0.05)
    ref = (y - 0.05 * g).astype(np.float32)
    assert float(np.abs(out - ref).max()) <= 1e-6


# ---------------------------------------------------------------------------
# Public-API GPU fast-path coverage — six more ops promoted from
# benchmark-local ctypes to `tessera.ga.*` / `tessera.ebm.*`.
# ---------------------------------------------------------------------------

def test_ga_exp_mv_public_api_matches_per_element_reference() -> None:
    import numpy as np
    import tessera.ga as ga
    a = ga.Cl(3, 0)
    rng = np.random.RandomState(42)
    A = rng.randn(8, 8).astype(np.float32)
    batched = ga.exp_mv(ga.Multivector(A, a)).coefficients
    elemwise = np.stack([
        ga.exp_mv(ga.Multivector(A[i], a)).coefficients for i in range(8)
    ])
    assert float(np.abs(batched - elemwise).max()) <= 5e-6


def test_ga_rotor_sandwich_public_api_matches_per_element_reference() -> None:
    import numpy as np
    import tessera.ga as ga
    a = ga.Cl(3, 0)
    rng = np.random.RandomState(43)
    R = rng.randn(8, 8).astype(np.float32) * 0.3
    X = rng.randn(8, 8).astype(np.float32)
    batched = ga.rotor_sandwich(ga.Multivector(R, a),
                                  ga.Multivector(X, a)).coefficients
    elemwise = np.stack([
        ga.rotor_sandwich(ga.Multivector(R[i], a),
                            ga.Multivector(X[i], a)).coefficients
        for i in range(8)
    ])
    assert float(np.abs(batched - elemwise).max()) <= 5e-6


def test_ga_norm_public_api_matches_per_element_reference() -> None:
    import numpy as np
    import tessera.ga as ga
    a = ga.Cl(3, 0)
    rng = np.random.RandomState(44)
    A = rng.randn(8, 8).astype(np.float32)
    batched = np.asarray(ga.norm(ga.Multivector(A, a)))
    elemwise = np.array([
        float(np.asarray(ga.norm(ga.Multivector(A[i], a)))) for i in range(8)
    ])
    assert float(np.abs(batched - elemwise).max()) <= 5e-6


def test_ebm_self_verify_public_api_hard_argmin_matches_reference() -> None:
    import numpy as np
    import tessera.ebm as ebm
    rng = np.random.RandomState(45)
    e = rng.randn(4, 8).astype(np.float32)
    c = rng.randn(4, 8, 16).astype(np.float32)
    out = ebm.self_verify(e, c)
    expected = c[np.arange(4), e.argmin(axis=1)]
    assert float(np.abs(out - expected).max()) == 0.0


def test_ebm_energy_quadratic_public_api_matches_numpy_reference() -> None:
    import numpy as np
    import tessera.ebm as ebm
    rng = np.random.RandomState(46)
    x = rng.randn(8, 4).astype(np.float32)
    y = rng.randn(8, 4).astype(np.float32)
    out = ebm.energy_quadratic(x, y)
    expected = (0.5 * np.sum((x - y) ** 2, axis=1)).astype(np.float32)
    assert float(np.abs(out - expected).max()) <= 5e-6


def test_ebm_decode_init_public_api_with_mean_matches_reference() -> None:
    import numpy as np
    import tessera.ebm as ebm
    from tessera.rng import RNGKey, normal as rng_normal
    key = RNGKey.from_seed(47)
    mean = np.random.RandomState(48).randn(4, 6, 12).astype(np.float32)
    out = ebm.decode_init(np.zeros((4, 12), dtype=np.float32),
                            K=6, init_strategy="noise",
                            rng_key=key, shape=(12,), dtype="fp32",
                            std=0.5, mean=mean)
    noise = rng_normal(key, shape=(4, 6, 12), dtype="fp32", std=1.0)
    expected = (mean + 0.5 * noise).astype(np.float32)
    assert out.shape == (4, 6, 12)
    assert float(np.abs(out - expected).max()) <= 5e-6


def test_ebm_ebt_tiny_public_api_matches_chained_reference() -> None:
    """`ebm.ebt_tiny(...)` returns the same `(B, D)` best-candidate
    matrix as the explicit `refinement → energy → self_verify` chain."""
    import numpy as np
    import tessera.ebm as ebm
    B, K, D, T = 4, 8, 16, 6
    rng = np.random.RandomState(2200)
    y0 = rng.randn(B * K, D).astype(np.float32)
    grad = rng.randn(B * K, D).astype(np.float32)
    eta = 0.03
    fused = ebm.ebt_tiny(y0, grad, eta=eta, T=T, B=B, K=K, D=D)

    # Reference: closed-form refinement (fixed grad) + numpy argmin.
    y_T = (y0 - T * eta * grad).astype(np.float32)
    energies = np.sum(y_T * y_T, axis=1).reshape(B, K)
    candidates = y_T.reshape(B, K, D)
    expected = candidates[np.arange(B), energies.argmin(axis=1)]
    assert fused.shape == (B, D)
    assert float(np.abs(fused - expected).max()) <= 5e-5


@pytest.mark.parametrize("op_name", [
    "reverse", "grade_involution", "conjugate",
])
def test_ga_unary_signflip_public_api_matches_per_element_reference(
        op_name: str) -> None:
    """Sign-flip unaries: reverse / grade_involution / conjugate all
    route through `_try_apple_gpu_unary_8x8_cl30_f32`."""
    import numpy as np
    import tessera.ga as ga
    a = ga.Cl(3, 0)
    fn = getattr(ga, op_name)
    rng = np.random.RandomState(50)
    A = rng.randn(8, 8).astype(np.float32)
    batched = fn(ga.Multivector(A, a)).coefficients
    elemwise = np.stack([
        fn(ga.Multivector(A[i], a)).coefficients for i in range(8)
    ])
    assert float(np.abs(batched - elemwise).max()) <= 5e-6


def test_ga_hodge_star_public_api_matches_per_element_reference() -> None:
    import numpy as np
    import tessera.ga as ga
    a = ga.Cl(3, 0)
    rng = np.random.RandomState(51)
    A = rng.randn(8, 8).astype(np.float32)
    batched = ga.hodge_star(ga.Multivector(A, a)).coefficients
    elemwise = np.stack([
        ga.hodge_star(ga.Multivector(A[i], a)).coefficients for i in range(8)
    ])
    assert float(np.abs(batched - elemwise).max()) <= 5e-6


def test_ga_log_mv_public_api_matches_per_element_reference() -> None:
    import numpy as np
    import tessera.ga as ga
    a = ga.Cl(3, 0)
    rng = np.random.RandomState(52)
    A = rng.randn(8, 8).astype(np.float32) * 0.3
    batched = ga.log_mv(ga.Multivector(A, a)).coefficients
    elemwise = np.stack([
        ga.log_mv(ga.Multivector(A[i], a)).coefficients for i in range(8)
    ])
    assert float(np.abs(batched - elemwise).max()) <= 5e-6


@pytest.mark.parametrize("op_name", [
    "geometric_product", "wedge", "left_contraction",
])
def test_ga_binary_8x8_public_api_matches_per_element_reference(
        op_name: str) -> None:
    """Binary 8×8→8 ops all route through
    `_try_apple_gpu_binary_8x8_cl30_f32`."""
    import numpy as np
    import tessera.ga as ga
    a = ga.Cl(3, 0)
    fn = getattr(ga, op_name)
    rng = np.random.RandomState(53)
    A = rng.randn(8, 8).astype(np.float32)
    B = rng.randn(8, 8).astype(np.float32)
    batched = fn(ga.Multivector(A, a), ga.Multivector(B, a)).coefficients
    elemwise = np.stack([
        fn(ga.Multivector(A[i], a),
           ga.Multivector(B[i], a)).coefficients for i in range(8)
    ])
    assert float(np.abs(batched - elemwise).max()) <= 5e-6


def test_ga_grade_projection_public_api_matches_per_element_reference() -> None:
    import numpy as np
    import tessera.ga as ga
    a = ga.Cl(3, 0)
    rng = np.random.RandomState(54)
    A = rng.randn(8, 8).astype(np.float32)
    grades = {0, 2}                      # even subalgebra
    batched = ga.grade_projection(ga.Multivector(A, a), grades).coefficients
    elemwise = np.stack([
        ga.grade_projection(ga.Multivector(A[i], a),
                              grades).coefficients for i in range(8)
    ])
    assert float(np.abs(batched - elemwise).max()) == 0.0


@pytest.mark.parametrize("op_name", ["ext_deriv", "vec_deriv", "codiff"])
def test_ga_field_op_public_api_matches_interior_reference(
        op_name: str) -> None:
    """Field ops (ext_deriv / vec_deriv / codiff) match the numpy
    reference on interior cells (kernel zero-pads boundaries)."""
    import numpy as np
    import tessera.ga as ga
    from tessera.ga.calculus import MultivectorField
    a = ga.Cl(3, 0)
    fn = getattr(__import__("tessera.ga.calculus",
                             fromlist=[op_name]), op_name)
    rng = np.random.RandomState(55)
    F32 = rng.randn(5, 6, 7, 8).astype(np.float32)
    F64 = F32.astype(np.float64)
    spacing = (0.1, 0.2, 0.25)
    field32 = MultivectorField(F32, a, spacing=spacing)
    field64 = MultivectorField(F64, a, spacing=spacing)
    gpu = fn(field32).values
    ref = fn(field64).values.astype(np.float32)
    sl = (slice(1, 4), slice(1, 5), slice(1, 6), slice(None))
    assert float(np.abs(gpu[sl] - ref[sl]).max()) <= 1e-3


def test_ga_integral_public_api_matches_weighted_sum() -> None:
    import numpy as np
    import tessera.ga as ga
    from tessera.ga.calculus import MultivectorField, integral
    from tessera.ga.manifold import Euclidean
    a = ga.Cl(3, 0)
    rng = np.random.RandomState(56)
    D0, D1, D2 = 4, 4, 4
    F = rng.randn(D0, D1, D2, 8).astype(np.float32)
    field = MultivectorField(F, a, spacing=(1.0, 1.0, 1.0))
    mf = Euclidean(bounds=[(0, 1)] * 3, resolution=(D0, D1, D2))
    out = integral(field, mf)
    expected = np.einsum("i,ij->j", mf.weights(),
                          F.reshape(-1, 8)).astype(np.float32)
    assert float(np.abs(out - expected).max()) <= 1e-3


def test_ebm_langevin_step_public_api_matches_closed_form() -> None:
    import numpy as np
    import tessera.ebm as ebm
    from tessera.rng import RNGKey
    y = np.random.RandomState(57).randn(16, 4).astype(np.float32)
    key = RNGKey.from_seed(57)
    grad_fn = lambda yy: 2.0 * yy
    energy_fn = lambda yy: np.sum(yy * yy, axis=1)
    out, _ = ebm.langevin_step(y, energy_fn, eta=0.01, temperature=0.0,
                                 rng_key=key, grad_fn=grad_fn)
    # T=0 ⇒ noise contribution vanishes, pure GD.
    expected = (y - 0.01 * 2.0 * y).astype(np.float32)
    assert float(np.abs(out - expected).max()) <= 5e-6


def test_ebm_bivector_langevin_public_api_matches_python_path() -> None:
    """Bivector Langevin should produce identical output through the
    public API regardless of whether the GPU fast path activates."""
    import numpy as np
    import tessera.ga as ga
    import tessera.ebm as ebm
    from tessera.rng import RNGKey
    a = ga.Cl(3, 0)
    coeffs = np.zeros(8, dtype=np.float32)
    coeffs[3] = 0.5
    coeffs[5] = -0.2
    coeffs[6] = 0.3
    state = ga.Multivector(coeffs, a)

    def grad_fn(mv):
        return ga.Multivector(mv.coefficients.copy(), mv.algebra)

    key = RNGKey.from_seed(58)
    out, _ = ebm.bivector_langevin_step(state, lambda m: 0.0,
                                          eta=0.01, temperature=0.0,
                                          rng_key=key, grad_fn=grad_fn)
    # T=0 + grad=state ⇒ result is (1 - eta) * state on the grade-2 blades only.
    for k in (3, 5, 6):
        assert abs(float(out.coefficients[k]) - 0.99 * coeffs[k]) <= 5e-6
    # Non-bivector blades stay zero.
    for k in (0, 1, 2, 4, 7):
        assert abs(float(out.coefficients[k])) <= 5e-6


def test_ebm_sphere_langevin_public_api_matches_python_path() -> None:
    """Sphere Langevin step (T=0) returns the unit-norm retraction
    of `x - eta * tangent_grad`."""
    import numpy as np
    import tessera.ebm as ebm
    from tessera.rng import RNGKey
    x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    grad = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    out, _ = ebm.sphere_langevin_step(x, lambda p: -float(p[0]),
                                        eta=0.005, temperature=0.0,
                                        rng_key=RNGKey.from_seed(59),
                                        grad_fn=lambda p: grad)
    # Tangent projection at x=(1,0,0): grad_tan = grad - <grad,x>x = 0.
    # So x' = x, retract → x.
    assert float(np.abs(out - x).max()) <= 1e-6


def test_ebm_refinement_public_api_matches_closed_form() -> None:
    """`ebm.refinement(y0, grad, eta, T)` runs T inner-step iterations
    in a single fused MSL kernel when GPU is up; closed-form numpy
    fallback `y_T = y0 - T*eta*grad` otherwise."""
    import numpy as np
    import tessera.ebm as ebm
    rng = np.random.RandomState(49)
    y0 = rng.randn(32, 16).astype(np.float32)
    grad = rng.randn(32, 16).astype(np.float32)
    eta, T = 0.01, 12
    out = ebm.refinement(y0, grad, eta=eta, T=T)
    expected = (y0 - T * eta * grad).astype(np.float32)
    assert float(np.abs(out - expected).max()) <= 5e-6


def test_workloads_use_public_apis_not_local_ctypes(report: dict) -> None:
    """Workload rows must carry `[via ga.* | ebm.*]` provenance in their
    symbol strings — proves the rewrite to public APIs landed."""
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    by_op = {r["op"]: r for r in _workload_native_rows(report)}
    ga_row = by_op["ga_feature_pipeline"]
    assert any("via ga.exp_mv" in s for s in ga_row["symbols"])
    assert any("via ga.rotor_sandwich" in s for s in ga_row["symbols"])
    assert any("via ga.norm" in s for s in ga_row["symbols"])
    ebt_row = by_op["ebt_tiny_refinement"]
    # After the fused-kernel optimization, ebt_tiny calls the new
    # ``ebm.ebt_tiny`` public API (single MSL dispatch) instead of
    # the refinement+self_verify chain.
    assert any("via ebm.ebt_tiny" in s for s in ebt_row["symbols"])


# ---------------------------------------------------------------------------
# Determinism + serialization + CLI smoke
# ---------------------------------------------------------------------------

def test_determinism_same_seeds_same_correctness_outcome(tmp_path) -> None:
    r1 = bench.run_report(reps=bench.DEFAULT_REPS_CI, tmp_dir=tmp_path / "run1")
    r2 = bench.run_report(reps=bench.DEFAULT_REPS_CI, tmp_dir=tmp_path / "run2")
    by_op_1 = {(r["namespace"], r["op"], r["backend"]): r for r in r1["runs"]}
    by_op_2 = {(r["namespace"], r["op"], r["backend"]): r for r in r2["runs"]}
    assert by_op_1.keys() == by_op_2.keys()
    for key in by_op_1:
        assert by_op_1[key]["ok"] == by_op_2[key]["ok"], f"non-det ok: {key}"
        assert by_op_1[key]["max_abs_err"] == by_op_2[key]["max_abs_err"], (
            f"non-det err for {key}"
        )


def test_report_json_roundtrips(report: dict, tmp_path: Path) -> None:
    out = tmp_path / "ga_ebm_bench.json"
    out.write_text(json.dumps(report, default=float))
    reloaded = json.loads(out.read_text())
    assert len(reloaded["runs"]) == len(report["runs"])


def test_main_writes_parseable_json_file(tmp_path: Path) -> None:
    out = tmp_path / "main_report.json"
    rc = bench.main(["--ci", "--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert "runs" in payload
    # The 8 Python-ref EBM rows + 2 Python-ref workload rows always emit.
    ops = {r["op"] for r in payload["runs"]}
    assert EXPECTED_EBM_PYTHON_ROWS <= ops
    assert EXPECTED_WORKLOAD_OPS <= ops


# ---------------------------------------------------------------------------
# JIT-bridge rows — Python → jit_context → manifest → shared loader → result
# ---------------------------------------------------------------------------

def _jit_bridge_rows(report):
    return _rows(report, lambda r: r.get("namespace") == "jit_bridge")


def test_jit_bridge_rows_emitted_when_runtime_available(report: dict) -> None:
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    rows = _jit_bridge_rows(report)
    ops = {r["op"] for r in rows}
    assert ops == {"clifford_inner", "ebm_inner_step"}
    assert report["jit_bridge_count"] == len(rows)


def test_jit_bridge_rows_carry_routes_with_jit_context(report: dict) -> None:
    """Each jit_bridge row must include a `routes` column populated by
    the bridge's thread-local trace, and the context tag must mark
    it as JIT-driven."""
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    for row in _jit_bridge_rows(report):
        assert "routes" in row
        assert isinstance(row["routes"], list)
        assert len(row["routes"]) >= 1, (
            f"{row['op']}: no routes recorded — bridge trace empty"
        )
        for route in row["routes"]:
            assert route["target"] == "apple_gpu"
            assert route["status"] == "fused"
            assert route["context"] == "jit:apple_gpu"
            assert "tessera_apple_gpu_" in route["symbol"]


def test_jit_bridge_routes_resolve_via_manifest(report: dict) -> None:
    """The symbol recorded in each route must be the one the manifest
    resolves for that op — proves we went through the manifest."""
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    from tessera.compiler import jit_bridge as bridge_mod
    for row in _jit_bridge_rows(report):
        for route in row["routes"]:
            expected = bridge_mod.lookup_apple_gpu_symbol(route["op"])
            assert route["symbol"] == expected, (
                f"{row['op']}: route symbol {route['symbol']!r} "
                f"!= manifest-resolved {expected!r}"
            )


def test_jit_bridge_rows_pass_correctness_gate(report: dict) -> None:
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    for row in _jit_bridge_rows(report):
        assert row["ok"] is True, (
            f"{row['op']}: bridge route diverged from numpy ref — "
            f"err={row['max_abs_err']} > tol={row['tolerance']}"
        )


def test_no_jit_bridge_flag_drops_jit_bridge_rows(tmp_path) -> None:
    r = bench.run_report(reps=bench.DEFAULT_REPS_CI, tmp_dir=tmp_path,
                          include_primitives=False,
                          include_workloads=False,
                          include_jit_bridge=False)
    rows = [row for row in r["runs"] if row.get("namespace") == "jit_bridge"]
    assert rows == []
    assert r["jit_bridge_count"] == 0


# ---------------------------------------------------------------------------
# Compiler vertical slice + GA+EBM fused workload
# ---------------------------------------------------------------------------

def _vertical_slice_rows(report):
    return _rows(report, lambda r: r.get("namespace") == "vertical_slice")


def test_vertical_slice_row_emitted_when_runtime_available(report: dict) -> None:
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    rows = _vertical_slice_rows(report)
    assert len(rows) == 1
    row = rows[0]
    assert row["op"] == "point_cloud_rotor_invariant"
    assert row["backend"] == "apple_gpu"
    assert row["mode"] == "jit_compiled"
    assert row["ok"] is True
    # Compiled artifact embedded in the row.
    assert "plan_hash" in row
    assert row["plan_hash"] != ""
    assert "compiled_artifact" in row
    art = row["compiled_artifact"]
    assert art["target"] == "apple_gpu"
    assert art["dtype"] == "f32"
    # The rotor_sandwich → norm chain fuses into one op (gap #6); every entry
    # must carry the manifest-resolved symbol.
    from tessera.compiler import jit_bridge as bridge_mod
    plan_ops = [e["op"] for e in art["plan"]]
    assert plan_ops == ["clifford_rotor_sandwich_norm"]
    for entry in art["plan"]:
        assert entry["target"] == "apple_gpu"
        assert entry["status"] == "fused"
        assert entry["symbol"] == bridge_mod.lookup_apple_gpu_symbol(entry["op"])


def test_vertical_slice_envelope_count(report: dict) -> None:
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    assert report["vertical_slice_count"] == 1


def test_rotor_conditioned_ebt_workload_emits_apple_gpu_row(report: dict) -> None:
    """The fused GA+EBM workload must emit a native row that
    dispatched on GPU, with all three op families represented."""
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    rows = [r for r in _workload_native_rows(report)
            if r["op"] == "rotor_conditioned_ebt"]
    assert len(rows) == 1
    row = rows[0]
    assert row["ok"] is True
    assert row["dispatched_on_gpu"] is True
    syms = row["symbols"]
    assert any("ga.exp_mv" in s for s in syms)
    assert any("ga.rotor_sandwich" in s for s in syms)
    assert any("ebm.ebt_tiny" in s for s in syms)


def test_compile_time_separated_from_dispatch_time(report: dict) -> None:
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    assert report["compile_time_ms"] > 0.0
    for row in report["runs"]:
        assert "compile_time_ms" not in row
