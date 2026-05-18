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
EXPECTED_NATIVE_EBM_OPS = {
    "ebm_inner_step", "ebm_refinement", "ebm_langevin_step",
    "ebm_decode_init", "ebm_bivector_langevin", "ebm_sphere_langevin",
}

# EBM ops that still have no native dispatch.
EXPECTED_PYTHON_ONLY_EBM_OPS = {
    "ebm_energy", "ebm_self_verify", "ebm_partition_exact",
}

# Every EBM op gets a python_ref row EXCEPT `ebm_refinement` — that's
# the multi-iteration native wrapper around `inner_step`, with no
# distinct Python entry-point (callers would just call `inner_step`
# in a loop). The Python comparison for refinement-style work is the
# `ebt_tiny_refinement` workload row.
EXPECTED_EBM_PYTHON_ROWS = (
    (EXPECTED_NATIVE_EBM_OPS - {"ebm_refinement"})
    | EXPECTED_PYTHON_ONLY_EBM_OPS
)

EXPECTED_WORKLOAD_OPS = {"ga_feature_pipeline", "ebt_tiny_refinement"}

REQUIRED_ROW_FIELDS = {
    "backend", "namespace", "op", "shape", "dtype", "mode", "reps",
    "latency_ms", "stdev_ms",
    "p10_ms", "p50_ms", "p90_ms", "min_ms", "max_ms",
    "max_abs_err", "tolerance", "ok",
    "device", "tessera_version",
}

REQUIRED_ENVELOPE_FIELDS = {
    "runs", "ga_primitives_count", "ebm_paths_count",
    "ebm_native_apple_gpu_count", "native_ebm_ops", "workload_count",
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


def test_ebt_tiny_workload_uses_ebm_refinement_kernel(report: dict) -> None:
    """The EBT-tiny workload must reference the native refinement kernel
    + host-side self_verify."""
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    row = next(r for r in _workload_native_rows(report)
               if r["op"] == "ebt_tiny_refinement")
    syms = row["symbols"]
    assert any("ebm_refinement_f32" in s for s in syms)
    assert any("self_verify" in s for s in syms)


# ---------------------------------------------------------------------------
# Manifest cross-check
# ---------------------------------------------------------------------------

def test_all_six_native_ebm_ops_in_fused_manifest_table() -> None:
    """``_EBM_APPLE_GPU_FUSED`` must list every op the benchmark calls
    native — keeps the manifest in lock-step with the runtime."""
    assert set(bm._EBM_APPLE_GPU_FUSED.keys()) == EXPECTED_NATIVE_EBM_OPS


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


def test_compile_time_separated_from_dispatch_time(report: dict) -> None:
    if not _apple_gpu_available(report):
        pytest.skip("apple_gpu unavailable")
    assert report["compile_time_ms"] > 0.0
    for row in report["runs"]:
        assert "compile_time_ms" not in row
