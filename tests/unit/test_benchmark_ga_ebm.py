"""CI-stable harness for ``benchmarks/apple_gpu/benchmark_ga_ebm.py``.

Drives the GA + EBM end-to-end benchmark with a CI-friendly rep count
and deterministic seeds, then validates:

  - Report envelope: GA + EBM row counts, compile_time_ms split,
    native_ebm_ops list, manifest-driven apple_gpu_status per EBM row.
  - GA stack walk: 17 fused MSL kernels, each row carries the symbol
    that `backend_manifest.clifford_manifest_for` resolves to.
  - Native EBM stack walk: `ebm_inner_step` + `ebm_refinement` ship
    fused MSL kernels (backend=apple_gpu, mode=fused), each carrying
    the symbol that `backend_manifest.ebm_manifest_for` resolves to.
  - Python-ref EBM rows: 7 entries with `backend=python_ref`,
    `mode=reference`, and `apple_gpu_status=planned` (i.e., the
    manifest agrees they don't yet have native dispatch).
  - Timing methodology: every row has p10/p50/p90/min/max columns;
    p10 ≤ p50 ≤ p90; median matches `latency_ms`.
  - Determinism: re-running with the same seeds reproduces ok-verdicts
    + correctness errors bit-for-bit.
  - JSON round-trip + CLI smoke.
  - Manifest cross-check: `ebm_manifest_for` reports `apple_gpu=fused`
    for the two native ops and `apple_gpu=planned` for the rest.

Graceful non-Darwin behavior: GA + native EBM rows are skipped (report
records `skipped_apple_gpu`); the 7 Python-ref EBM rows still run.
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


# All 17 GA primitives the benchmark walks when Apple GPU is up.
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
EXPECTED_NATIVE_EBM_OPS = {"ebm_inner_step", "ebm_refinement"}

# EBM ops with Python-reference timing only (apple_gpu_status=planned).
EXPECTED_PYTHON_EBM_OPS = {
    "ebm_energy", "ebm_langevin_step", "ebm_self_verify",
    "ebm_decode_init", "ebm_partition_exact",
    "ebm_bivector_langevin", "ebm_sphere_langevin",
}

EXPECTED_EBM_OPS = EXPECTED_NATIVE_EBM_OPS | EXPECTED_PYTHON_EBM_OPS

REQUIRED_ROW_FIELDS = {
    "backend", "namespace", "op", "shape", "dtype", "mode", "reps",
    "latency_ms", "stdev_ms",
    "p10_ms", "p50_ms", "p90_ms", "min_ms", "max_ms",
    "max_abs_err", "tolerance", "ok",
    "device", "tessera_version",
}

REQUIRED_ENVELOPE_FIELDS = {
    "runs", "ga_primitives_count", "ebm_paths_count",
    "ebm_native_apple_gpu_count", "native_ebm_ops",
    "compile_time_ms", "skipped_apple_gpu",
    "device", "tessera_version", "reps",
}


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> dict:
    """Build the GA/EBM benchmark report once for all assertions.

    Uses ``bench.DEFAULT_REPS_CI`` (=2) — small enough to stay snappy
    on every host, large enough to populate p10/p90 columns."""
    tmp_dir = tmp_path_factory.mktemp("ga_ebm_bench")
    return bench.run_report(reps=bench.DEFAULT_REPS_CI, tmp_dir=tmp_dir)


def _rows(report: dict, predicate) -> list[dict]:
    return [r for r in report["runs"] if predicate(r)]


def _ga_rows(report: dict) -> list[dict]:
    return _rows(report, lambda r: r["op"].startswith("clifford_"))


def _ebm_rows(report: dict) -> list[dict]:
    return _rows(report, lambda r: r["op"].startswith("ebm_"))


def _ebm_native_rows(report: dict) -> list[dict]:
    return _rows(report, lambda r: r["op"].startswith("ebm_")
                                 and r["backend"] == "apple_gpu")


def _ebm_python_rows(report: dict) -> list[dict]:
    return _rows(report, lambda r: r["op"].startswith("ebm_")
                                 and r["backend"] == "python_ref")


def _apple_gpu_available(report: dict) -> bool:
    return report.get("skipped_apple_gpu") is None


# ---------------------------------------------------------------------------
# Envelope checks (cross-platform)
# ---------------------------------------------------------------------------

def test_report_envelope_fields_present(report: dict) -> None:
    missing = REQUIRED_ENVELOPE_FIELDS - report.keys()
    assert not missing, f"envelope missing fields: {missing}"
    assert isinstance(report["runs"], list)
    assert isinstance(report["compile_time_ms"], (int, float))
    assert report["compile_time_ms"] >= 0.0
    assert report["reps"] == bench.DEFAULT_REPS_CI


def test_every_row_has_required_schema_fields(report: dict) -> None:
    for row in report["runs"]:
        missing = REQUIRED_ROW_FIELDS - row.keys()
        assert not missing, f"row {row['op']} missing: {missing}"


def test_timing_percentiles_consistent_for_every_row(report: dict) -> None:
    """p10 ≤ p50 ≤ p90; min ≤ max; p50 == latency_ms."""
    for row in report["runs"]:
        p10, p50, p90 = row["p10_ms"], row["p50_ms"], row["p90_ms"]
        mn, mx = row["min_ms"], row["max_ms"]
        op = row["op"]
        assert p10 <= p50 <= p90, f"{op}: p10/p50/p90 not monotone: {p10}/{p50}/{p90}"
        assert mn <= mx, f"{op}: min > max"
        assert mn <= p10, f"{op}: min > p10"
        assert mx >= p90, f"{op}: max < p90"
        assert row["latency_ms"] == p50, f"{op}: latency_ms != p50"


# ---------------------------------------------------------------------------
# EBM coverage — Python-ref rows always present; native rows gated on
# the Apple GPU runtime
# ---------------------------------------------------------------------------

def test_python_ref_ebm_rows_always_emitted(report: dict) -> None:
    ops = {r["op"] for r in _ebm_python_rows(report)}
    assert ops == EXPECTED_PYTHON_EBM_OPS


@pytest.mark.parametrize("op", sorted(EXPECTED_PYTHON_EBM_OPS))
def test_each_python_ebm_row_correct_and_marked_planned(report: dict, op: str) -> None:
    row = next(r for r in _ebm_python_rows(report) if r["op"] == op)
    assert row["ok"] is True, f"{op} failed: err={row['max_abs_err']}"
    assert row["backend"] == "python_ref"
    assert row["mode"] == "reference"
    assert row["namespace"] == "ebm"
    # The manifest column should agree these don't have native dispatch yet.
    assert row["apple_gpu_status"] == "planned", (
        f"{op}: manifest now reports apple_gpu_status="
        f"{row['apple_gpu_status']!r} — promote this op to the native set"
    )
    assert row["max_abs_err"] >= 0.0
    assert row["latency_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Apple-GPU gates — skip cleanly when the runtime can't be built.
# ---------------------------------------------------------------------------

def test_apple_gpu_rows_emitted_when_runtime_available(report: dict) -> None:
    if not _apple_gpu_available(report):
        pytest.skip(f"apple_gpu unavailable: {report.get('skipped_apple_gpu')}")
    ga_ops = {r["op"] for r in _ga_rows(report)}
    assert ga_ops == EXPECTED_GA_OPS
    assert report["ga_primitives_count"] == len(EXPECTED_GA_OPS)


def test_native_ebm_rows_emitted_when_runtime_available(report: dict) -> None:
    if not _apple_gpu_available(report):
        pytest.skip(f"apple_gpu unavailable: {report.get('skipped_apple_gpu')}")
    native_ops = {r["op"] for r in _ebm_native_rows(report)}
    assert native_ops == EXPECTED_NATIVE_EBM_OPS
    assert set(report["native_ebm_ops"]) == EXPECTED_NATIVE_EBM_OPS
    assert report["ebm_native_apple_gpu_count"] == len(EXPECTED_NATIVE_EBM_OPS)


def test_non_apple_host_records_skip_reason(report: dict) -> None:
    if _apple_gpu_available(report):
        pytest.skip("apple_gpu runtime is available on this host")
    assert isinstance(report["skipped_apple_gpu"], str)
    assert report["ga_primitives_count"] == 0
    assert report["ebm_native_apple_gpu_count"] == 0
    assert _ga_rows(report) == []
    assert _ebm_native_rows(report) == []


@pytest.mark.parametrize("op", sorted(EXPECTED_GA_OPS))
def test_each_ga_row_passes_correctness_gate(report: dict, op: str) -> None:
    if not _apple_gpu_available(report):
        pytest.skip(f"apple_gpu unavailable: {report.get('skipped_apple_gpu')}")
    row = next(r for r in _ga_rows(report) if r["op"] == op)
    assert row["ok"] is True, (
        f"{op}: GPU diverged — err={row['max_abs_err']} > "
        f"tolerance={row['tolerance']}"
    )
    assert row["backend"] == "apple_gpu"
    assert row["mode"] == "fused"
    assert row["namespace"] == "ga"
    assert row["latency_ms"] > 0.0


@pytest.mark.parametrize("op", sorted(EXPECTED_GA_OPS))
def test_each_ga_row_carries_manifest_resolved_symbol(report: dict, op: str) -> None:
    if not _apple_gpu_available(report):
        pytest.skip(f"apple_gpu unavailable: {report.get('skipped_apple_gpu')}")
    row = next(r for r in _ga_rows(report) if r["op"] == op)
    assert row["symbol"] == bench._resolve_symbol(op)


@pytest.mark.parametrize("op", sorted(EXPECTED_NATIVE_EBM_OPS))
def test_each_native_ebm_row_passes_correctness_and_manifest_gate(
        report: dict, op: str) -> None:
    if not _apple_gpu_available(report):
        pytest.skip(f"apple_gpu unavailable: {report.get('skipped_apple_gpu')}")
    row = next(r for r in _ebm_native_rows(report) if r["op"] == op)
    assert row["ok"] is True, (
        f"{op}: native GPU EBM diverged — err={row['max_abs_err']}"
    )
    assert row["backend"] == "apple_gpu"
    assert row["mode"] == "fused"
    assert row["namespace"] == "ebm"
    # The reported symbol must equal the one the manifest carries.
    spec = bm._EBM_APPLE_GPU_FUSED[op]
    assert row["symbol"] == spec["symbol"]
    assert row["apple_gpu_status"] == "fused"


# ---------------------------------------------------------------------------
# Manifest cross-check — the dispatch table itself agrees with what the
# benchmark reports.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op", sorted(EXPECTED_NATIVE_EBM_OPS))
def test_native_ebm_op_has_fused_manifest_entry(op: str) -> None:
    by_target = {e.target: e for e in bm.ebm_manifest_for(op)}
    assert by_target["apple_gpu"].status == "fused"
    assert by_target["x86"].status == "reference"
    assert by_target["apple_cpu"].status == "reference"


@pytest.mark.parametrize("op", sorted(EXPECTED_PYTHON_EBM_OPS))
def test_python_ref_ebm_op_marked_planned_in_manifest(op: str) -> None:
    by_target = {e.target: e for e in bm.ebm_manifest_for(op)}
    assert by_target["apple_gpu"].status == "planned"
    assert by_target["x86"].status == "reference"


def test_manifest_for_routes_ebm_prefix_to_ebm_table() -> None:
    """`manifest_for("ebm_*")` must hit the EBM dispatch table — proves
    the prefix routing wired in `backend_manifest.manifest_for`."""
    via_router = bm.manifest_for("ebm_inner_step")
    via_table = bm.ebm_manifest_for("ebm_inner_step")
    assert [(e.target, e.status) for e in via_router] == \
           [(e.target, e.status) for e in via_table]


# ---------------------------------------------------------------------------
# Determinism + serialization + CLI smoke
# ---------------------------------------------------------------------------

def test_determinism_same_seeds_same_correctness_outcome(tmp_path) -> None:
    r1 = bench.run_report(reps=bench.DEFAULT_REPS_CI, tmp_dir=tmp_path / "run1")
    r2 = bench.run_report(reps=bench.DEFAULT_REPS_CI, tmp_dir=tmp_path / "run2")
    by_op_1 = {r["op"]: r for r in r1["runs"]}
    by_op_2 = {r["op"]: r for r in r2["runs"]}
    assert by_op_1.keys() == by_op_2.keys()
    for op in by_op_1:
        assert by_op_1[op]["ok"] == by_op_2[op]["ok"], f"non-det ok: {op}"
        assert by_op_1[op]["max_abs_err"] == by_op_2[op]["max_abs_err"], (
            f"non-det err for {op}"
        )


def test_report_json_roundtrips(report: dict, tmp_path: Path) -> None:
    out = tmp_path / "ga_ebm_bench.json"
    out.write_text(json.dumps(report, default=float))
    reloaded = json.loads(out.read_text())
    assert reloaded["runs"] and len(reloaded["runs"]) == len(report["runs"])
    assert {r["op"] for r in reloaded["runs"]} == {r["op"] for r in report["runs"]}
    assert reloaded["compile_time_ms"] == report["compile_time_ms"]


def test_main_writes_parseable_json_file(tmp_path: Path) -> None:
    out = tmp_path / "main_report.json"
    rc = bench.main(["--ci", "--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert "runs" in payload
    # At minimum the 7 Python-ref EBM rows are always emitted.
    ops = {r["op"] for r in payload["runs"]}
    assert EXPECTED_PYTHON_EBM_OPS <= ops


def test_compile_time_separated_from_dispatch_time(report: dict) -> None:
    """`compile_time_ms` lives at the envelope level, not in rows.

    On Apple Silicon the clang++ build of the runtime dylib costs
    seconds, while per-row `latency_ms` is sub-millisecond.  Mixing
    them would make the per-row column meaningless.
    """
    if not _apple_gpu_available(report):
        pytest.skip(f"apple_gpu unavailable: {report.get('skipped_apple_gpu')}")
    assert report["compile_time_ms"] > 0.0
    for row in report["runs"]:
        # No row-local compile-time column — compile cost is amortized.
        assert "compile_time_ms" not in row
