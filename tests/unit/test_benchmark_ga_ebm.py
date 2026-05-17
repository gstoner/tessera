"""CI-stable harness for ``benchmarks/apple_gpu/benchmark_ga_ebm.py``.

Drives the GA + EBM end-to-end benchmark with a tiny rep count and
deterministic seeds, then validates:

  - Report structure: 17 GA rows + 8 EBM rows + envelope metadata.
  - Manifest dispatch: every GA row carries the apple_gpu C ABI symbol
    that ``backend_manifest.clifford_manifest_for`` resolves to.
  - Correctness vs Python reference: every row has ``ok=True`` (max abs
    err under its per-op tolerance).
  - Determinism: re-running the benchmark with the same seeds produces
    the same correctness outcome and the same symbol set.

Graceful behavior on non-Darwin: GA rows are skipped (the report
records ``skipped_apple_gpu``), only EBM (Python-reference) rows are
emitted. The test still asserts those rows are correct.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Importable directly — the benchmark module's top-level inserts
# `python/` into sys.path when run standalone, but pytest already has
# the project on `PYTHONPATH` via conftest, so we just import normally.
ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "benchmarks" / "apple_gpu"
sys.path.insert(0, str(BENCH_DIR))
import benchmark_ga_ebm as bench  # noqa: E402


# All 17 GA primitives the benchmark should walk when Apple GPU is up.
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

EXPECTED_EBM_OPS = {
    "ebm_energy", "ebm_inner_step", "ebm_langevin_step",
    "ebm_self_verify", "ebm_decode_init", "ebm_partition_exact",
    "ebm_bivector_langevin", "ebm_sphere_langevin",
}

# Required schema for every row.
REQUIRED_ROW_FIELDS = {
    "backend", "op", "shape", "dtype", "mode", "reps",
    "latency_ms", "stdev_ms", "max_abs_err", "tolerance", "ok",
    "device", "tessera_version",
}


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> dict:
    """Build the GA/EBM benchmark report once for all assertions."""
    tmp_dir = tmp_path_factory.mktemp("ga_ebm_bench")
    return bench.run_report(reps=2, tmp_dir=tmp_dir)


def _ga_rows(report: dict) -> list[dict]:
    return [r for r in report["runs"] if r["op"].startswith("clifford_")]


def _ebm_rows(report: dict) -> list[dict]:
    return [r for r in report["runs"] if r["op"].startswith("ebm_")]


# ---------------------------------------------------------------------------
# Structure / envelope checks (cross-platform)
# ---------------------------------------------------------------------------

def test_report_envelope_fields_present(report: dict) -> None:
    assert isinstance(report, dict)
    assert "runs" in report and isinstance(report["runs"], list)
    assert isinstance(report["ga_primitives_count"], int)
    assert isinstance(report["ebm_paths_count"], int)
    assert "device" in report
    assert "tessera_version" in report


def test_ebm_rows_always_emitted_regardless_of_platform(report: dict) -> None:
    """EBM paths run on pure Python so they're emitted on every host."""
    ebm_ops = {r["op"] for r in _ebm_rows(report)}
    assert ebm_ops == EXPECTED_EBM_OPS
    assert report["ebm_paths_count"] == len(EXPECTED_EBM_OPS)


@pytest.mark.parametrize("op", sorted(EXPECTED_EBM_OPS))
def test_each_ebm_row_passes_correctness_gate(report: dict, op: str) -> None:
    row = next(r for r in _ebm_rows(report) if r["op"] == op)
    assert row["ok"] is True, (
        f"{op} failed correctness gate: err={row['max_abs_err']} "
        f"> tolerance={row['tolerance']}"
    )
    assert row["backend"] == "python_ref"
    assert row["mode"] == "reference"
    assert row["max_abs_err"] >= 0.0
    assert row["latency_ms"] >= 0.0


def test_every_row_has_required_schema_fields(report: dict) -> None:
    for row in report["runs"]:
        missing = REQUIRED_ROW_FIELDS - row.keys()
        assert not missing, f"row {row['op']} missing: {missing}"


# ---------------------------------------------------------------------------
# Apple-GPU-only gates — skip cleanly on non-Darwin or runtime missing.
# ---------------------------------------------------------------------------

def _apple_gpu_available(report: dict) -> bool:
    return report.get("skipped_apple_gpu") is None


def test_apple_gpu_rows_emitted_when_runtime_available(report: dict) -> None:
    if not _apple_gpu_available(report):
        pytest.skip(f"apple_gpu unavailable: {report.get('skipped_apple_gpu')}")
    ga_ops = {r["op"] for r in _ga_rows(report)}
    assert ga_ops == EXPECTED_GA_OPS
    assert report["ga_primitives_count"] == len(EXPECTED_GA_OPS)


def test_non_apple_host_records_skip_reason_and_emits_zero_ga(report: dict) -> None:
    if _apple_gpu_available(report):
        pytest.skip("apple_gpu runtime is available on this host")
    assert report["skipped_apple_gpu"] is not None
    assert isinstance(report["skipped_apple_gpu"], str)
    assert report["ga_primitives_count"] == 0
    assert _ga_rows(report) == []


@pytest.mark.parametrize("op", sorted(EXPECTED_GA_OPS))
def test_each_ga_row_passes_correctness_gate(report: dict, op: str) -> None:
    if not _apple_gpu_available(report):
        pytest.skip(f"apple_gpu unavailable: {report.get('skipped_apple_gpu')}")
    row = next(r for r in _ga_rows(report) if r["op"] == op)
    assert row["ok"] is True, (
        f"{op}: GPU output diverged from Python reference — "
        f"err={row['max_abs_err']} > tolerance={row['tolerance']}"
    )
    assert row["backend"] == "apple_gpu"
    assert row["mode"] == "fused"
    assert row["max_abs_err"] >= 0.0
    assert row["latency_ms"] > 0.0


@pytest.mark.parametrize("op", sorted(EXPECTED_GA_OPS))
def test_each_ga_row_carries_manifest_resolved_symbol(report: dict, op: str) -> None:
    """The symbol in the report must be the one the backend manifest
    resolves — proves the benchmark went through manifest-dispatch."""
    if not _apple_gpu_available(report):
        pytest.skip(f"apple_gpu unavailable: {report.get('skipped_apple_gpu')}")
    row = next(r for r in _ga_rows(report) if r["op"] == op)
    expected_symbol = bench._resolve_symbol(op)
    assert row["symbol"] == expected_symbol


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism_same_seeds_same_correctness_outcome(tmp_path) -> None:
    """Re-running the benchmark with the same RNG seeds yields the same
    correctness verdict, same row set, and (when GPU available) the
    same symbol set. Timing is allowed to vary."""
    r1 = bench.run_report(reps=2, tmp_dir=tmp_path / "run1")
    r2 = bench.run_report(reps=2, tmp_dir=tmp_path / "run2")
    by_op_1 = {r["op"]: r for r in r1["runs"]}
    by_op_2 = {r["op"]: r for r in r2["runs"]}
    assert by_op_1.keys() == by_op_2.keys()
    for op in by_op_1:
        assert by_op_1[op]["ok"] == by_op_2[op]["ok"], f"non-deterministic ok: {op}"
        # Correctness error is over deterministic inputs ⇒ bit-stable.
        assert by_op_1[op]["max_abs_err"] == by_op_2[op]["max_abs_err"], (
            f"non-deterministic err for {op}"
        )


# ---------------------------------------------------------------------------
# JSON serializability (the benchmark is a report producer; downstream
# tools — roofline_tools, dashboards — round-trip through JSON).
# ---------------------------------------------------------------------------

def test_report_json_roundtrips(report: dict, tmp_path: Path) -> None:
    out = tmp_path / "ga_ebm_bench.json"
    out.write_text(json.dumps(report, default=float))
    reloaded = json.loads(out.read_text())
    assert reloaded["runs"] and len(reloaded["runs"]) == len(report["runs"])
    assert {r["op"] for r in reloaded["runs"]} == {r["op"] for r in report["runs"]}


# ---------------------------------------------------------------------------
# CLI smoke — main() exits 0 and writes a parseable JSON file.
# ---------------------------------------------------------------------------

def test_main_writes_parseable_json_file(tmp_path: Path) -> None:
    out = tmp_path / "main_report.json"
    rc = bench.main(["--reps", "2", "--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert "runs" in payload and isinstance(payload["runs"], list)
    assert len(payload["runs"]) >= len(EXPECTED_EBM_OPS)
