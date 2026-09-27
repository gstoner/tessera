"""Decision #12 (amended 2026-08-30): a benchmark row carries its route.

The route is additive to the stable schema, derived from the executed
artifact's provenance rather than a label the benchmark types, and old rows
without it still load in the roofline reader -- reported as ``unknown``, never
guessed.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common.route_provenance import (  # noqa: E402
    PROVENANCE_ROW_FIELDS,
    STABLE_ROW_FIELDS,
    UNKNOWN_ROUTE,
    RouteProvenance,
    route_from_descriptor,
    route_from_runtime_artifact,
    route_unavailable,
    stable_row,
)


def _ingest():
    """The roofline tools are a standalone script tree, not a package on the
    test path; load their ingest module the way ``cli_v2.py`` does."""
    pytest.importorskip("yaml")
    base = ROOT / "tools/roofline_tools/tools/roofline"
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    spec = importlib.util.find_spec("tprof_roofline.ingest")
    assert spec is not None
    return importlib.import_module("tprof_roofline.ingest")



# ── derivation ──────────────────────────────────────────────────────────────

def test_route_is_read_from_the_runtime_artifact():
    import tessera

    @tessera.jit
    def mm(a, b):
        return tessera.ops.matmul(a, b)

    route = route_from_runtime_artifact(mm.runtime_artifact())
    assert route.route == mm.runtime_artifact().metadata["compiler_path"]
    assert route.source == "runtime_artifact.metadata.compiler_path"


def test_route_from_descriptor_prefers_the_scheduled_name():
    class D:
        provenance = {"route": "canonical_descriptor", "schedule": "sm120_scheduled_x"}

    assert route_from_descriptor(D()) == RouteProvenance(
        "sm120_scheduled_x", "descriptor.provenance.schedule")

    class R:
        provenance = {"route": "apple_value_executor"}

    assert route_from_descriptor(R()).route == "apple_value_executor"


def test_missing_provenance_is_unknown_and_says_why():
    assert route_from_runtime_artifact({}).route == UNKNOWN_ROUTE
    assert "no compiler_path" in route_from_runtime_artifact({}).source

    class Bare:
        provenance: dict = {}

    assert route_from_descriptor(Bare()).route == UNKNOWN_ROUTE
    with pytest.raises(ValueError):
        route_unavailable("")


def test_stable_row_refuses_a_typed_route_label():
    common = dict(backend="cpu", op="matmul", shape=[8, 8, 8], dtype="fp32",
                  latency_ms=1.0, tflops=0.1, memory_bw_gb_s=1.0,
                  device="host:x", tessera_version="0", timing_source="host_wall_clock")
    with pytest.raises(TypeError, match="RouteProvenance"):
        stable_row(route="tessera_jit_cpu", **common)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timing_source"):
        stable_row(route=route_unavailable("x"), **{**common, "timing_source": "vibes"})
    row = stable_row(route=route_unavailable("model"), **common)
    assert tuple(row)[:len(STABLE_ROW_FIELDS)] == STABLE_ROW_FIELDS
    assert set(PROVENANCE_ROW_FIELDS) <= set(row)


# ── writer ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def suite_payload():
    from benchmarks.run_all import run_all_benchmarks

    suite = run_all_benchmarks(
        gemm_sizes=[(16, 16, 16)], attn_configs=[(1, 1, 16, 8)],
        collective_ranks=[2], collective_sizes=[1024],
        use_compiler=True, verbose=False)
    return suite.to_dict()


def test_writer_emits_stable_rows_with_route(suite_payload):
    rows = suite_payload["rows"]
    assert rows
    for row in rows:
        assert set(STABLE_ROW_FIELDS) <= set(row)       # nothing removed
        assert set(PROVENANCE_ROW_FIELDS) <= set(row)   # route added
        assert row["route_source"]


def test_writer_route_is_derived_not_the_benchmark_label(suite_payload):
    gemm = suite_payload["gemm"][0]
    if gemm["compiler_path"] != "tessera_jit_cpu":
        pytest.skip(f"GEMM compiler lane did not execute here ({gemm['compiler_path']})")
    # The benchmark labels its lane `tessera_jit_cpu`; the artifact the JIT
    # built names its own path. The row carries the artifact's.
    assert gemm["route_source"] == "runtime_artifact.metadata.compiler_path"
    assert gemm["route"] != gemm["compiler_path"]
    assert gemm["timing_source"] == "host_wall_clock"


def test_modelled_latencies_are_unknown_route_not_a_guess(suite_payload):
    for section in ("attention", "collective"):
        for row in suite_payload[section]:
            assert row["route"] == UNKNOWN_ROUTE, row
            assert row["route_source"].startswith("unavailable:")
            assert row["timing_source"] == "analytical_model"


# ── reader ──────────────────────────────────────────────────────────────────

OLD_ROW = {"backend": "cpu", "op": "matmul", "shape": [64, 64, 64], "dtype": "fp32",
           "latency_ms": 2.0, "tflops": 0.5, "memory_bw_gb_s": 10.0,
           "device": "host:x", "tessera_version": "0.1.0"}


def test_reader_accepts_old_rows_as_unknown(tmp_path):
    ingest = _ingest()
    path = tmp_path / "old.json"
    path.write_text(json.dumps([OLD_ROW]))
    (sample,) = ingest.read_benchmark_json(str(path))
    assert sample.time_ms == 2.0
    assert sample.flop_count == pytest.approx(0.5e12 * 2e-3)
    assert sample.dram_bytes == pytest.approx(10e9 * 2e-3)
    for field_name in PROVENANCE_ROW_FIELDS:
        assert sample.meta[field_name] == "unknown"
    assert sample.meta["backend"] == "cpu"     # not promoted into a route


def test_reader_accepts_new_rows_and_keeps_their_route(tmp_path, suite_payload):
    ingest = _ingest()
    path = tmp_path / "new.json"
    path.write_text(json.dumps(suite_payload))
    samples = ingest.read_benchmark_json(str(path))
    assert len(samples) == len(suite_payload["rows"])
    by_route = {s.meta["route"] for s in samples}
    assert UNKNOWN_ROUTE in by_route
    for sample, row in zip(samples, suite_payload["rows"]):
        assert sample.meta["route"] == row["route"]
        assert sample.meta["timing_source"] == row["timing_source"]


def test_reader_runs_through_the_cli(tmp_path, suite_payload):
    pytest.importorskip("matplotlib")
    import subprocess

    data = tmp_path / "bench.json"
    data.write_text(json.dumps({"rows": [OLD_ROW]}))
    peaks = next((ROOT / "tools/roofline_tools/tools/roofline/peaks").glob("*.yaml"))
    subprocess.check_call([
        sys.executable, str(ROOT / "tools/roofline_tools/tools/roofline/cli_v2.py"),
        "one", "--peaks", str(peaks), "--input", str(data), "--fmt", "benchmark",
        "--outdir", str(tmp_path / "out")])
    assert (tmp_path / "out" / "roofline_report.html").exists()
