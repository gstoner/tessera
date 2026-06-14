"""P2 (2026-06-09) — Apple GPU perf ratchet (audit Next-Work #6).

Three layers:
1. Manifest linkage — every named hot path (matmul, fused epilogues,
   conv2d, decode chain via bmm, packaged kernels) carries
   ``benchmark_json`` pointing at evidence that exists on disk.
2. Ratchet evaluator — ``perf_gate.evaluate_ratchet`` synthetic-row
   semantics (pass / regression / coverage hole).
3. Live ratchet (slow, Darwin+Metal) — re-time the hot paths through the
   production dispatchers and gate vs the recorded baseline.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / "benchmarks" / "baselines" / "apple_gpu_hot_paths.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


perf_gate = _load(REPO_ROOT / "benchmarks" / "perf_gate.py", "perf_gate")


# ── 1. Manifest linkage ───────────────────────────────────────────────

HOT_PATH_OPS = ("matmul", "softmax", "rmsnorm", "flash_attn", "bmm", "conv2d")


@pytest.mark.parametrize("op", HOT_PATH_OPS)
def test_hot_path_manifest_rows_carry_benchmark_json(op):
    from tessera.compiler import backend_manifest as bm

    entry = next(e for e in bm.manifest_for(op) if e.target == "apple_gpu")
    assert entry.benchmark_json, f"{op}: hot-path row missing benchmark_json"
    assert (REPO_ROOT / entry.benchmark_json).is_file(), entry.benchmark_json


def test_packaged_rows_carry_benchmark_json():
    from tessera.compiler.apple_packaged_manifest import PACKAGED_PRODUCTION_KERNELS

    for e in PACKAGED_PRODUCTION_KERNELS:
        assert e.benchmark_json, f"packaged row missing benchmark_json: {e.notes[:40]}"
        assert (REPO_ROOT / e.benchmark_json).is_file()


def test_baseline_covers_named_hot_paths():
    rows = json.loads(BASELINE.read_text())["rows"]
    ops = {r["op"] for r in rows}
    assert {"matmul", "matmul_softmax", "matmul_gelu", "matmul_rmsnorm",
            "matmul_softmax_matmul", "swiglu", "conv2d",
            "grouped_gemm", "moe_swiglu_block"} <= ops
    for r in rows:
        assert r["max_latency_ms"] > r["median_ms"] > 0


# ── 1b. Even benchmark-metadata attachment (P1, 2026-06-10) ───────────

def _apple_gpu_entry(op):
    from tessera.compiler import backend_manifest as bm
    return next(e for e in bm.manifest_for(op) if e.target == "apple_gpu")


@pytest.mark.parametrize("op", HOT_PATH_OPS + ("grouped_gemm", "moe_swiglu_block"))
def test_hot_path_rows_carry_structured_benchmark_metadata(op):
    # P1: every hot-path / MoE row carries a structured BenchmarkMetadata, not
    # just the loose benchmark_json pointer.
    from tessera.compiler.backend_manifest import (
        BenchmarkMetadata, BENCHMARK_HOT_PATH_GROUPS,
    )

    md = _apple_gpu_entry(op).benchmark_metadata
    assert isinstance(md, BenchmarkMetadata), f"{op}: missing benchmark_metadata"
    assert md.hot_path_group in BENCHMARK_HOT_PATH_GROUPS
    assert (REPO_ROOT / md.harness).is_file(), f"{op}: harness {md.harness} absent"


def test_benchmark_json_rows_are_evenly_metadata_attached():
    # The "even attachment" invariant: any Apple GPU row that carries a
    # benchmark_json must ALSO carry structured benchmark_metadata.
    from tessera.compiler import backend_manifest as bm
    from tessera.compiler.backend_manifest import BenchmarkMetadata

    offenders = []
    for op, entries in bm.all_manifests().items():
        for e in entries:
            if e.target == "apple_gpu" and e.benchmark_json and not isinstance(
                e.benchmark_metadata, BenchmarkMetadata
            ):
                offenders.append(op)
    assert not offenders, f"benchmark_json rows without benchmark_metadata: {offenders}"


def test_ratcheted_metadata_keys_exist_in_baseline():
    # honesty gate: a row marked ratcheted MUST name a key that is a real row
    # in the baseline (so "ratcheted=True" can never be aspirational).
    from tessera.compiler import backend_manifest as bm

    baseline_ops = {r["op"] for r in json.loads(BASELINE.read_text())["rows"]}
    for op, entries in bm.all_manifests().items():
        for e in entries:
            md = e.benchmark_metadata
            if md is not None and getattr(md, "ratcheted", False):
                assert md.ratchet_key in baseline_ops, (
                    f"{op}: ratcheted but ratchet_key {md.ratchet_key!r} "
                    "is not a baseline row")


# ── 2. Evaluator semantics ────────────────────────────────────────────

_BASE = {"schema": "tessera.benchmark.ratchet.v1",
         "rows": [{"op": "matmul", "shape": "8x8x8", "dtype": "f32",
                   "mode": "fused", "median_ms": 1.0, "max_latency_ms": 2.0}]}


def _row(latency):
    return {"op": "matmul", "shape": "8x8x8", "dtype": "f32",
            "mode": "fused", "latency_ms": latency}


def test_ratchet_passes_under_cap():
    assert perf_gate.evaluate_ratchet([_row(1.5)], _BASE) == []


def test_ratchet_fails_regression():
    fails = perf_gate.evaluate_ratchet([_row(2.5)], _BASE)
    assert len(fails) == 1 and "exceeds ratchet cap" in fails[0]


def test_ratchet_fails_missing_coverage():
    fails = perf_gate.evaluate_ratchet([], _BASE)
    assert len(fails) == 1 and "no measurement" in fails[0]


def test_ratchet_rejects_unknown_schema():
    assert perf_gate.evaluate_ratchet([], {"schema": "bogus"}) == ["unsupported ratchet baseline schema 'bogus'"]


# ── 3. Live ratchet (Darwin + Metal; slow) ───────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="Metal required")
def test_live_hot_paths_within_ratchet():
    from tessera import runtime as rt

    recorder = _load(
        REPO_ROOT / "benchmarks" / "apple_gpu" / "record_hot_path_baseline.py",
        "record_hot_path_baseline")
    baseline = json.loads(BASELINE.read_text())
    rows = []
    for op, shape, dtype, thunk in recorder.hot_path_cases(rt):
        med = recorder._median_ms(thunk, reps=10)
        rows.append({"op": op, "shape": shape, "dtype": dtype,
                     "mode": recorder.MODE_BY_OP.get(op, "fused"),
                     "latency_ms": med})
    failures = perf_gate.evaluate_ratchet(rows, baseline)
    assert not failures, "\n".join(failures)
