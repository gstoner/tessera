"""E2 (2026-07-06) — ROCm gfx1151 perf ratchet (real-hardware perf ratchet).

Mirrors the Apple GPU ratchet (`test_apple_gpu_perf_ratchet.py`) three layers,
adapted to the hardware-gated ROCm lane:

1. Recorder + evaluator wiring — the recorder module exposes the shared
   `hot_path_cases` / `_median_ms` / `OUT` surface the live layer re-times,
   and the runtime availability guard the recorder skip-cleans on exists.
2. Ratchet evaluator — `perf_gate.evaluate_ratchet` semantics on the ROCm
   `wmma` mode (pass / regression / coverage hole / bad schema).
3. Live ratchet (slow, needs a live AMD GPU) — re-time the hot paths through
   the production WMMA symbol and gate vs the recorded baseline.

Honesty note: the committed baseline JSON is recorded ON the gfx1151 box (no
GPU here → no fabricated numbers, repo Decision #26). Every assertion that
reads the baseline is guarded on the file existing, so layers 1–2 stay green
in GPU-less CI and turn into real checks the moment the baseline lands.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / "benchmarks" / "baselines" / "rocm_gfx1151_hot_paths.json"
RECORDER = REPO_ROOT / "benchmarks" / "rocm" / "record_hot_path_baseline.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


perf_gate = _load(REPO_ROOT / "benchmarks" / "perf_gate.py", "perf_gate")
recorder = _load(RECORDER, "rocm_record_hot_path_baseline")


# ── 1. Recorder + evaluator wiring (GPU-free) ─────────────────────────

def test_recorder_exposes_shared_surface():
    # The live layer re-times through these; keep them stable.
    assert callable(recorder.hot_path_cases)
    assert callable(recorder._median_ms)
    assert recorder.OUT == BASELINE
    assert recorder.HOT_PATH_SIZES, "recorder must name at least one hot path"


def test_runtime_availability_guard_is_honest():
    # The recorder skip-cleans on this guard; it must exist and answer a bool
    # WITHOUT raising on a GPU-less host (returns False here, True on the box).
    from tessera import runtime as rt

    assert isinstance(rt._rocm_wmma_runtime_available(), bool)


def test_baseline_well_formed_when_present():
    if not BASELINE.is_file():
        pytest.skip("rocm baseline not recorded yet (needs the gfx1151 box)")
    doc = json.loads(BASELINE.read_text())
    assert doc["schema"] == "tessera.benchmark.ratchet.v1"
    rows = doc["rows"]
    ops = {(r["op"], r["shape"]) for r in rows}
    for (m, n, k) in recorder.HOT_PATH_SIZES:
        assert ("matmul", f"{m}x{n}x{k}") in ops, f"baseline missing {m}x{n}x{k}"
    for r in rows:
        assert r["mode"] == "wmma"
        assert r["max_latency_ms"] > r["median_ms"] > 0


# ── 2. Evaluator semantics (GPU-free) ─────────────────────────────────

_BASE = {"schema": "tessera.benchmark.ratchet.v1",
         "rows": [{"op": "matmul", "shape": "512x512x512", "dtype": "f16",
                   "mode": "wmma", "median_ms": 1.0, "max_latency_ms": 2.0}]}


def _row(latency):
    return {"op": "matmul", "shape": "512x512x512", "dtype": "f16",
            "mode": "wmma", "latency_ms": latency}


def test_ratchet_passes_under_cap():
    assert perf_gate.evaluate_ratchet([_row(1.5)], _BASE) == []


def test_ratchet_fails_regression():
    fails = perf_gate.evaluate_ratchet([_row(2.5)], _BASE)
    assert len(fails) == 1 and "exceeds ratchet cap" in fails[0]


def test_ratchet_fails_missing_coverage():
    fails = perf_gate.evaluate_ratchet([], _BASE)
    assert len(fails) == 1 and "no measurement" in fails[0]


def test_ratchet_rejects_unknown_schema():
    assert perf_gate.evaluate_ratchet([], {"schema": "bogus"}) == [
        "unsupported ratchet baseline schema 'bogus'"]


# ── 3. Live ratchet (needs a live AMD GPU; slow) ──────────────────────

def _rocm_live():
    try:
        from tessera import runtime as rt
        return rt._rocm_wmma_runtime_available()
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _rocm_live(), reason="live AMD GPU (gfx1151) required")
def test_live_hot_paths_within_ratchet():
    if not BASELINE.is_file():
        pytest.skip("rocm baseline not recorded yet — run the recorder first")
    from tessera import runtime as rt

    baseline = json.loads(BASELINE.read_text())
    rows = []
    for op, shape, dtype, mode, thunk in recorder.hot_path_cases(rt):
        med = recorder._median_ms(thunk, reps=10)
        rows.append({"op": op, "shape": shape, "dtype": dtype,
                     "mode": mode, "latency_ms": med})
    failures = perf_gate.evaluate_ratchet(rows, baseline)
    assert not failures, "\n".join(failures)
