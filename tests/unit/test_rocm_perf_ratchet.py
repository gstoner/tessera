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
    # The recorder skip-cleans on these guards; each must exist and answer a
    # bool WITHOUT raising on a GPU-less host (False here, True on the box).
    # Two independent lanes: the shipped WMMA GEMM symbol, and the compiled
    # FA-2 flash_attn forward (the second hot path gates on the latter).
    from tessera import runtime as rt

    assert isinstance(rt._rocm_wmma_runtime_available(), bool)
    assert isinstance(rt._rocm_compiled_flash_attn_available(), bool)


def test_baseline_well_formed_when_present():
    if not BASELINE.is_file():
        pytest.skip("rocm baseline not recorded yet (needs the gfx1151 box)")
    doc = json.loads(BASELINE.read_text())
    assert doc["schema"] == "tessera.benchmark.ratchet.v1"
    rows = doc["rows"]
    ops = {(r["op"], r["shape"]) for r in rows}
    # matmul WMMA ladder is always recorded.
    for (m, n, k) in recorder.HOT_PATH_SIZES:
        assert ("matmul", f"{m}x{n}x{k}") in ops, f"baseline missing {m}x{n}x{k}"
    # flash_attn is the second hot path — recorded only on an FA-capable box, so
    # its rows are optional; but IF any are present they must cover the full
    # recorded ladder (a partial flash row set means a broken recording).
    if any(r["mode"] == "flash_attn" for r in rows):
        for (b, h, s, d) in recorder.FLASH_ATTN_SHAPES:
            assert ("flash_attn", f"{b}x{h}x{s}x{d}") in ops, \
                f"baseline missing flash_attn {b}x{h}x{s}x{d}"
    # flash_attn backward — same gate (recorded only when the compiled FA lane is
    # live); the full recorded ladder must be present if any bwd row is.
    if any(r["mode"] == "flash_attn_bwd" for r in rows):
        for (b, h, s, d) in recorder.FLASH_ATTN_BWD_SHAPES:
            assert ("flash_attn_bwd", f"{b}x{h}x{s}x{d}") in ops, \
                f"baseline missing flash_attn_bwd {b}x{h}x{s}x{d}"
    # register-blocked f32 GEMM — same tessera-opt gate; full ladder if present.
    if any(r["mode"] == "gemm_f32" for r in rows):
        for (m, n, k) in recorder.GEMM_F32_SIZES:
            assert ("gemm_f32", f"{m}x{n}x{k}") in ops, \
                f"baseline missing gemm_f32 {m}x{n}x{k}"
    # memory-bound movement lanes (bandwidth-modeled, recorded on a GPU box).
    if any(r["mode"] == "kv_cache_append" for r in rows):
        for n in recorder.KV_APPEND_ROWS:
            assert ("kv_cache_append", f"{n}x{recorder.KV_ROW_WIDTH}") in ops, \
                f"baseline missing kv_cache_append {n}"
    if any(r["mode"] == "moe_dispatch" for r in rows):
        # moe shapes are plan-derived (packed slot count), so just assert each
        # recorded dispatch row has a paired combine row at the same shape.
        disp = {r["shape"] for r in rows if r["mode"] == "moe_dispatch"}
        comb = {r["shape"] for r in rows if r["mode"] == "moe_combine"}
        assert disp == comb, "moe dispatch/combine shapes must pair"
    _MOVEMENT = {"kv_cache_append", "kv_cache_read", "moe_dispatch", "moe_combine"}
    for r in rows:
        assert r["mode"] in {"wmma", "flash_attn", "flash_attn_bwd",
                             "gemm_f32"} | _MOVEMENT
        assert r["max_latency_ms"] > r["median_ms"] > 0
        # Movement rows carry bandwidth attainment; compute rows carry FLOP
        # attainment — never both, and each carries the matching floor.
        if r["mode"] in _MOVEMENT:
            assert "pct_peak_bw" in r and "bw_attainment_floor" in r
            assert "pct_peak" not in r


def test_hot_path_manifest_rows_carry_benchmark_json():
    # E2 layer-1 linkage (Apple-style): the gfx1151 hot-path manifest rows point
    # at the recorded ratchet baseline, and that file exists on disk. Runs
    # GPU-free (manifest synthesis needs no device).
    from tessera.compiler import backend_manifest as bm

    rel = str(BASELINE.relative_to(REPO_ROOT))
    for op in ("matmul", "flash_attn"):
        rocm = [e for e in bm.manifest_for(op) if e.target == "rocm"]
        assert rocm, f"no rocm manifest row for {op}"
        linked = [e for e in rocm if e.benchmark_json == rel]
        assert linked, f"{op} rocm row missing baseline pointer {rel}"
        for e in linked:
            assert (REPO_ROOT / e.benchmark_json).is_file()


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
#
# Gated PER LANE, not by one blanket probe: the two ROCm hot paths have
# independent availability (the shipped WMMA GEMM symbol vs the compiled FA-2
# lane that shells to tessera-opt). A host with the GEMM lane but no compiled
# flash lane records matmul only — so the flash live check must skip there, not
# fail on the baseline's flash rows it legitimately can't re-time.

def _rocm_wmma_live():
    try:
        from tessera import runtime as rt
        return rt._rocm_wmma_runtime_available()
    except Exception:
        return False


def _rocm_flash_live():
    try:
        from tessera import runtime as rt
        return rt._rocm_compiled_flash_attn_available()
    except Exception:
        return False


def _rocm_gemm_f32_live():
    # Own probe: the f32 GEMM pass is separate from the flash lane, so a broken
    # flash lane must not skip the gemm_f32 re-timing.
    try:
        from tessera import runtime as rt
        return rt._rocm_compiled_gemm_f32_available()
    except Exception:
        return False


def _live_ratchet_failures(rt, modes):
    """Re-time only the hot paths in ``modes`` and gate them against the matching
    baseline rows. Filtering the baseline to ``modes`` keeps an unavailable lane
    from registering as a coverage hole (the recorder skip-cleans it)."""
    baseline = json.loads(BASELINE.read_text())
    rows = []
    for op, shape, dtype, mode, thunk in recorder.hot_path_cases(rt):
        if mode not in modes:
            continue
        med = recorder._median_ms(thunk, reps=10)
        rows.append({"op": op, "shape": shape, "dtype": dtype,
                     "mode": mode, "latency_ms": med})
    gated = {"schema": baseline["schema"],
             "rows": [r for r in baseline["rows"] if r["mode"] in modes]}
    return perf_gate.evaluate_ratchet(rows, gated)


@pytest.mark.slow
@pytest.mark.skipif(not _rocm_wmma_live(),
                    reason="live AMD GPU (gfx1151) WMMA lane required")
def test_live_matmul_within_ratchet():
    if not BASELINE.is_file():
        pytest.skip("rocm baseline not recorded yet — run the recorder first")
    from tessera import runtime as rt

    failures = _live_ratchet_failures(rt, {"wmma"})
    assert not failures, "\n".join(failures)


@pytest.mark.slow
@pytest.mark.skipif(not _rocm_flash_live(),
                    reason="live compiled ROCm flash-attn lane required")
def test_live_flash_attn_within_ratchet():
    if not BASELINE.is_file():
        pytest.skip("rocm baseline not recorded yet — run the recorder first")
    from tessera import runtime as rt

    baseline = json.loads(BASELINE.read_text())
    if not any(r["mode"] == "flash_attn" for r in baseline["rows"]):
        pytest.skip("baseline has no flash_attn rows (recorded pre-flash lane)")
    failures = _live_ratchet_failures(rt, {"flash_attn"})
    assert not failures, "\n".join(failures)


@pytest.mark.slow
@pytest.mark.skipif(not _rocm_flash_live(),
                    reason="live compiled ROCm flash-attn lane required")
def test_live_flash_attn_bwd_within_ratchet():
    # The backward lane (rocm_flash_attn_bwd_compiled) rides the same compiled
    # FA pass as the forward, so it shares the flash gate. Re-time it live so a
    # backward perf regression actually fails against the committed baseline.
    if not BASELINE.is_file():
        pytest.skip("rocm baseline not recorded yet — run the recorder first")
    from tessera import runtime as rt

    baseline = json.loads(BASELINE.read_text())
    if not any(r["mode"] == "flash_attn_bwd" for r in baseline["rows"]):
        pytest.skip("baseline has no flash_attn_bwd rows (recorded pre-bwd lane)")
    failures = _live_ratchet_failures(rt, {"flash_attn_bwd"})
    assert not failures, "\n".join(failures)


@pytest.mark.slow
@pytest.mark.skipif(not _rocm_gemm_f32_live(),
                    reason="live compiled ROCm f32 GEMM lane required")
def test_live_gemm_f32_within_ratchet():
    # Gated on the f32 GEMM's OWN probe (not the flash lane): re-time the blocked
    # kernel live so a regression (e.g. a lost register-blocking win) fails
    # against the committed baseline even if the flash lane is unavailable.
    if not BASELINE.is_file():
        pytest.skip("rocm baseline not recorded yet — run the recorder first")
    from tessera import runtime as rt

    baseline = json.loads(BASELINE.read_text())
    if not any(r["mode"] == "gemm_f32" for r in baseline["rows"]):
        pytest.skip("baseline has no gemm_f32 rows (recorded pre-f32-gemm lane)")
    failures = _live_ratchet_failures(rt, {"gemm_f32"})
    assert not failures, "\n".join(failures)


def _rocm_movement_live():
    # The kv/moe movement lanes build + launch a gather/scatter hsaco — same gate
    # the moe-transport test uses (tessera-opt + a usable GPU).
    try:
        from tessera import runtime as rt
        return (rt._tessera_opt_path() is not None
                and rt._rocm_wmma_runtime_available())
    except Exception:
        return False


_MOVEMENT_MODES = {"kv_cache_append", "kv_cache_read",
                   "moe_dispatch", "moe_combine"}


@pytest.mark.slow
@pytest.mark.skipif(not _rocm_movement_live(),
                    reason="live ROCm gather/scatter movement lanes required")
def test_live_movement_lanes_within_ratchet():
    # The bandwidth-bound kv/moe gather/scatter lanes: re-time them live so a
    # movement regression fails against the committed baseline (latency ratchet;
    # the absolute bandwidth floor is gated separately by roofline.evaluate_
    # attainment in test_roofline_attainment).
    if not BASELINE.is_file():
        pytest.skip("rocm baseline not recorded yet — run the recorder first")
    from tessera import runtime as rt

    baseline = json.loads(BASELINE.read_text())
    present = {r["mode"] for r in baseline["rows"]} & _MOVEMENT_MODES
    if not present:
        pytest.skip("baseline has no movement rows (recorded pre-movement lanes)")
    failures = _live_ratchet_failures(rt, present)
    assert not failures, "\n".join(failures)


@pytest.mark.slow
@pytest.mark.skipif(not _rocm_movement_live(),
                    reason="live ROCm gather/scatter movement lanes required")
def test_live_movement_lanes_within_bandwidth_attainment():
    # The ABSOLUTE bar for the movement lanes: re-time and gate achieved GB/s ÷
    # peak BW against each row's bw_attainment_floor, so a bandwidth regression
    # (not just a latency one) fails against the committed baseline.
    if not BASELINE.is_file():
        pytest.skip("rocm baseline not recorded yet — run the recorder first")
    import importlib.util as _ilu

    from tessera import runtime as rt
    spec = _ilu.spec_from_file_location("roofline", REPO_ROOT / "benchmarks"
                                        / "roofline.py")
    roofline = _ilu.module_from_spec(spec)
    spec.loader.exec_module(roofline)

    baseline = json.loads(BASELINE.read_text())
    present = {r["mode"] for r in baseline["rows"]} & _MOVEMENT_MODES
    if not present:
        pytest.skip("baseline has no movement rows (recorded pre-movement lanes)")
    rows = []
    for op, shape, dtype, mode, thunk in recorder.hot_path_cases(rt):
        if mode not in present:
            continue
        rows.append({"op": op, "shape": shape, "dtype": dtype, "mode": mode,
                     "latency_ms": recorder._median_ms(thunk, reps=10)})
    # Gate ONLY the movement floors — filtering the baseline keeps the
    # un-measured compute floors from registering as coverage holes.
    gated = {"schema": baseline["schema"],
             "rows": [r for r in baseline["rows"] if r["mode"] in present]}
    failures = roofline.evaluate_attainment(rows, gated, "rocm:gfx1151")
    assert not failures, "\n".join(failures)
