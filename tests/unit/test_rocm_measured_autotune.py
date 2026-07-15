"""Workstream D2 — ROCm measured-autotune wiring + the fleet-shared corpus.

Three slices, mirroring ``test_rocm_plugin.py``:

1. **Device tag dispatch (host-free)** — ``autotune._device_id`` keys the cache by
   the live gfx arch (``rocm:gfx1151``) so a verdict measured on one device is
   never reused on another; off-GPU it falls back to the bare target id.
2. **Corpus round-trip (host-free)** — the measured verdicts serialize to a
   JSON-safe, self-describing corpus and warm-start a fresh cache, keyed by device
   (a gfx1151 record is inert on a CDNA/NVIDIA box). Warm-start is once, and a
   measure-on-this-box verdict beats a stale corpus row.
3. **Live gate (needs a live gfx1151)** — ``measured_arbitrate`` over the ROCm
   fused-region candidates times each F4-passing lane, caches the fastest per
   shape-bucket, and re-queries hit the cache without re-timing. Proves the
   measured loop overrides tier-priority (the Tier-1 generic lane wins the tiny
   bucket over the Tier-3 WMMA crown jewel).
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import tessera.compiler.fusion as F
import tessera.compiler.emit.rocm_hip as rocm  # noqa: F401 — self-registers
from tessera.compiler.emit import autotune as AT
from tessera.compiler.emit.candidate import OP_FUSED_REGION, OP_MATMUL


def _rocm_hip_live() -> bool:
    if not (shutil.which("hipcc") or os.path.exists("/opt/rocm/bin/hipcc")):
        return False
    try:
        from tessera import runtime as rt
        return rt._rocm_wmma_runtime_available()
    except Exception:
        return False


# ── 1. Device-tag dispatch (host-free) ────────────────────────────────────────

def test_device_id_uses_rocm_probe(monkeypatch):
    from tessera import runtime as rt
    monkeypatch.setattr(rt, "_rocm_device_name", lambda: "gfx1151")
    assert AT._device_id("rocm") == "rocm:gfx1151"


def test_device_id_falls_back_off_gpu(monkeypatch):
    # No device → probe returns None → bare target id (never keyed as a device).
    from tessera import runtime as rt
    monkeypatch.setattr(rt, "_rocm_device_name", lambda: None)
    assert AT._device_id("rocm") == "rocm"


def test_device_id_probe_exception_is_swallowed(monkeypatch):
    from tessera import runtime as rt

    def _boom():
        raise RuntimeError("no hip")
    monkeypatch.setattr(rt, "_rocm_device_name", _boom)
    assert AT._device_id("rocm") == "rocm"


def test_rocm_device_name_resolution(monkeypatch):
    # The tag identifies the ACTUAL device the measurement ran on — never a compile
    # default — so the corpus's per-device isolation holds on a mixed ROCm fleet.
    from tessera import runtime as rt

    # (1) no executable ROCm device here → None (no gfx key on a GPU-less host).
    rt._rocm_device_name_probe = False
    monkeypatch.setattr(rt, "_rocm_wmma_runtime_available", lambda: False)
    assert rt._rocm_device_name() is None

    # (2) available + an explicit TESSERA_ROCM_CHIP pin is authoritative (it IS the
    #     offload-arch), overriding the live probe.
    rt._rocm_device_name_probe = False
    monkeypatch.setattr(rt, "_rocm_wmma_runtime_available", lambda: True)
    monkeypatch.setenv("TESSERA_ROCM_CHIP", "gfx942")
    monkeypatch.setattr(rt, "_rocm_live_arch", lambda: "gfx1151")   # ignored
    assert rt._rocm_device_name() == "gfx942"

    # (3) available, no pin → the LIVE device arch (a gfx1100 host tags gfx1100,
    #     NOT the gfx1151 compile default — the PR #308 review fix).
    rt._rocm_device_name_probe = False
    monkeypatch.delenv("TESSERA_ROCM_CHIP", raising=False)
    monkeypatch.setattr(rt, "_rocm_live_arch", lambda: "gfx1100")
    assert rt._rocm_device_name() == "gfx1100"

    # (4) available, no pin, arch unknowable → None (never guess a default arch).
    rt._rocm_device_name_probe = False
    monkeypatch.setattr(rt, "_rocm_live_arch", lambda: None)
    assert rt._rocm_device_name() is None
    rt._rocm_device_name_probe = False  # leave the module cache clean


# ── 2. Corpus round-trip + warm-start (host-free) ─────────────────────────────

def _record(dev, bucket, winner, cands):
    return (dev, "rocm", OP_FUSED_REGION, bucket, "f16"), AT.MeasureRecord(
        winner=winner, latency_ms=cands[winner], candidates=cands)


def _seed_cache():
    cache = AT.MeasureCache()
    k1, r1 = _record("rocm:gfx1151", ("64", "64"), "rocm_generic_hip",
                     {"rocm_generic_hip": 0.8, "rocm_wmma_gemm": 1.8})
    k2, r2 = _record("rocm:gfx1151", ("512", "512"), "rocm_wmma_gemm",
                     {"rocm_generic_hip": 28.1, "rocm_wmma_gemm": 3.3})
    cache.put(k1, r1)
    cache.put(k2, r2)
    return cache


def test_corpus_round_trips_through_json():
    cache = _seed_cache()
    payload = cache.to_dict()
    assert payload["version"] == AT.CORPUS_VERSION
    assert len(payload["records"]) == 2

    fresh = AT.MeasureCache()
    n = fresh.load_dict(payload)
    assert n == 2
    assert fresh.to_dict() == payload            # key + record survive the trip


def test_corpus_save_load_disk(tmp_path):
    cache = _seed_cache()
    p = AT.save_corpus(tmp_path / "corpus.json", cache=cache)
    assert p.exists()
    fresh = AT.MeasureCache()
    assert AT.load_corpus(p, cache=fresh) == 2
    assert fresh.size == 2


def test_committed_corpus_contains_live_device_records_and_loads():
    # The corpus carries only device-keyed, measured records and warm-starts.
    fresh = AT.MeasureCache()
    n = AT.load_corpus(cache=fresh)
    assert n >= 1
    for rec in fresh.to_dict()["records"]:
        assert rec["device"] in ("rocm:gfx1151", "nvidia:sm_120")
        # Candidate names grow as new measured lanes land.  The durable corpus
        # invariant is that its winner was actually timed in that record.
        assert rec["candidates"]
        assert rec["winner"] in rec["candidates"]
        assert rec["latency_ms"] == rec["candidates"][rec["winner"]]


def test_committed_corpus_has_sm120_matmul_comparisons():
    fresh = AT.MeasureCache()
    AT.load_corpus(cache=fresh)
    rows = [r for r in fresh.to_dict()["records"]
            if r["device"] == "nvidia:sm_120" and r["op"] == OP_MATMUL
            and r["timing"] == "end_to_end"]
    assert {tuple(r["bucket"]) for r in rows} >= {
        (64, 64, 64), (256, 256, 256), (128, 256, 64),
        (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)
    }
    for row in rows:
        assert {"nvidia_mma_gemm_shipped", "nvidia_mma_gemm_emitted"} <= set(
            row["candidates"])
        if tuple(row["bucket"]) in {
                (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)}:
            assert {"nvidia_tile_matmul_direct",
                    "nvidia_tile_matmul_shared"} <= set(row["candidates"])


def test_committed_corpus_has_sm120_device_resident_tile_crossover():
    fresh = AT.MeasureCache()
    AT.load_corpus(cache=fresh)
    rows = [r for r in fresh.to_dict()["records"]
            if r["device"] == "nvidia:sm_120" and r["op"] == OP_MATMUL
            and r["timing"] == "device"]
    by_key = {(r["dtype"], tuple(r["bucket"])): r for r in rows}
    for dtype in ("float16", "bfloat16"):
        assert by_key[(dtype, (512, 512, 512))]["winner"] == (
            "nvidia_tile_matmul_direct")
        assert by_key[(dtype, (1024, 1024, 1024))]["winner"] == (
            "nvidia_tile_matmul_shared")
        assert set(by_key[(dtype, (1024, 1024, 1024))]["candidates"]) == {
            "nvidia_tile_matmul_direct", "nvidia_tile_matmul_shared"}


def test_committed_corpus_has_sm120_model_workload_comparisons():
    fresh = AT.MeasureCache()
    AT.load_corpus(cache=fresh)
    # Keep this assertion scoped to the original end-to-end f16 comparisons.
    # Device-timed TF32/FP8 rows intentionally share these op names and must not
    # overwrite them merely because they appear later in the persisted corpus.
    nvidia = [r for r in fresh.to_dict()["records"]
              if r["device"] == "nvidia:sm_120"
              and r["timing"] == "end_to_end" and r["dtype"] == "f16"]
    fused = [r for r in nvidia if r["op"] == OP_FUSED_REGION]
    attention = [r for r in nvidia if r["op"] == "attention"]
    assert fused and attention
    assert all(set(r["candidates"]) == {
        "nvidia_generic_cuda", "nvidia_mma_fused"} for r in fused)
    assert all(set(r["candidates"]) == {
        "nvidia_flash_attn", "nvidia_mma_attn"} for r in attention)

    fused_buckets = {tuple(r["bucket"]) for r in fused
                     if r["op"] == OP_FUSED_REGION}
    attention_buckets = {tuple(r["bucket"]) for r in attention}
    assert fused_buckets >= {(64, 64, 64), (256, 256, 256), (128, 512, 256)}
    assert attention_buckets >= {(128, 128, 64, 64), (64, 512, 64, 64)}


def test_warm_start_does_not_clobber_local_measurement():
    # A verdict already measured on this box wins over the corpus row (overwrite
    # defaults off) — measure-on-this-box is more trustworthy than a shipped row.
    cache = _seed_cache()
    k = ("rocm:gfx1151", "rocm", OP_FUSED_REGION, ("64", "64"), "f16")
    local = AT.MeasureRecord(winner="rocm_wmma_gemm", latency_ms=0.5,
                             candidates={"rocm_wmma_gemm": 0.5})
    cache.put(k, local)
    other = _seed_cache()          # a corpus that says generic wins the 64 bucket
    cache.load_dict(other.to_dict())
    assert cache.get(k).winner == "rocm_wmma_gemm"   # local kept, not clobbered
    # …unless overwrite is explicitly requested.
    cache.load_dict(other.to_dict(), overwrite=True)
    assert cache.get(k).winner == "rocm_generic_hip"


def test_corpus_version_mismatch_loads_nothing():
    fresh = AT.MeasureCache()
    assert fresh.load_dict({"version": 999, "records": [
        {"device": "d", "target": "rocm", "op": OP_FUSED_REGION,
         "bucket": None, "dtype": "f16", "winner": "w", "latency_ms": 1.0,
         "candidates": {}}]}) == 0
    assert fresh.size == 0


def test_v1_corpus_rows_migrate_to_end_to_end_timing():
    fresh = AT.MeasureCache()
    assert fresh.load_dict({"version": 1, "records": [
        {"device": "d", "target": "rocm", "op": OP_FUSED_REGION,
         "bucket": None, "dtype": "f16", "winner": "w", "latency_ms": 1.0,
         "candidates": {"w": 1.0}}]}) == 1
    row = fresh.to_dict()["records"][0]
    assert row["timing"] == "end_to_end"
    assert fresh.to_dict()["version"] == AT.CORPUS_VERSION == 2


def test_bucket_none_key_round_trips():
    # dims=None → bucket=None must survive serialization (a dynamic-shape verdict).
    cache = AT.MeasureCache()
    k = ("rocm:gfx1151", "rocm", OP_FUSED_REGION, None, "f16")
    cache.put(k, AT.MeasureRecord(winner="w", latency_ms=1.0))
    fresh = AT.MeasureCache()
    fresh.load_dict(cache.to_dict())
    assert fresh.get(k) is not None


def test_missing_corpus_loads_nothing(tmp_path):
    assert AT.load_corpus(tmp_path / "nope.json", cache=AT.MeasureCache()) == 0


# ── 3. Live gate — measured arbitrate on gfx1151 ──────────────────────────────

@pytest.mark.skipif(not _rocm_hip_live(),
                    reason="needs a live gfx1151 + hipcc")
def test_live_rocm_device_tag():
    from tessera import runtime as rt
    tag = rt._rocm_device_name()
    assert tag is not None and tag.startswith("gfx")   # the real live arch
    assert AT._device_id("rocm") == f"rocm:{tag}"


@pytest.mark.skipif(not _rocm_hip_live(),
                    reason="needs a live gfx1151 + hipcc")
def test_live_measured_arbitrate_caches_per_bucket():
    region = F.FusedRegion(epilogue=("bias", "gelu"))
    rng = np.random.default_rng(0)
    cache = AT.MeasureCache()

    def _run(S):
        A = rng.standard_normal((S, S)).astype(np.float32)
        B = rng.standard_normal((S, S)).astype(np.float32)
        bias = rng.standard_normal((S,)).astype(np.float32)
        return AT.measured_arbitrate(region, OP_FUSED_REGION, "rocm", A, B, bias,
                                     dims=(S, S), dtype="f16", cache=cache,
                                     reps=6, warmup=2)

    w_small = _run(64)
    assert w_small is not None
    assert cache.misses == 1 and cache.size == 1

    # Re-query the same bucket → cache hit, no re-measure.
    before = cache.hits
    w_small2 = _run(64)
    assert cache.hits == before + 1
    assert w_small2.name == w_small.name

    # A distinct bucket measures separately.
    w_big = _run(512)
    assert w_big is not None
    assert cache.size == 2


@pytest.mark.skipif(not _rocm_hip_live(),
                    reason="needs a live gfx1151 + hipcc")
def test_live_measure_overrides_tier_priority_on_tiny_shape():
    # The whole point of D2: at a tiny shape the measured loop prefers the Tier-1
    # generic lane over the Tier-3 WMMA crown jewel (which tier-priority would
    # pick). Both must be F4-passing for this to be a fair, lead-safe override.
    region = F.FusedRegion(epilogue=("bias", "gelu"))
    rng = np.random.default_rng(1)
    A = rng.standard_normal((64, 64)).astype(np.float32)
    B = rng.standard_normal((64, 64)).astype(np.float32)
    bias = rng.standard_normal((64,)).astype(np.float32)
    cache = AT.MeasureCache()
    w = AT.measured_arbitrate(region, OP_FUSED_REGION, "rocm", A, B, bias,
                              dims=(64, 64), dtype="f16", cache=cache,
                              reps=8, warmup=3)
    rec = cache.to_dict()["records"][0]
    # both lanes were timed (fair comparison), and the faster one was chosen
    assert set(rec["candidates"]) == {"rocm_generic_hip", "rocm_wmma_gemm"}
    assert w.name == min(rec["candidates"], key=rec["candidates"].get)
