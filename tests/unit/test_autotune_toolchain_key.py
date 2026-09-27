"""Decision #11 (amended 2026-08-30): the autotune key carries the toolchain and
delegate identity, so a toolkit upgrade or rebuilt delegate MISSES rather than
returning a stale measurement.

Covers every persisted autotune cache: `autotune_v2` (SQLite, and the public
`tessera.autotune` facade over it), the arbiter's `emit.autotune` corpus,
`flywheel` dispatch distillation and `tuned_dispatch` (their tests live beside
them). For each: an unchanged toolchain hits, a changed one misses, and a row
written before the key carried the toolchain never returns a stale hit.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from tessera.compiler import gpu_target, rocm_target
from tessera.compiler import toolchain_identity as TI
from tessera.compiler.autotune_v2 import (
    BayesianAutotuner,
    GEMMWorkload,
    TuningConfig,
    TuningResult,
)
from tessera.compiler.emit import autotune as AT
from tessera.compiler.emit.candidate import OP_MATMUL, Candidate, Tier, register_candidate


@pytest.fixture(autouse=True)
def _fresh_identity_cache():
    TI.clear_identity_cache()
    yield
    TI.clear_identity_cache()


# ── the identity itself ─────────────────────────────────────────────────────

def test_identity_reads_the_declared_pins_not_a_probe():
    nv = TI.toolchain_identity("sm_120")
    assert nv.family == "nvidia"
    assert nv.toolchain["cuda_toolkit"] == gpu_target.TESSERA_TARGET_CUDA_TOOLKIT
    assert nv.toolchain["ptx_isa"] == gpu_target.TESSERA_TARGET_PTX_ISA
    assert nv.toolchain["driver_jit_ptx_isa"] == gpu_target.TESSERA_TARGET_DRIVER_JIT_PTX_ISA
    rocm = TI.toolchain_identity("rocm:gfx1151")
    assert rocm.family == "rocm"
    assert rocm.toolchain["rocm"] == rocm_target.TESSERA_TARGET_ROCM
    assert rocm.toolchain["hip"] == rocm_target.TESSERA_TARGET_HIP
    # The LLVM pin comes from TesseraToolchainPins.cmake via runtime_abi_audit.
    from tessera.compiler.runtime_abi_audit import cmake_toolchain_pins
    assert nv.toolchain["llvm_mlir"] == cmake_toolchain_pins()["llvm_version"]


def test_arch_spellings_of_one_family_share_an_identity():
    assert TI.toolchain_identity("sm_120").digest == TI.toolchain_identity("nvidia").digest
    assert TI.toolchain_identity("gfx1201").digest == TI.toolchain_identity("rocm:gfx1151").digest
    assert TI.toolchain_identity("rocm").digest != TI.toolchain_identity("nvidia").digest


def test_a_pin_bump_changes_the_digest(monkeypatch):
    before = TI.toolchain_identity("nvidia").digest
    monkeypatch.setattr(gpu_target, "TESSERA_TARGET_CUDA_TOOLKIT", "13.5")
    TI.clear_identity_cache()
    assert TI.toolchain_identity("nvidia").digest != before
    # ...and only the family whose toolchain moved.
    assert TI.toolchain_identity("rocm").digest == TI.toolchain_identity("gfx1151").digest


def test_artifact_identities_key_rebuilds(tmp_path, monkeypatch):
    """P1-4: identity is pin-based; what catches a rebuilt artifact is the
    per-candidate identity -- a delegate library's or tessera-opt's digest."""
    base = TI.toolchain_identity("rocm")
    opt = tmp_path / "tessera-opt"
    opt.write_bytes(b"compiler-build-1")
    from tessera import runtime as rt
    monkeypatch.setattr(rt, "_tessera_opt_path", lambda: opt)
    first = TI.tessera_opt_identity()
    assert first["generator"] == "tessera-opt"
    opt.write_bytes(b"compiler-build-2-rebuilt")
    assert TI.tessera_opt_identity()["abi_digest"] != first["abi_digest"]
    monkeypatch.setattr(rt, "_tessera_opt_path", lambda: None)
    assert TI.tessera_opt_identity() is None

    lib = tmp_path / "libdelegate.so"
    lib.write_bytes(b"v1")
    v1 = TI.delegate_library_identity(lib, cmake_target="tessera_x")
    assert v1["build_record"] == "unrecorded"      # not inside a build tree
    lib.write_bytes(b"v2-rebuilt")
    v2 = TI.delegate_library_identity(lib)
    assert v1["abi_digest"] != v2["abi_digest"]
    assert base.with_delegate(v1).digest != base.with_delegate(v2).digest


# ── autotune_v2 SQLite cache (and the public facade over it) ────────────────

def _tuner(toolchain=None):
    tuner = BayesianAutotuner(GEMMWorkload(M=256, N=256, K=256, arch="sm_120"),
                              toolchain=toolchain)
    tuner._results.append(TuningResult(
        config=TuningConfig(128, 128, 32), latency_ms=1.0, tflops=33.5,
        trial_id=0, method="measured"))
    return tuner


def test_sqlite_unchanged_toolchain_hits(tmp_path):
    db = str(tmp_path / "t.db")
    _tuner().save_to_cache(db)
    reader = BayesianAutotuner(GEMMWorkload(M=256, N=256, K=256, arch="sm_120"))
    assert reader.warm_start_from_cache(db) == 1
    assert reader.warm_start_skipped == []


def test_sqlite_changed_toolchain_misses_and_says_so(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    _tuner().save_to_cache(db)
    monkeypatch.setattr(gpu_target, "TESSERA_TARGET_CUDA_TOOLKIT", "13.5")
    TI.clear_identity_cache()
    upgraded = BayesianAutotuner(GEMMWorkload(M=256, N=256, K=256, arch="sm_120"))
    assert upgraded.warm_start_from_cache(db) == 0
    assert upgraded.best is None
    assert len(upgraded.warm_start_skipped) == 1
    assert "measured under toolchain" in upgraded.warm_start_skipped[0]


def test_sqlite_delegate_rebuild_misses(tmp_path):
    db = str(tmp_path / "t.db")
    base = TI.toolchain_identity("sm_120")
    _tuner(base.with_delegate({"abi_digest": "sha256:v1"})).save_to_cache(db)
    same = BayesianAutotuner(GEMMWorkload(M=256, N=256, K=256, arch="sm_120"),
                             toolchain=base.with_delegate({"abi_digest": "sha256:v1"}))
    assert same.warm_start_from_cache(db) == 1
    rebuilt = BayesianAutotuner(GEMMWorkload(M=256, N=256, K=256, arch="sm_120"),
                                toolchain=base.with_delegate({"abi_digest": "sha256:v2"}))
    assert rebuilt.warm_start_from_cache(db) == 0


def test_sqlite_old_schema_row_is_never_a_stale_hit(tmp_path):
    """A cache written by a build whose schema had no toolchain column."""
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE tuning_results (M INT, N INT, K INT, dtype TEXT,"
            " arch TEXT, layout TEXT, movement_json TEXT,"
            " tile_m INT, tile_n INT, tile_k INT, num_warps INT, num_stages INT,"
            " latency_ms REAL, tflops REAL, sampled_at REAL, trial_id INT)")
        conn.execute(
            "INSERT INTO tuning_results VALUES (256,256,256,'bf16','sm_120',"
            "'row_major','{\"overlap\": \"compute\", \"prefetch\": \"auto\"}',"
            "128,128,32,4,2,1.0,33.5,0.0,0)")
    tuner = BayesianAutotuner(GEMMWorkload(M=256, N=256, K=256, arch="sm_120"))
    assert tuner.warm_start_from_cache(str(db)) == 0
    assert "no toolchain identity" in tuner.warm_start_skipped[0]
    # Writing to the old file migrates it; the legacy row stays a miss.
    _tuner().save_to_cache(str(db))
    again = BayesianAutotuner(GEMMWorkload(M=256, N=256, K=256, arch="sm_120"))
    assert again.warm_start_from_cache(str(db)) == 1
    assert "no toolchain identity" in again.warm_start_skipped[0]


def test_cache_key_and_schedule_artifact_carry_the_toolchain(monkeypatch):
    tuner = _tuner()
    key = tuner.cache_key(arch="sm_120")
    assert key["toolchain"]["digest"] == TI.toolchain_identity("sm_120").digest
    before = tuner.schedule_hash(arch="sm_120")
    upgraded = BayesianAutotuner(
        tuner.workload, toolchain=TI.toolchain_identity("sm_120").with_delegate({"x": "y"}))
    upgraded._results = list(tuner._results)
    upgraded._best = tuner.best
    assert upgraded.schedule_hash(arch="sm_120") != before


def test_public_facade_cache_key_carries_the_toolchain(monkeypatch):
    from tessera import autotune as public
    key = public.cache_key("matmul", (128, 128, 128), arch="sm_120")
    assert key[-1] == TI.toolchain_identity("sm_120").digest
    monkeypatch.setattr(gpu_target, "TESSERA_TARGET_CUDA_TOOLKIT", "13.5")
    TI.clear_identity_cache()
    assert public.cache_key("matmul", (128, 128, 128), arch="sm_120") != key


# ── the arbiter's measured corpus (emit.autotune) ───────────────────────────

KEY = ("nvidia:sm_120", "nvidia", OP_MATMUL, (64, 64, 64), "float16", "device")


def _record():
    return AT.MeasureRecord(winner="w", latency_ms=1.0, candidates={"w": 1.0},
                            unmeasured={})


def test_corpus_unchanged_toolchain_hits():
    cache = AT.MeasureCache()
    cache.put(KEY, _record(), fresh=True)
    assert cache.get(KEY).evidence["toolchain_digest"] == TI.toolchain_identity("nvidia").digest
    warm = AT.MeasureCache()
    assert warm.load_dict(cache.to_dict()) == 1
    assert warm.get(KEY) is not None and not warm.stale_records()


def test_corpus_changed_toolchain_misses_but_is_kept(monkeypatch):
    cache = AT.MeasureCache()
    cache.put(KEY, _record(), fresh=True)
    payload = cache.to_dict()
    monkeypatch.setattr(gpu_target, "TESSERA_TARGET_CUDA_TOOLKIT", "13.5")
    TI.clear_identity_cache()
    upgraded = AT.MeasureCache()
    assert upgraded.load_dict(payload) == 0
    assert upgraded.get(KEY) is None
    (_, reason), = upgraded.stale_records().values()
    assert "measured under toolchain" in reason
    # A recorder that loads and re-saves does not delete the old evidence.
    assert upgraded.to_dict()["records"] == payload["records"]
    # A fresh measurement under the new toolchain supersedes it.
    upgraded.put(KEY, _record(), fresh=True)
    assert upgraded.get(KEY) is not None and not upgraded.stale_records()
    assert len(upgraded.to_dict()["records"]) == 1


def test_pre_v4_corpus_rows_are_never_stale_hits():
    v3 = {"version": 3, "records": [{
        "device": "nvidia:sm_120", "target": "nvidia", "op": OP_MATMUL,
        "bucket": [64, 64, 64], "dtype": "float16", "timing": "device",
        "winner": "w", "latency_ms": 1.0, "candidates": {"w": 1.0},
        "evidence": {"compiler_fingerprint": "sha256:x"}}]}
    cache = AT.MeasureCache()
    assert cache.load_dict(v3) == 0
    assert cache.get(KEY) is None
    (_, reason), = cache.stale_records().values()
    assert "no toolchain identity" in reason


def test_committed_corpus_serves_only_current_toolchain_rows():
    """A committed row selects a route only if it carries today's identity;
    every other row is held stale, and none is lost on a load/save round trip
    (rows recorded before 2026-09-26 carry no identity at all)."""
    import json

    cache = AT.MeasureCache()
    loaded = AT.load_corpus(cache=cache)
    stale = cache.stale_records()
    on_disk = json.loads(AT.corpus_path().read_text())["records"]
    assert loaded + len(stale) == len(on_disk) == len(cache.to_dict()["records"])
    for key, rec in cache._store.items():
        assert rec.evidence["toolchain_digest"] == TI.toolchain_identity(key[1]).digest


def test_put_never_resurrects_a_stale_row():
    """Reviewer's repro (P1-1): load the committed corpus, take a row from
    `stale_records()`, `put` it. It used to be stamped current, served, and
    re-saved as fresh v4."""
    cache = AT.MeasureCache()
    AT.load_corpus(cache=cache)
    stale = cache.stale_records()
    if not stale:
        pytest.skip("the committed corpus holds no stale rows to resurrect")
    key, (record, _) = next(iter(stale.items()))
    with pytest.raises(ValueError, match="no toolchain identity|stale"):
        cache.put(key, record)
    with pytest.raises(ValueError, match="stale"):
        cache.put(key, record, fresh=True)          # not even when vouched for
    assert cache.get(key) is None
    assert key in cache.stale_records()
    saved = {AT._key_from_json(r): r for r in cache.to_dict()["records"]}
    assert "toolchain_digest" not in saved[key].get("evidence", {})


def test_identity_less_record_needs_an_explicit_fresh_measurement():
    with pytest.raises(ValueError, match="fresh=True"):
        AT.MeasureCache().put(KEY, _record())
    cache = AT.MeasureCache()
    cache.put(KEY, _record(), fresh=True)
    assert cache.get(KEY) is not None


def test_a_stale_row_never_replaces_a_fresh_one():
    cache = AT.MeasureCache()
    cache.put(KEY, _record(), fresh=True)
    fresh = cache.get(KEY)
    old = {"version": 3, "records": [{**AT._key_to_json(KEY), "winner": "old",
                                      "latency_ms": 9.0, "candidates": {"old": 9.0}}]}
    for overwrite in (False, True):
        assert cache.load_dict(old, overwrite=overwrite) == 0
        assert cache.get(KEY) is fresh
    assert [r["winner"] for r in cache.to_dict()["records"]] == ["w"]


def test_every_hand_tuned_candidate_declares_an_artifact_identity():
    """P1-2 guard: a Tier-3 candidate is a versioned artifact (a delegate
    library, or a kernel tessera-opt generates at run time). One registered
    without `delegate_identity()` would let a rebuilt artifact reuse a stale
    verdict, so the registry is enumerated rather than listed by hand."""
    import importlib
    import pkgutil

    import tessera.compiler.emit as emit_pkg
    from tessera.compiler.emit.candidate import _CANDIDATES

    for mod in pkgutil.iter_modules(emit_pkg.__path__):
        importlib.import_module(f"tessera.compiler.emit.{mod.name}")
    for extra in ("tessera.compiler.native_ann", "tessera.compiler.native_ann_gpu"):
        importlib.import_module(extra)

    hand_tuned = [c for cands in _CANDIDATES.values() for c in cands
                  if c.tier == Tier.HAND_TUNED and not type(c).__module__.startswith("tests")
                  and type(c).__module__.startswith("tessera.")]
    assert hand_tuned, "the registry enumeration found no Tier-3 candidates"
    missing = sorted(c.name for c in hand_tuned
                     if type(c).delegate_identity is Candidate.delegate_identity)
    assert not missing, f"Tier-3 candidates without delegate_identity(): {missing}"


def test_apple_identity_does_not_depend_on_path(tmp_path):
    """P1-3: the digest must not move when Homebrew llvm is first on PATH."""
    import os
    import subprocess
    import sys

    from tests._support.apple import require_apple_metal

    require_apple_metal()
    fake = tmp_path / "bin"
    fake.mkdir()
    clang = fake / "clang"
    clang.write_text("#!/bin/sh\necho 'not Apple clang 99.9'\n")
    clang.chmod(0o755)
    code = ("from tessera.compiler.toolchain_identity import toolchain_identity;"
            "print(toolchain_identity('apple_gpu').digest)")
    env = dict(os.environ, PYTHONPATH="python")

    def digest(path_prefix):
        run_env = dict(env, PATH=path_prefix + os.pathsep + env.get("PATH", ""))
        return subprocess.run([sys.executable, "-c", code], env=run_env, check=True,
                              capture_output=True, text=True).stdout.strip()

    assert digest(str(fake)) == digest("/usr/bin")


def test_put_refuses_a_foreign_toolchain_record():
    rec = _record()
    rec.evidence.update({"toolchain_digest": "sha256:someone-elses"})
    with pytest.raises(ValueError, match="Decision #11"):
        AT.MeasureCache().put(KEY, rec, fresh=True)


class _Region:
    dtype = "float16"

    def reference(self, A, B):
        return np.asarray(A, np.float32) @ np.asarray(B, np.float32)


class _Delegate(Candidate):
    op = OP_MATMUL
    tier = Tier.HAND_TUNED

    def __init__(self, name, target, build):
        self.name, self.target, self.build = name, target, build
        self.timed = 0

    def delegate_identity(self):
        return {"library": "libfake.so", "abi_digest": self.build}

    def run(self, region, A, B, *a, **k):
        return region.reference(A, B), "fake_delegate"

    def measure_device_latency(self, region, *inputs, reps=100, warmup=10):
        self.timed += 1
        return 1.0


def test_measured_verdict_misses_when_the_delegate_is_rebuilt():
    tgt = "d11_delegate_target"
    cand = _Delegate("d11_delegate", tgt, "sha256:build-1")
    register_candidate(cand)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4)).astype(np.float32)
    B = rng.standard_normal((4, 4)).astype(np.float32)
    cache = AT.MeasureCache()

    def race():
        return AT.measured_arbitrate(
            _Region(), OP_MATMUL, tgt, A, B, dims=(4, 4, 4), dtype="float16",
            cache=cache, device="fakedev", timing=AT.TIMING_DEVICE, device_repeats=1)

    assert race().name == "d11_delegate"
    rec = next(iter(cache._store.values()))
    assert rec.evidence["delegate_identities"]["d11_delegate"]["abi_digest"] == "sha256:build-1"
    timed = cand.timed
    race()
    assert cand.timed == timed, "same build: the cached verdict must be reused"
    cand.build = "sha256:build-2"
    assert AT.corpus_winner(
        _Region(), OP_MATMUL, tgt, A, B, dims=(4, 4, 4), dtype="float16",
        cache=cache, device="fakedev", timing=AT.TIMING_DEVICE) is None
    race()
    assert cand.timed > timed, "a rebuilt delegate must be re-measured"
