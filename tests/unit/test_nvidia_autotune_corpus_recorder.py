"""Host-free checks of ``benchmarks/nvidia/record_autotune_corpus.py`` under the
Decision #11 (v4) corpus.

The recorder used to start from an empty cache unless ``--warm-start`` was
passed, and ``save_corpus`` writes the whole cache -- so a default sm_120 run
deleted every gfx1151 row. With ``--warm-start`` it stamped the nvcc evidence
block onto every row in ``_store``, which after Decision #11 includes a current
gfx1151 row on any host (the ROCm identity is pin-derived). Both are exercised
here with the device, nvcc and arbiter stubbed; nothing is measured.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_recorder():
    spec = importlib.util.spec_from_file_location(
        "_record_autotune_corpus_under_test",
        ROOT / "benchmarks/nvidia/record_autotune_corpus.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(device, target, op, bucket, dtype, timing, winner, evidence=None):
    return {"device": device, "target": target, "op": op, "bucket": bucket,
            "dtype": dtype, "timing": timing, "winner": winner,
            "latency_ms": 1.0, "candidates": {winner: 1.0},
            **({"evidence": evidence} if evidence is not None else {})}


@pytest.fixture
def recorder_env(tmp_path, monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler.emit import autotune as at
    from tessera.compiler.emit import nvidia_cuda
    from tessera.compiler.emit.kernel_emitter import SpecPolicy, bucket_key

    rocm_evidence = {"separation_note": "rocm-owned", **at.toolchain_evidence("rocm")}
    mm_bucket = list(bucket_key((64, 64, 64), SpecPolicy.BUCKET))
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"version": 3, "records": [
        _row("rocm:gfx1151", "rocm", "paged_kv_decode", [1, 4, 4, 512, 32, 16],
             "f32", "end_to_end", "direct", rocm_evidence),
        _row("nvidia:sm_120", "nvidia", "matmul", mm_bucket, "float16",
             "device", "stale_winner"),
        _row("nvidia:sm_120", "nvidia", "matmul", [9999, 9999, 9999],
             "float16", "device", "never_rerun"),
        _row("nvidia:sm_120", "nvidia", "paged_kv_decode", [1, 8, 128, 64],
             "f32", "device", "fused_paged_attention"),
    ]}))
    monkeypatch.setenv("TESSERA_AUTOTUNE_CORPUS", str(corpus))
    monkeypatch.setattr(rt, "_nvidia_device_name", lambda: "sm_120")

    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="nvcc: stub 13.4\n"))

    def fake_arbitrate(region, op, target, *inputs, dims=None, dtype="f32",
                       cache=None, reps=20, warmup=3, timing="end_to_end",
                       device_repeats=3, device=None):
        key = ("nvidia:sm_120", target, op,
               bucket_key(dims, SpecPolicy.BUCKET), dtype, timing)
        if cache.get(key) is None:
            cache.put(key, at.MeasureRecord("fresh_winner", 1.0,
                                            {"fresh_winner": 1.0}), fresh=True)
        return SimpleNamespace(name="fresh_winner")

    monkeypatch.setattr(at, "measured_arbitrate", fake_arbitrate)
    monkeypatch.setattr(nvidia_cuda, "run_conv2d_resident_candidate",
                        lambda x, w, route, **k: (None, {"direct": 1.0,
                                                         "shared": 2.0,
                                                         "im2col_tf32": 3.0}[route]))
    return corpus, rocm_evidence, mm_bucket


_ARGS = ["--matmul-shapes", "64x64x64", "--fused-shapes", "64x64x64",
         "--attention-shapes", "64x64x64x64", "--gated-shapes", "64x64x64",
         "--conv-shapes", "1x8x8x8x3x3x8", "--matmul-dtypes", "float16",
         "--composed-dtypes", "f32"]


@pytest.mark.parametrize("warm_start", [False, True])
def test_recorder_keeps_other_devices_and_replaces_stale_rows(
        recorder_env, monkeypatch, capsys, warm_start):
    corpus, rocm_evidence, mm_bucket = recorder_env
    recorder = _load_recorder()
    monkeypatch.setattr(sys, "argv", ["record_autotune_corpus.py", *_ARGS,
                                      *(["--warm-start"] if warm_start else [])])
    assert recorder.main() == 0
    out = capsys.readouterr().out

    from tessera.compiler.emit import autotune as at
    from tessera.compiler.toolchain_identity import toolchain_identity

    rows = json.loads(corpus.read_text())["records"]
    assert json.loads(corpus.read_text())["version"] == at.CORPUS_VERSION
    by_key = {(r["device"], r["op"], tuple(r["bucket"]), r["dtype"], r["timing"]): r
              for r in rows}

    # The gfx1151 row survives byte-for-byte in evidence: no nvcc fingerprint.
    rocm = by_key[("rocm:gfx1151", "paged_kv_decode", (1, 4, 4, 512, 32, 16),
                   "f32", "end_to_end")]
    assert rocm["evidence"] == rocm_evidence

    # The stale sm_120 row at a re-raced key is replaced by a stamped row.
    mm = by_key[("nvidia:sm_120", "matmul", tuple(mm_bucket), "float16", "device")]
    assert mm["winner"] == "fresh_winner"
    assert mm["evidence"]["toolchain_digest"] == toolchain_identity("nvidia").digest
    assert mm["evidence"]["compiler_fingerprint"].startswith("sha256:")

    # Rows this invocation did not re-race stay (stale), and the owned one is named.
    assert by_key[("nvidia:sm_120", "matmul", (9999, 9999, 9999), "float16",
                   "device")]["winner"] == "never_rerun"
    assert "9999" in out and "remain stale" in out
    serving = by_key[("nvidia:sm_120", "paged_kv_decode", (1, 8, 128, 64), "f32",
                      "device")]
    assert serving["winner"] == "fused_paged_attention"
    assert "evidence" not in serving or "toolchain_digest" not in serving["evidence"]

    # Every row this run wrote is served under today's identity.
    cache = at.MeasureCache()
    at.load_corpus(corpus, cache=cache)
    served_owned = [k for k in cache._store if recorder._owned(k)]
    assert served_owned
    for key in served_owned:
        assert "compiler_fingerprint" in cache._store[key].evidence
