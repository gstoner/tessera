"""KV-cache paged-movement lane on AMD ROCm gfx1151 (§5.6 of the ROCm MFMA
kernel inventory) — the append/read/prune core executed on-device by COMPOSING
the existing gfx1151 scatter (append row write) + masked-gather (read/prune)
kernels with host page-index math. Reachable via
`compiler_path="rocm_kv_cache_compiled"`. Validated against the KVCacheHandle
reference on gfx1151. Skip-clean: tessera-opt not built / no GPU.

The three kv_cache_* ops are stateful over a KVCacheHandle; this lane executes
their tensor movement over a resident cache buffer `(max_seq, H, D)`. K and V
are independent buffers, so the lane runs once per buffer — the tests drive the
K buffer and compare against `handle.keys` (the V path is identical).
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, op, operands, kwargs):
    names = ["b", "n"][: len(operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_kv_cache_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs}]})


def _run(rt, op, operands, **kwargs):
    res = rt.launch(_art(rt, op, operands, kwargs), tuple(operands))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_kv_cache_compiled"
    return np.asarray(res["output"], np.float32)


_RNG = np.random.default_rng(41)


def _handle(max_seq=32, H=3, D=8):
    return tessera.cache.KVCacheHandle(num_heads=H, head_dim=D, max_seq=max_seq)


# ── append ────────────────────────────────────────────────────────────────

def test_append_from_empty():
    rt = _rocm_or_skip()
    h = _handle()
    k = _RNG.standard_normal((5, 3, 8)).astype(np.float32)
    v = _RNG.standard_normal((5, 3, 8)).astype(np.float32)
    h.append(k, v)
    buf = np.zeros((32, 3, 8), np.float32)      # resident K buffer, empty
    out = _run(rt, "tessera.kv_cache.append", [buf, k], start=0)
    np.testing.assert_allclose(out, h.keys.astype(np.float32), rtol=0, atol=0)


def test_append_midstream():
    rt = _rocm_or_skip()
    h = _handle()
    k0 = _RNG.standard_normal((7, 3, 8)).astype(np.float32)
    h.append(k0, k0)                             # prefill 7 tokens
    k1 = _RNG.standard_normal((4, 3, 8)).astype(np.float32)
    # lane appends onto the buffer that already holds the prefill.
    buf = h.keys.astype(np.float32).copy()
    out = _run(rt, "tessera.kv_cache.append", [buf, k1], start=h.current_seq)
    h.append(k1, k1)
    np.testing.assert_allclose(out, h.keys.astype(np.float32), rtol=0, atol=0)


def test_append_out_of_bounds_rejected():
    rt = _rocm_or_skip()
    buf = np.zeros((8, 2, 4), np.float32)
    new = _RNG.standard_normal((5, 2, 4)).astype(np.float32)
    res = rt.launch(
        _art(rt, "tessera.kv_cache.append", [buf, new], {"start": 6}),
        (buf, new))
    assert res["ok"] is False           # [6, 11) exceeds max_seq=8


# ── read ──────────────────────────────────────────────────────────────────

def test_read_slice():
    rt = _rocm_or_skip()
    h = _handle()
    k = _RNG.standard_normal((12, 3, 8)).astype(np.float32)
    h.append(k, k)
    ks, _ = h.read(2, 9)
    out = _run(rt, "tessera.kv_cache.read", [h.keys.astype(np.float32)],
               start=2, end=9)
    np.testing.assert_allclose(out, np.asarray(ks, np.float32), rtol=0, atol=0)


def test_read_single_token_default_end():
    rt = _rocm_or_skip()
    h = _handle()
    k = _RNG.standard_normal((6, 3, 8)).astype(np.float32)
    h.append(k, k)
    ks, _ = h.read(4)                            # default end = start+1
    out = _run(rt, "tessera.kv_cache.read", [h.keys.astype(np.float32)],
               start=4)
    np.testing.assert_allclose(out, np.asarray(ks, np.float32), rtol=0, atol=0)


# ── prune ─────────────────────────────────────────────────────────────────

def test_prune_sliding_window():
    rt = _rocm_or_skip()
    h = _handle()
    k = _RNG.standard_normal((20, 3, 8)).astype(np.float32)
    h.append(k, k)
    buf = h.keys.astype(np.float32).copy()
    out = _run(rt, "tessera.kv_cache.prune", [buf], limit=6,
               current_seq=h.current_seq)
    h.prune(6)                                   # keep trailing 6
    np.testing.assert_allclose(out, h.keys.astype(np.float32), rtol=0, atol=0)


def test_prune_noop_when_limit_exceeds_seq():
    rt = _rocm_or_skip()
    h = _handle()
    k = _RNG.standard_normal((5, 3, 8)).astype(np.float32)
    h.append(k, k)
    buf = h.keys.astype(np.float32).copy()
    out = _run(rt, "tessera.kv_cache.prune", [buf], limit=99,
               current_seq=h.current_seq)
    np.testing.assert_allclose(out, buf, rtol=0, atol=0)
