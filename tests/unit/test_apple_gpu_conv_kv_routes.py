"""Workstream #11 — Apple GPU conv2d + kv_cache_read read paths.

HONEST SCOPE (2026-06-19): both ops have working Metal execution via their
*direct* dispatchers, but their ``@jit→launch`` integration is still
``unimplemented`` (launch returns ``artifact_only`` for conv2d), so the
conformance Evaluator correctly will not corroborate them at rung 7. We do NOT
claim conformance ``complete`` here. What IS landed and tested:

  * conv2d computes correctly on Metal via ``_apple_gpu_dispatch_conv2d``.
  * ``apple_gpu_kv_cache_read`` returns a device-resident (DeviceTensor) slice
    view on unified memory, provenance-gated on ``DeviceTensor.is_metal()``.

The remaining gap (the conformance closer) is the @jit→launch wiring — tracked
as a follow-on. See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (#11).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.cache import KVCacheHandle
from tessera.runtime import apple_gpu_kv_cache_read, DeviceTensor


# ── conv2d genuinely computes on Metal (direct dispatcher) ───────────────────


def test_conv2d_dispatch_matches_reference():
    from tessera.runtime import _apple_gpu_dispatch_conv2d
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 8, 8, 3)).astype(np.float32)   # NHWC
    w = rng.standard_normal((3, 3, 3, 4)).astype(np.float32)   # kH,kW,Cin,Cout
    out = _apple_gpu_dispatch_conv2d([x, w], {"stride": 1, "padding": 0,
                                              "layout": "nhwc"}, np)
    assert out is not None
    out = np.asarray(out)
    assert out.shape == (1, 6, 6, 4)
    ref = np.zeros((1, 6, 6, 4), np.float32)
    for i in range(6):
        for j in range(6):
            patch = x[0, i:i + 3, j:j + 3, :]
            for co in range(4):
                ref[0, i, j, co] = np.sum(patch * w[:, :, :, co])
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


def test_conv2d_jit_launch_is_honestly_unimplemented():
    """Pin the honest gap: the @jit→launch path does NOT yet execute conv2d —
    it reports artifact_only/unimplemented (not a silent wrong result). When the
    launch wiring lands, this test flips and conv2d can claim conformance complete."""
    import tessera
    @tessera.jit(target="apple_gpu")
    def f(x, w):
        return tessera.ops.conv2d(x, w, stride=1, padding=0, layout="nhwc")
    x = np.random.randn(1, 8, 8, 3).astype(np.float32)
    w = np.random.randn(3, 3, 3, 4).astype(np.float32)
    res = tessera.runtime.launch(f.runtime_artifact(), (x, w))
    # Honest provenance — not executed via launch yet, and never a fake success.
    assert res.get("runtime_status") != "success" or \
        res.get("execution_kind") not in ("native_gpu", "metal_runtime")


# ── kv_cache_read device-resident slice read ─────────────────────────────────


def test_kv_cache_read_device_resident_matches_source():
    h = KVCacheHandle(num_heads=2, head_dim=4, max_seq=32, page_size=8)
    k = np.arange(10 * 2 * 4, dtype=np.float32).reshape(10, 2, 4)
    v = k + 100.0
    h.append(k, v)
    kview, vview, exe = apple_gpu_kv_cache_read(h, 2, 7)
    np.testing.assert_array_equal(kview, k[2:7])
    np.testing.assert_array_equal(vview, v[2:7])
    assert exe in {"metal_runtime", "reference"}
    if DeviceTensor.is_metal():
        assert exe == "metal_runtime"   # genuine device residency


def test_kv_cache_read_single_token():
    h = KVCacheHandle(num_heads=1, head_dim=4, max_seq=16)
    k = np.ones((4, 1, 4), np.float32) * 3.0
    h.append(k, k)
    kview, vview, exe = apple_gpu_kv_cache_read(h, 0)  # single-token read
    assert kview.shape == (1, 1, 4)
    np.testing.assert_array_equal(kview, k[0:1])


def test_kv_cache_read_provenance_honest_without_metal(monkeypatch):
    # Force is_metal False → must report reference, never claim native.
    monkeypatch.setattr(DeviceTensor, "is_metal", staticmethod(lambda: False))
    h = KVCacheHandle(num_heads=1, head_dim=2, max_seq=8)
    h.append(np.ones((3, 1, 2), np.float32), np.ones((3, 1, 2), np.float32))
    _, _, exe = apple_gpu_kv_cache_read(h, 0, 3)
    assert exe == "reference"
