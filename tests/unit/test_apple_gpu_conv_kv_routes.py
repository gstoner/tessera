"""Workstream #11/#17 — Apple GPU conv2d + kv_cache_read read paths.

SCOPE (2026-06-20): the @jit→launch wiring (the #17 conformance closer) landed
for conv2d, which now **executes through @jit→launch on Metal** with honest
provenance — so conv2d/apple_gpu is conformance ``complete`` and the generic
Evaluator corroborates it at HARDWARE_VERIFIED. kv_cache_read is a *stateful*
op (over ``KVCacheType``); its device-resident read executes on Metal via
``apple_gpu_kv_cache_read`` (provenance-gated, proven by the Evaluator verdict
``kv_cache_read_native_equivalence``), but it does not flow the pure-tensor
@jit→launch conformance matrix, so its cell stays below ``complete`` honestly
(numerical fixture for a stateful op is the remaining gated piece).

What IS landed and tested here:

  * conv2d executes through @jit→launch on Metal (``native_gpu`` when the Metal
    symbol ran; honest ``reference`` on host fallback — never fake native).
  * conv2d computes correctly on Metal via ``_apple_gpu_dispatch_conv2d``.
  * ``apple_gpu_kv_cache_read`` returns a device-resident (DeviceTensor) slice
    view on unified memory, provenance-gated on ``DeviceTensor.is_metal()``.

See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (#11/#17).
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


def test_conv2d_jit_launch_executes_natively():
    """#17 closer: conv2d now executes through @jit→launch on Metal. Provenance
    is honest — ``native_gpu`` only when the Metal conv symbol ran, else an
    ``reference`` host fallback (never a fake native success) — and the launched
    result is numerically correct. Flips the prior 'honestly unimplemented' pin."""
    import tessera
    @tessera.jit(target="apple_gpu")
    def f(x, w):
        return tessera.ops.conv2d(x, w, stride=1, padding=0, layout="nhwc")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 8, 8, 3)).astype(np.float32)
    w = rng.standard_normal((3, 3, 3, 4)).astype(np.float32)
    res = tessera.runtime.launch(f.runtime_artifact(), (x, w))
    assert res["ok"] and res["runtime_status"] == "success"
    # Honest provenance gate: native only when Metal genuinely ran the conv.
    assert res["execution_kind"] in ("native_gpu", "reference")
    if DeviceTensor.is_metal():
        assert res["execution_kind"] == "native_gpu"
    # The launched result is numerically correct.
    out = np.asarray(res["output"])
    ref = np.zeros((1, 6, 6, 4), np.float32)
    for i in range(6):
        for j in range(6):
            patch = x[0, i:i + 3, j:j + 3, :]
            for co in range(4):
                ref[0, i, j, co] = np.sum(patch * w[:, :, :, co])
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


def test_conv2d_jit_launch_never_fakes_native_on_fallback(monkeypatch):
    """If the Metal conv symbol is unavailable, launch must report a host
    ``reference`` (still correct), never ``native_gpu`` — the rung-7 provenance
    gate. Forces the dispatcher to miss Metal and checks the demotion."""
    import tessera
    import tessera.runtime as R
    monkeypatch.setattr(R, "_apple_gpu_dispatch_conv2d", lambda a, k, np: None)
    monkeypatch.setattr(R, "_APPLE_GPU_LANE_HANDLERS", None)  # force lane rebuild
    @tessera.jit(target="apple_gpu")
    def f(x, w):
        return tessera.ops.conv2d(x, w, stride=1, padding=0, layout="nhwc")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 6, 6, 2)).astype(np.float32)
    w = rng.standard_normal((3, 3, 2, 5)).astype(np.float32)
    res = tessera.runtime.launch(f.runtime_artifact(), (x, w))
    assert res["execution_kind"] == "reference"   # demoted, not fake native
    assert res["runtime_status"] == "success"
    out = np.asarray(res["output"])
    ref = np.zeros((1, 4, 4, 5), np.float32)
    for i in range(4):
        for j in range(4):
            patch = x[0, i:i + 3, j:j + 3, :]
            for co in range(5):
                ref[0, i, j, co] = np.sum(patch * w[:, :, :, co])
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)
    R._APPLE_GPU_LANE_HANDLERS = None  # restore (monkeypatch undoes the attr)


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


# ── kv_cache_read execution-proven via the Evaluator verdict (#17) ───────────


def test_kv_cache_read_native_equivalence_verdict():
    """The stateful kv_cache_read accessor is 'execution-proven' via a dedicated
    Evaluator verdict (it does not flow the pure-tensor conformance matrix): on
    Metal it runs on metal_runtime AND matches the host source slice; off Metal
    the provenance gate keeps it inconclusive (never a false native rung)."""
    from tessera.compiler.evaluator import kv_cache_read_native_equivalence

    h = KVCacheHandle(num_heads=2, head_dim=4, max_seq=32, page_size=8)
    k = np.arange(12 * 2 * 4, dtype=np.float32).reshape(12, 2, 4)
    h.append(k, k + 50.0)
    verdict = kv_cache_read_native_equivalence(h, 3, 9)
    if DeviceTensor.is_metal():
        assert verdict.relation == "equivalent", verdict.detail
        assert verdict.max_abs_err == 0.0   # device-resident view == source slice
    else:
        assert verdict.relation == "inconclusive"   # provenance gate


def test_kv_cache_read_native_equivalence_single_token():
    from tessera.compiler.evaluator import kv_cache_read_native_equivalence

    h = KVCacheHandle(num_heads=1, head_dim=4, max_seq=16)
    h.append(np.arange(5 * 4, dtype=np.float32).reshape(5, 1, 4) + 1.0,
             np.arange(5 * 4, dtype=np.float32).reshape(5, 1, 4) + 2.0)
    verdict = kv_cache_read_native_equivalence(h, 2)   # single-token read
    assert verdict.relation in {"equivalent", "inconclusive"}
