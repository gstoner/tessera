"""Apple GPU R2 — encoded elementwise ops complete the command-buffer block.

The R2 encode session previously covered bmm / rowop (rmsnorm/softmax) / gumbel.
These tests lock the elementwise additions (relu / silu / add / mul / silu_mul)
that let a *single* command buffer express a full transformer/MLP block —
residual adds, SwiGLU, ReLU heads, additive attention masks — instead of just
the MLA decode chain. Everything is validated against a float64 numpy reference;
off Darwin the encode session's reference path keeps the math correct.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as R
from tessera.runtime import AppleGPUEncodeSession, DeviceTensor


def _require():
    if not AppleGPUEncodeSession().available:
        pytest.skip("apple gpu encode session unavailable")


def _silu(x):
    return x / (1.0 + np.exp(-x))


def test_encode_elementwise_matches_numpy():
    _require()
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 5)).astype(np.float32)
    b = rng.standard_normal((4, 5)).astype(np.float32)
    da, db = DeviceTensor.from_numpy(a), DeviceTensor.from_numpy(b)
    with AppleGPUEncodeSession() as s:
        relu = s.relu(da)
        silu = s.silu(da)
        add = s.add(da, db)
        mul = s.mul(da, db)
        sm = s.silu_mul(da, db)
    np.testing.assert_allclose(relu.numpy(), np.maximum(a, 0.0), atol=1e-6)
    np.testing.assert_allclose(silu.numpy(), _silu(a), atol=1e-5)
    np.testing.assert_allclose(add.numpy(), a + b, atol=1e-6)
    np.testing.assert_allclose(mul.numpy(), a * b, atol=1e-6)
    np.testing.assert_allclose(sm.numpy(), _silu(a) * b, atol=1e-5)
    for t in (da, db):
        t.free()


def test_encode_shape_agnostic_3d():
    _require()
    rng = np.random.default_rng(1)
    a = rng.standard_normal((2, 3, 4)).astype(np.float32)
    b = rng.standard_normal((2, 3, 4)).astype(np.float32)
    da, db = DeviceTensor.from_numpy(a), DeviceTensor.from_numpy(b)
    with AppleGPUEncodeSession() as s:
        out = s.add(s.relu(da), db)
    np.testing.assert_allclose(out.numpy(), np.maximum(a, 0.0) + b, atol=1e-6)
    da.free(); db.free()


def test_resident_mlp_block_one_command_buffer():
    """A pre-norm residual block — RMSNorm -> value projection -> residual add
    -> RMSNorm -> SwiGLU -> residual add — encoded into ONE command buffer via
    the session (rmsnorm + bmm + add + silu_mul chained device-resident),
    matched against a float64 numpy reference. Exercises exactly the op mix a
    Gumiho serial-head block needs."""
    _require()
    rng = np.random.default_rng(2)
    T, d, ffn = 4, 16, 16
    x = rng.standard_normal((T, d)).astype(np.float32)
    g1 = rng.standard_normal(d).astype(np.float32)
    g2 = rng.standard_normal(d).astype(np.float32)
    wv = rng.standard_normal((d, d)).astype(np.float32) * 0.2
    wo = rng.standard_normal((d, d)).astype(np.float32) * 0.2
    wg = rng.standard_normal((d, ffn)).astype(np.float32) * 0.2
    wu = rng.standard_normal((d, ffn)).astype(np.float32) * 0.2
    wd = rng.standard_normal((ffn, d)).astype(np.float32) * 0.2

    def rms(v, gm):
        v = v.astype(np.float64)
        return v / np.sqrt((v * v).mean(-1, keepdims=True) + 1e-6) * gm

    n1 = rms(x, g1)
    h = x.astype(np.float64) + (n1 @ wv) @ wo
    n2 = rms(h, g2)
    ref = h + (_silu(n2 @ wg) * (n2 @ wu)) @ wd

    dx = DeviceTensor.from_numpy(x)
    dg1, dg2 = DeviceTensor.from_numpy(g1), DeviceTensor.from_numpy(g2)
    dwv, dwo = DeviceTensor.from_numpy(wv[None]), DeviceTensor.from_numpy(wo[None])
    dwg, dwu = DeviceTensor.from_numpy(wg[None]), DeviceTensor.from_numpy(wu[None])
    dwd = DeviceTensor.from_numpy(wd[None])
    with AppleGPUEncodeSession() as s:
        n1d = s.rmsnorm(dx, dg1, 1e-6)
        v = s.bmm(n1d.reshape_view(1, T, d), dwv)
        attn_out = s.bmm(v.reshape_view(1, T, d), dwo)
        h_d = s.add(dx, attn_out.reshape_view(T, d))
        n2d = s.rmsnorm(h_d, dg2, 1e-6)
        gate = s.bmm(n2d.reshape_view(1, T, d), dwg)
        up = s.bmm(n2d.reshape_view(1, T, d), dwu)
        act = s.silu_mul(gate.reshape_view(T, ffn), up.reshape_view(T, ffn))
        down = s.bmm(act.reshape_view(1, T, ffn), dwd)
        out = s.add(h_d, down.reshape_view(T, d))
    np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-4)
    for t in (dx, dg1, dg2, dwv, dwo, dwg, dwu, dwd):
        t.free()
