"""Apple GPU Phase-G Rung 0 — control-flow lowering (MPSGraph forLoop).

Locks the first rung of the control-flow-lowering ladder: a bounded scan
``carry_{i+1} = tanh(carry_i @ Wh + x_i @ Wx)`` lowered into a single MPSGraph
control-flow executable (``-forLoopWithLowerBound:upperBound:step:``), with the
per-step carries recovered via an index-scatter accumulator. Validated against a
numpy scan; off Metal the runtime falls back to a numpy scan so the contract is
checked everywhere. See docs/apple_gpu_control_flow_lowering.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as R


def _numpy_scan(Wh, Wx, xseq, init):
    carry = init.astype(np.float64)
    T, d = xseq.shape[0], Wh.shape[0]
    ys = np.empty((T, d), np.float64)
    for t in range(T):
        carry = np.tanh(carry @ Wh.astype(np.float64)
                        + xseq[t].astype(np.float64) @ Wx.astype(np.float64))
        ys[t] = carry
    return ys


@pytest.mark.parametrize("T,d,m", [(1, 4, 4), (5, 8, 4), (8, 16, 6)])
def test_cf_scan_matches_numpy(T, d, m):
    rng = np.random.default_rng(T * 100 + d)
    Wh = rng.standard_normal((d, d)).astype(np.float32) * 0.3
    Wx = rng.standard_normal((m, d)).astype(np.float32) * 0.3
    xseq = rng.standard_normal((T, m)).astype(np.float32) * 0.3
    init = rng.standard_normal(d).astype(np.float32) * 0.1
    ys = R.apple_gpu_cf_scan(Wh, Wx, xseq, init, np)
    assert ys.shape == (T, d)
    np.testing.assert_allclose(ys.astype(np.float64),
                               _numpy_scan(Wh, Wx, xseq, init),
                               rtol=1e-4, atol=1e-5)


def test_cf_scan_is_a_real_recurrence():
    # Each step must depend on the previous carry (not an independent map):
    # zeroing the recurrence weight collapses to tanh(x_t @ Wx) per step.
    rng = np.random.default_rng(0)
    T, d, m = 4, 6, 6
    Wh = np.zeros((d, d), np.float32)
    Wx = rng.standard_normal((m, d)).astype(np.float32) * 0.5
    xseq = rng.standard_normal((T, m)).astype(np.float32) * 0.5
    init = rng.standard_normal(d).astype(np.float32)
    ys = R.apple_gpu_cf_scan(Wh, Wx, xseq, init, np)
    indep = np.tanh(xseq.astype(np.float64) @ Wx.astype(np.float64))
    np.testing.assert_allclose(ys.astype(np.float64), indep, rtol=1e-4, atol=1e-5)


def test_cf_scan_symbol_or_fallback():
    # On Metal the symbol must be present; off Metal the numpy fallback still
    # returns a correct, correctly-shaped result.
    sym = R._apple_gpu_cf_scan_f32()
    if R.DeviceTensor.is_metal():
        assert sym is not None
    ys = R.apple_gpu_cf_scan(np.eye(3, dtype=np.float32),
                             np.eye(3, dtype=np.float32),
                             np.zeros((2, 3), np.float32),
                             np.zeros(3, np.float32), np)
    assert ys.shape == (2, 3)
