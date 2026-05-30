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


def _np_generate(W, lm, h0, start, eos, maxs):
    h = h0.astype(np.float64)
    Wf, lmf = W.astype(np.float64), lm.astype(np.float64)
    out, last, step = [], start, 0
    while step < maxs and last != eos:
        h = np.tanh(h @ Wf)
        last = int(np.argmax(h @ lmf))
        out.append(last)
        step += 1
    return out, step


def test_cf_while_generate_runs_to_max_without_eos():
    # Rung 2: predicate-driven while — no EOS reachable, so it runs to max_steps.
    rng = np.random.default_rng(1)
    d, V = 8, 16
    W = rng.standard_normal((d, d)).astype(np.float32) * 0.5
    lm = rng.standard_normal((d, V)).astype(np.float32) * 0.5
    h0 = rng.standard_normal(d).astype(np.float32) * 0.3
    toks, n = R.apple_gpu_cf_while_generate(W, lm, h0, 0, -1, 6, d, V, np)
    ref, rn = _np_generate(W, lm, h0, 0, -1, 6)
    assert n == rn == 6
    assert toks == ref


def test_cf_while_generate_early_stops_on_eos():
    # The variable-trip-count win: setting EOS to a token the loop will emit
    # must terminate the loop early (data-dependent trip count).
    rng = np.random.default_rng(1)
    d, V = 8, 16
    W = rng.standard_normal((d, d)).astype(np.float32) * 0.5
    lm = rng.standard_normal((d, V)).astype(np.float32) * 0.5
    h0 = rng.standard_normal(d).astype(np.float32) * 0.3
    full, _ = _np_generate(W, lm, h0, 0, -999, 6)
    eos = full[2]                                   # emitted at step index 2
    toks, n = R.apple_gpu_cf_while_generate(W, lm, h0, 0, eos, 6, d, V, np)
    ref, rn = _np_generate(W, lm, h0, 0, eos, 6)
    assert n == rn < 6                              # stopped before max_steps
    assert toks == ref
    assert toks[-1] == eos                          # eos is included


def _ref_spec_accept(draft, target):
    P, depth = draft.shape
    bp, bl, bb = 0, -1, 0
    for p in range(P):
        length = 0
        for i in range(depth):
            if int(draft[p, i]) == int(target[p, i]):
                length += 1
            else:
                break
        if length > bl:
            bl, bp = length, p
            bb = int(target[p, length])
    return bp, bl, bb, [int(draft[bp, i]) for i in range(bl)]


def test_rung3_spec_accept_multipath():
    """Phase-G Rung 3 via MSL: the dynamic speculative accept+select (variable-
    trip loops + cross-path argmax + bonus) as one MSL kernel — the frontier the
    MPSGraph route can't express. Longest-accepted-prefix path wins."""
    draft = np.array([[7, 3, 9, 1], [7, 8, 2, 4], [7, 3, 5, 6]], np.int32)
    target = np.array([[7, 3, 0, 0, 0], [7, 0, 0, 0, 0], [7, 3, 5, 0, 9]], np.int32)
    got = R.apple_gpu_msl_spec_accept(draft, target, np)
    assert got == _ref_spec_accept(draft, target)
    assert got[:3] == (2, 3, 0)            # path 2, len 3, bonus target[2][3]=0


def test_rung3_spec_accept_full_accept_bonus():
    # A fully-accepted path: bonus is the target token after the whole prefix.
    draft = np.array([[1, 2, 3]], np.int32)
    target = np.array([[1, 2, 3, 9]], np.int32)   # depth+1; last col = bonus
    bp, ln, bonus, acc = R.apple_gpu_msl_spec_accept(draft, target, np)
    assert (bp, ln, bonus, acc) == (0, 3, 9, [1, 2, 3])


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_rung3_spec_accept_matches_reference_random(seed):
    rng = np.random.default_rng(seed)
    P, depth, V = 8, 7, 16
    draft = rng.integers(0, V, size=(P, depth), dtype=np.int32)
    target = rng.integers(0, V, size=(P, depth + 1), dtype=np.int32)
    # make a couple of paths share a prefix with the target so accepts vary
    target[:, 0] = draft[:, 0]
    got = R.apple_gpu_msl_spec_accept(draft, target, np)
    assert got == _ref_spec_accept(draft, target)


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
