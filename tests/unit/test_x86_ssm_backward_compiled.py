"""x86 AVX-512 backward pass for Mamba2 selective_ssm.

`tessera_x86_selective_ssm_bwd_f32` is the reverse-mode adjoint of the AVX-512
selective-scan forward — a sequential reverse scan (t = S-1 → 0 per (b,d),
vectorized over the state dim N) that recomputes the forward trajectory then
accumulates (dx, dA, dB, dC, ddelta). This validates it against the numpy VJP
(`autodiff.vjp.vjp_selective_ssm`) — the exact chain-rule reference — across
2-d/1-d A, gate, state, and the SIMD path + scalar tail. Skip-clean: x86 lib not
built.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.autodiff.vjp import vjp_selective_ssm


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _inputs(rng, b, s, d, n, a_1d):
    x = rng.standard_normal((b, s, d)).astype(np.float32)
    A = (-np.abs(rng.standard_normal((d,) if a_1d else (d, n)))).astype(np.float32)
    B = rng.standard_normal((b, s, n)).astype(np.float32)
    C = rng.standard_normal((b, s, n)).astype(np.float32)
    delta = np.abs(rng.standard_normal((b, s, d)) * 0.1).astype(np.float32)
    dout = rng.standard_normal((b, s, d)).astype(np.float32)
    return x, A, B, C, delta, dout


@pytest.mark.parametrize("n", [16, 10])           # SIMD path + scalar tail
@pytest.mark.parametrize("a_1d", [False, True])
@pytest.mark.parametrize("gate_on,state_on", [(False, False), (True, True)])
def test_x86_ssm_backward_matches_vjp(n, a_1d, gate_on, state_on):
    rt = _rt_or_skip()
    rng = np.random.default_rng(n * 10 + a_1d + (gate_on << 2) + (state_on << 3))
    b, s, d = 2, 7, 4
    x, A, B, C, delta, dout = _inputs(rng, b, s, d, n, a_1d)
    gate = rng.standard_normal((b, s, d)).astype(np.float32) if gate_on else None
    state = rng.standard_normal((b, d, n)).astype(np.float32) if state_on else None

    ref = vjp_selective_ssm(dout, x, A, B, C, delta, gate=gate, state=state)
    got = rt._x86_selective_ssm_bwd(x, A, B, C, delta, dout, gate, state, np)
    names = ("dx", "dA", "dB", "dC", "ddelta")
    for name, g, r in zip(names, got, ref):
        np.testing.assert_allclose(g, r, rtol=0, atol=2e-3,
                                   err_msg=f"{name} mismatch")
