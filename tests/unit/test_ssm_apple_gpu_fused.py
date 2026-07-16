"""Track-R (ReplaySSM) Phase 5 — fused single-dispatch SSM decode kernel.

``tessera_apple_gpu_ssm_replay_decode_f32`` reconstructs the output-only decode
step in one Metal dispatch, keeping the checkpoint state ``S0`` resident and
reading only the small replay inputs (the state-traffic-halving kernel).  On
Darwin it runs the MSL kernel; on non-Metal hosts the C ABI symbol runs its own
host reference — either way these tests validate numerically against the eager
``selective_ssm`` reference (f32 precision) and the numpy reference handle.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera
from tessera import runtime as rt
from tessera.cache import SSMStateHandle


def _decode(handle, dt, x, Bp, Cp, gate=None):
    B, S, D = x.shape
    y = np.zeros((B, S, D))
    for t in range(S):
        gt = None if gate is None else gate[:, t, :]
        y[:, t, :] = handle.step(dt[:, t, :], x[:, t, :], Bp[:, t, :], Cp[:, t, :], gate_t=gt)
    return y


def _inputs(rng, B, S, D, N):
    x = rng.standard_normal((B, S, D))
    a = -np.abs(rng.standard_normal(D))
    Bp = rng.standard_normal((B, S, N))
    Cp = rng.standard_normal((B, S, N))
    dt = np.abs(rng.standard_normal((B, S, D))) * 0.5
    return x, a, Bp, Cp, dt


def test_fused_factory_wires_decode_fn():
    h = rt.apple_gpu_fused_ssm_state_handle(1, 4, 3, -np.ones(4), capacity=8)
    assert isinstance(h, SSMStateHandle)
    assert h.backend == "apple_gpu_fused"
    assert h.decode_fn is not None


@pytest.mark.hardware_apple_gpu
def test_decode_symbol_resolves():
    """The C ABI symbol must be present in the (on-demand-compiled) runtime."""
    sym = rt._apple_gpu_ssm_replay_decode_f32()
    assert callable(sym)


@pytest.mark.parametrize("B,S,D,N,cap,spec", [
    (1, 20, 4, 3, 64, 0),
    (2, 24, 8, 4, 6, 2),
    (3, 18, 6, 5, 9, 1),
])
def test_fused_decode_matches_eager_f32(B, S, D, N, cap, spec):
    rng = np.random.default_rng(B * 17 + S + D + N)
    x, a, Bp, Cp, dt = _inputs(rng, B, S, D, N)
    y_eager = np.asarray(tessera.ops.selective_ssm(x, a, Bp, Cp, dt))
    h = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=cap, spec_window=spec)
    y = _decode(h, dt, x, Bp, Cp)
    assert np.max(np.abs(y - y_eager)) < 5e-4


def test_fused_matches_reference_handle():
    rng = np.random.default_rng(2)
    B, S, D, N = 2, 16, 6, 4
    x, a, Bp, Cp, dt = _inputs(rng, B, S, D, N)
    ref = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a, capacity=5)
    fused = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=5)
    y_ref = _decode(ref, dt, x, Bp, Cp)
    y_fused = _decode(fused, dt, x, Bp, Cp)
    assert np.max(np.abs(y_ref - y_fused)) < 5e-4


def test_fused_applies_output_gate():
    rng = np.random.default_rng(8)
    B, S, D, N = 1, 12, 5, 3
    x, a, Bp, Cp, dt = _inputs(rng, B, S, D, N)
    gate = np.abs(rng.standard_normal((B, S, D)))
    y_eager = np.asarray(tessera.ops.selective_ssm(x, a, Bp, Cp, dt, gate=gate))
    h = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=4, spec_window=1)
    y = _decode(h, dt, x, Bp, Cp, gate=gate)
    assert np.max(np.abs(y - y_eager)) < 5e-4


def test_fused_speculative_rollback_still_exact():
    """The fused decode path must compose with speculative rollback (Phase 3):
    appending drafts then rolling back leaves identical continuations."""
    from tessera.speculative import advance_ssm
    rng = np.random.default_rng(4)
    B, S, D, N = 1, 8, 4, 3
    x, a, Bp, Cp, dt = _inputs(rng, B, S, D, N)
    base = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=64, spec_window=4)
    for t in range(3):
        base.step(dt[:, t, :], x[:, t, :], Bp[:, t, :], Cp[:, t, :])
    spec = base.clone()
    for _ in range(3):  # 3 drafts, all rejected
        spec.step(rng.standard_normal((B, D)), rng.standard_normal((B, D)),
                  rng.standard_normal((B, N)), rng.standard_normal((B, N)))
    advance_ssm(spec, 0, num_drafts=3)
    for t in range(3, S):
        yb = base.step(dt[:, t, :], x[:, t, :], Bp[:, t, :], Cp[:, t, :])
        ys = spec.step(dt[:, t, :], x[:, t, :], Bp[:, t, :], Cp[:, t, :])
        assert np.max(np.abs(yb - ys)) < 1e-5
