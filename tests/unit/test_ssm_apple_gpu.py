"""Track-R (ReplaySSM) Phase 4 — scalar-A Mamba-2 decode on the Apple GPU
bmm lane (correct-but-composed).

The reconstruction is unchanged; only its batched contractions (projection /
gram / state update) move to Metal via the ``matmul3d`` hook
(``runtime.apple_gpu_ssm_state_handle``).  On Darwin the bmm runs on the GPU
(MPSGraph); elsewhere it falls back to numpy — either way the math is the same,
so these tests are portable.  The Apple GPU path matches the eager reference at
f32 precision; the numpy-backed path matches at f64.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera
from tessera import runtime as rt
from tessera.cache import SSMStateHandle


def _decode(handle, dt, x, Bp, Cp):
    B, S, D = x.shape
    y = np.zeros((B, S, D))
    for t in range(S):
        y[:, t, :] = handle.step(dt[:, t, :], x[:, t, :], Bp[:, t, :], Cp[:, t, :])
    return y


def _inputs(rng, B, S, D, N):
    x = rng.standard_normal((B, S, D))
    a = -np.abs(rng.standard_normal(D))            # scalar-state A
    Bp = rng.standard_normal((B, S, N))
    Cp = rng.standard_normal((B, S, N))
    dt = np.abs(rng.standard_normal((B, S, D))) * 0.5
    return x, a, Bp, Cp, dt


def test_apple_gpu_factory_sets_backend_and_bmm():
    h = rt.apple_gpu_ssm_state_handle(1, 4, 3, -np.ones(4), capacity=8, spec_window=2)
    assert isinstance(h, SSMStateHandle)
    assert h.backend == "apple_gpu"
    assert h.matmul3d is not None


@pytest.mark.parametrize("B,S,D,N,cap,spec", [
    (1, 20, 4, 3, 64, 0),
    (2, 24, 8, 4, 6, 2),
    (3, 18, 6, 5, 9, 1),
])
def test_apple_gpu_decode_matches_eager_f32(B, S, D, N, cap, spec):
    rng = np.random.default_rng(B * 31 + S + D + N)
    x, a, Bp, Cp, dt = _inputs(rng, B, S, D, N)
    y_eager = np.asarray(tessera.ops.selective_ssm(x, a, Bp, Cp, dt))
    h = rt.apple_gpu_ssm_state_handle(B, D, N, a, capacity=cap, spec_window=spec)
    y_gpu = _decode(h, dt, x, Bp, Cp)
    assert np.max(np.abs(y_gpu - y_eager)) < 5e-4   # f32 GPU bmm tolerance


def test_scalar_bmm_path_with_numpy_is_exact():
    """The bmm reformulation with the numpy backend (default) must reproduce the
    eager reference at f64 — proving the contraction rewrite is exact, not just
    f32-close."""
    rng = np.random.default_rng(7)
    B, S, D, N = 2, 22, 7, 4
    x, a, Bp, Cp, dt = _inputs(rng, B, S, D, N)
    y_eager = np.asarray(tessera.ops.selective_ssm(x, a, Bp, Cp, dt))
    h = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a, capacity=5)  # matmul3d=None → np.matmul
    y_ref = _decode(h, dt, x, Bp, Cp)
    assert np.max(np.abs(y_ref - y_eager)) < 1e-9


def test_gpu_and_reference_handles_agree():
    rng = np.random.default_rng(2)
    B, S, D, N = 2, 16, 6, 4
    x, a, Bp, Cp, dt = _inputs(rng, B, S, D, N)
    ref = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a, capacity=5)
    gpu = rt.apple_gpu_ssm_state_handle(B, D, N, a, capacity=5)
    y_ref = _decode(ref, dt, x, Bp, Cp)
    y_gpu = _decode(gpu, dt, x, Bp, Cp)
    assert np.max(np.abs(y_ref - y_gpu)) < 5e-4


def test_materialize_state_bmm_matches_reference():
    rng = np.random.default_rng(4)
    B, S, D, N = 2, 14, 5, 3
    x, a, Bp, Cp, dt = _inputs(rng, B, S, D, N)
    ref = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a, capacity=6)
    gpu = rt.apple_gpu_ssm_state_handle(B, D, N, a, capacity=6)
    _decode(ref, dt, x, Bp, Cp)
    _decode(gpu, dt, x, Bp, Cp)
    assert np.max(np.abs(ref.materialize_state() - gpu.materialize_state())) < 5e-4


def test_general_a_ignores_bmm_backend():
    """A rank-2 (D, N) A is not bmm-separable, so the handle stays on the dense
    einsum path even when a bmm backend is wired — still exact vs eager."""
    rng = np.random.default_rng(5)
    B, S, D, N = 1, 16, 4, 3
    x = rng.standard_normal((B, S, D))
    a = -np.abs(rng.standard_normal((D, N)))       # full (D, N) A
    Bp = rng.standard_normal((B, S, N))
    Cp = rng.standard_normal((B, S, N))
    dt = np.abs(rng.standard_normal((B, S, D))) * 0.5
    y_eager = np.asarray(tessera.ops.selective_ssm(x, a, Bp, Cp, dt))
    h = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a, capacity=4,
                       matmul3d=rt.apple_gpu_bmm_callable(), backend="apple_gpu")
    assert not h._scalar_a
    y = _decode(h, dt, x, Bp, Cp)
    assert np.max(np.abs(y - y_eager)) < 1e-9      # dense path → f64 exact
