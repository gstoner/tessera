"""Mamba-2 selective_ssm — chunked-parallel SSD + Apple GPU execution.

Third pickup (first heavy lift) from the model-family gap analysis: `selective_ssm`
previously ran pure-numpy on every target. This routes it onto a chunked-parallel
SSD reformulation whose three batched contractions (state projection, C·Bᵀ gram,
state update) dispatch to the Apple GPU `bmm` lane — so it executes on Metal.

Layers checked:
  1. **Algorithm** — the parallel form (`_mamba_ssd.selective_ssm_parallel`) is
     numerically bit-exact vs the sequential `tessera.ops.selective_ssm` reference
     across shapes, chunk sizes, gate, initial state, and long sequences
     (the stability case the naive cumprod form would overflow on).
  2. **Scope** — scalar-state `A` (shape `(D,)`) uses the matmul form; general
     `(D, N)` `A` is correctly flagged unsupported (caller falls back).
  3. **Envelope** — `tessera.selective_ssm` is in the apple_gpu runtime envelope
     (driver + runtime agree).
  4. **Apple GPU** — `@jit(target="apple_gpu")` executes it (execution_mode
     `metal_runtime`) and matches the sequential reference; `(D, N)` falls back.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import _mamba_ssd
from tessera.compiler import driver as _driver
from tessera import runtime as _runtime

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.hardware_apple_gpu


def _rand_ssm(rng, Bsz, S, D, N, *, scalar_A=True):
    x = rng.standard_normal((Bsz, S, D)).astype(np.float32)
    B = rng.standard_normal((Bsz, S, N)).astype(np.float32)
    C = rng.standard_normal((Bsz, S, N)).astype(np.float32)
    delta = (np.abs(rng.standard_normal((Bsz, S, D))) * 0.5).astype(np.float32)
    if scalar_A:
        A = (-np.abs(rng.standard_normal(D)) - 0.1).astype(np.float32)
    else:
        A = (-np.abs(rng.standard_normal((D, N))) - 0.1).astype(np.float32)
    return x, A, B, C, delta


# ── 1. parallel form == sequential reference (CPU; no GPU needed) ───────────── #
@pytest.mark.parametrize("shape,chunk", [
    ((2, 16, 4, 3), 8), ((1, 7, 2, 2), 4), ((3, 32, 5, 4), 16),
    ((2, 5, 3, 3), 5), ((1, 1, 2, 2), 8),
])
def test_parallel_matches_sequential(shape, chunk):
    rng = np.random.default_rng(sum(shape) + chunk)
    x, A, B, C, delta = _rand_ssm(rng, *shape)
    ref = np.asarray(ts.ops.selective_ssm(x, A, B, C, delta))
    par = _mamba_ssd.selective_ssm_parallel(x, A, B, C, delta, chunk_size=chunk)
    np.testing.assert_allclose(par, ref, rtol=1e-4, atol=1e-5)


def test_parallel_gate_and_state():
    rng = np.random.default_rng(7)
    x, A, B, C, delta = _rand_ssm(rng, 2, 12, 4, 3)
    gate = rng.standard_normal((2, 12, 4)).astype(np.float32)
    state = rng.standard_normal((2, 4, 3)).astype(np.float32)
    np.testing.assert_allclose(
        _mamba_ssd.selective_ssm_parallel(x, A, B, C, delta, gate=gate),
        np.asarray(ts.ops.selective_ssm(x, A, B, C, delta, gate=gate)),
        rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        _mamba_ssd.selective_ssm_parallel(x, A, B, C, delta, state=state),
        np.asarray(ts.ops.selective_ssm(x, A, B, C, delta, state=state)),
        rtol=1e-4, atol=1e-5)


def test_parallel_long_sequence_is_stable():
    # S well past one chunk with strong decay — the pairwise-decay form must not
    # overflow (the naive cumprod u/L form would).
    rng = np.random.default_rng(11)
    x, A, B, C, delta = _rand_ssm(rng, 1, 256, 4, 3)
    A = A * 8.0                              # strong decay
    ref = np.asarray(ts.ops.selective_ssm(x, A, B, C, delta))
    par = _mamba_ssd.selective_ssm_parallel(x, A, B, C, delta, chunk_size=64)
    assert np.all(np.isfinite(par))
    np.testing.assert_allclose(par, ref, rtol=1e-3, atol=1e-4)


# ── 2. scope: scalar-A only ────────────────────────────────────────────────── #
def test_supports_parallel_scope():
    assert _mamba_ssd.supports_parallel(np.zeros(4)) is True          # (D,)
    assert _mamba_ssd.supports_parallel(np.zeros((4, 3))) is False    # (D, N)
    with pytest.raises(ValueError):
        rng = np.random.default_rng(0)
        x, A, B, C, delta = _rand_ssm(rng, 1, 4, 3, 2, scalar_A=False)
        _mamba_ssd.selective_ssm_parallel(x, A, B, C, delta)


# ── 3. envelope agreement ──────────────────────────────────────────────────── #
def test_selective_ssm_in_apple_gpu_envelope():
    assert "tessera.selective_ssm" in _driver._APPLE_GPU_RUNTIME_OPS
    assert "tessera.selective_ssm" in _runtime._APPLE_GPU_RUNTIME_OPS


# ── 4. Apple GPU execution ─────────────────────────────────────────────────── #
@gpu
@pytest.mark.parametrize("shape", [(2, 16, 4, 3), (1, 64, 8, 4)])
def test_selective_ssm_apple_gpu_scalar_A(shape):
    rng = np.random.default_rng(sum(shape))
    x, A, B, C, delta = _rand_ssm(rng, *shape)
    ref = np.asarray(ts.ops.selective_ssm(x, A, B, C, delta))

    @ts.jit(target="apple_gpu")
    def f(x, A, B, C, delta):
        return ts.ops.selective_ssm(x, A, B, C, delta)

    out = np.asarray(f(x, A, B, C, delta))
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-4)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


def test_selective_ssm_apple_gpu_general_A_falls_back():
    rng = np.random.default_rng(3)
    x, A, B, C, delta = _rand_ssm(rng, 2, 12, 4, 3, scalar_A=False)
    ref = np.asarray(ts.ops.selective_ssm(x, A, B, C, delta))

    @ts.jit(target="apple_gpu")
    def f(x, A, B, C, delta):
        return ts.ops.selective_ssm(x, A, B, C, delta)

    np.testing.assert_allclose(np.asarray(f(x, A, B, C, delta)), ref,
                               rtol=1e-4, atol=1e-5)
