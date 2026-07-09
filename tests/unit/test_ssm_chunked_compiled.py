"""Chunked-parallel (SSD) selective_ssm lane — x86 AVX-512 + ROCm gfx1151.

For scalar-state A (shape (D,), the common Mamba-2 config) the sequential scan is
reformulated as the chunked-parallel SSD algorithm (`_mamba_ssd`), whose three
batched contractions run on the device GEMM (AVX-512 f32 / WMMA f16) and the
per-chunk state recurrence on host — the standard Mamba-2 decomposition. This
validates that the f32 scalar-A forward now routes through the chunked form and
matches the sequential `tessera.ops.selective_ssm` reference; (D, N) A stays on
the sequential per-(b,d) kernel.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera import runtime as rt


def _x86_or_skip():
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")


def _rocm_or_skip():
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")


def _inputs(rng, b, s, d, n):
    x = rng.standard_normal((b, s, d)).astype(np.float32)
    A = (-np.abs(rng.standard_normal(d))).astype(np.float32)     # scalar-state
    B = (rng.standard_normal((b, s, n)) * 0.3).astype(np.float32)
    C = (rng.standard_normal((b, s, n)) * 0.3).astype(np.float32)
    delta = np.abs(rng.standard_normal((b, s, d)) * 0.1).astype(np.float32)
    return x, A, B, C, delta


@pytest.mark.parametrize("s", [40, 64])   # multi-chunk (chunk_size=128 default)
def test_x86_chunked_matches_sequential(s):
    _x86_or_skip()
    rng = np.random.default_rng(s)
    x, A, B, C, delta = _inputs(rng, 2, s, 4, 8)
    ref = np.asarray(ts.ops.selective_ssm(x, A, B, C, delta))
    got = np.asarray(rt._x86_selective_ssm(x, A, B, C, delta, None, None, np))
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-4)


def test_x86_chunked_gate_state():
    _x86_or_skip()
    rng = np.random.default_rng(5)
    b, s, d, n = 2, 48, 4, 8
    x, A, B, C, delta = _inputs(rng, b, s, d, n)
    gate = rng.standard_normal((b, s, d)).astype(np.float32)
    state = rng.standard_normal((b, d, n)).astype(np.float32)
    ref = np.asarray(ts.ops.selective_ssm(x, A, B, C, delta, gate=gate,
                                          state=state))
    got = np.asarray(rt._x86_selective_ssm(x, A, B, C, delta, gate, state, np))
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-4)


@pytest.mark.parametrize("s", [40, 64])
def test_rocm_chunked_matches_sequential(s):
    _rocm_or_skip()
    rng = np.random.default_rng(100 + s)
    x, A, B, C, delta = _inputs(rng, 2, s, 4, 8)
    ref = np.asarray(ts.ops.selective_ssm(x, A, B, C, delta))
    got = np.asarray(rt._rocm_selective_ssm(x, A, B, C, delta, None, None, np))
    # WMMA f16 bmm — matches the f32 sequential reference to ~1e-4.
    np.testing.assert_allclose(got, ref, rtol=0, atol=3e-3)


def test_chunked_only_for_scalar_a(monkeypatch):
    # (D, N) A must NOT take the chunked path (it has no clean matmul form) — it
    # stays on the sequential per-(b,d) kernel.
    _x86_or_skip()
    calls = []
    orig = rt._x86_selective_ssm_chunked
    monkeypatch.setattr(rt, "_x86_selective_ssm_chunked",
                        lambda *a, **k: calls.append(1) or orig(*a, **k))
    rng = np.random.default_rng(7)
    b, s, d, n = 2, 40, 4, 8
    x = rng.standard_normal((b, s, d)).astype(np.float32)
    A2d = (-np.abs(rng.standard_normal((d, n)))).astype(np.float32)   # (D, N)
    B = rng.standard_normal((b, s, n)).astype(np.float32)
    C = rng.standard_normal((b, s, n)).astype(np.float32)
    delta = np.abs(rng.standard_normal((b, s, d)) * 0.1).astype(np.float32)
    rt._x86_selective_ssm(x, A2d, B, C, delta, None, None, np)
    assert not calls, "(D, N) A must stay on the sequential kernel"
