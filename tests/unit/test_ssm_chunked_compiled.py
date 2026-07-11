"""Chunked-parallel (SSD) selective_ssm lane — x86 AVX-512 + ROCm gfx1151.

For scalar-state A (shape (D,), the common Mamba-2 config) the sequential scan is
reformulated as the chunked-parallel SSD algorithm (`_mamba_ssd`), whose three
batched contractions run on the native f32 GEMM device kernel and the per-chunk
state recurrence on host — the standard Mamba-2 decomposition.

  * x86: scalar-A f32 routes through the chunked form (its batched GEMM is a
    host-side BLAS loop with no device round-trip → a win). (D, N) A + f16/bf16
    stay on the sequential per-(b,d) kernel.
  * ROCm: the chunked form now HAS a numerically-safe f32 substrate on the #356
    f32 GEMM device kernel, verified correct here — but it is a measured 4–100×
    REGRESSION vs the single-launch sequential scan (per-call GEMM H2D/D2H
    overhead), so it is a CORRECTNESS REFERENCE RUNG and the sequential scan stays
    the default. A single-launch batched f32 GEMM is the prerequisite for it to
    win (a follow-up).
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


@pytest.mark.parametrize("s", [40, 64])   # multi-chunk (chunk_size=128 default)
def test_rocm_chunked_reference_rung_matches_sequential(s):
    # The ROCm chunked SSD helper (native f32 GEMM) is a correctness-verified
    # REFERENCE RUNG — call it DIRECTLY (it is NOT the default; the sequential scan
    # is faster on gfx1151, see the helper docstring). It must match the reference.
    _rocm_or_skip()
    rng = np.random.default_rng(s + 1)
    x, A, B, C, delta = _inputs(rng, 2, s, 4, 8)
    ref = np.asarray(ts.ops.selective_ssm(x, A, B, C, delta))
    got = np.asarray(
        rt._rocm_selective_ssm_chunked(x, A, B, C, delta, None, None, np))
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-4)


def test_rocm_chunked_reference_rung_gate_state():
    _rocm_or_skip()
    rng = np.random.default_rng(9)
    b, s, d, n = 2, 48, 4, 8
    x, A, B, C, delta = _inputs(rng, b, s, d, n)
    gate = rng.standard_normal((b, s, d)).astype(np.float32)
    state = rng.standard_normal((b, d, n)).astype(np.float32)
    ref = np.asarray(ts.ops.selective_ssm(x, A, B, C, delta, gate=gate,
                                          state=state))
    got = np.asarray(
        rt._rocm_selective_ssm_chunked(x, A, B, C, delta, gate, state, np))
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-4)


def test_rocm_default_is_sequential_not_chunked():
    # The production ROCm path stays the single-launch sequential scan for ALL
    # dtypes — it is 4–100× faster than the per-call-GEMM chunked rung on gfx1151.
    # Assert the default never routes through the chunked helper, and stays finite
    # + correct even on inputs beyond the fp16 range.
    _rocm_or_skip()
    fp16_max = 65504.0
    rng = np.random.default_rng(101)
    b, s, d, n = 2, 40, 4, 64
    x = rng.standard_normal((b, s, d)).astype(np.float32)
    A = (-np.abs(rng.standard_normal(d))).astype(np.float32)          # scalar-A
    B = rng.standard_normal((b, s, n)).astype(np.float32)
    C = rng.standard_normal((b, s, n)).astype(np.float32)
    delta = np.abs(rng.standard_normal((b, s, d)) * 0.001).astype(np.float32)
    x[0, 0, 0] = 2.0 * fp16_max
    B[0, 0, 0] = 3.0 * fp16_max
    C[0, 0, 0] = 1.5 * fp16_max
    assert x.max() > fp16_max and B.max() > fp16_max and C.max() > fp16_max

    calls = []
    orig = rt._rocm_selective_ssm_chunked
    rt._rocm_selective_ssm_chunked = lambda *a, **k: (
        calls.append(1), orig(*a, **k))[1]
    try:
        got = np.asarray(rt._rocm_selective_ssm(x, A, B, C, delta, None, None, np))
    finally:
        rt._rocm_selective_ssm_chunked = orig
    assert not calls, "the default ROCm path must NOT route through the chunked rung"
    assert np.isfinite(got).all(), "sequential f32 scan must stay finite"
    ref = np.asarray(ts.ops.selective_ssm(x, A, B, C, delta))
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-4)


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
