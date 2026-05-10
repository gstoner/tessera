"""Conv1d VJP/JVP coverage — S7 layer hardening.

Tests the (V/J)VPs across the full `conv1d` parameter matrix:
stride, padding, dilation, groups, with/without bias. Each case
compares against a central finite-difference reference at fp64.

The forward function `tessera.nn.functional.conv1d` casts to fp32
internally; tests use a relaxed tolerance (`atol=5e-3`) to absorb the
fp32 quantization noise from the FD reference, matching the convention
already used by the max_pool/avg_pool VJP tests.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import tessera.nn.functional as nn_functional
from tessera.autodiff.jvp import get_jvp
from tessera.autodiff.vjp import _conv1d_forward_fp64, get_vjp


# ── helpers ────────────────────────────────────────────────────────────────


def _numeric_grad(fn, x, eps=1e-4):
    g = np.zeros_like(x, dtype=np.float64)
    x = x.astype(np.float64).copy()
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = float(np.asarray(fn(x)).sum())
        x[idx] = orig - eps
        f_minus = float(np.asarray(fn(x)).sum())
        x[idx] = orig
        g[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return g


def _numeric_jvp(fn, x, dx, eps=1e-5):
    plus = np.asarray(fn(x + eps * dx), dtype=np.float64)
    minus = np.asarray(fn(x - eps * dx), dtype=np.float64)
    return (plus - minus) / (2 * eps)


# ── registration smoke tests ───────────────────────────────────────────────


def test_conv1d_vjp_and_jvp_are_registered():
    assert get_vjp("conv1d") is not None
    assert get_jvp("conv1d") is not None


# ── fp64 forward helper sanity ─────────────────────────────────────────────


def test_conv1d_fp64_helper_matches_nn_functional_within_fp32_tolerance():
    """The fp64 helper is the analytic ground truth — fp32 forward
    should match it up to fp32 quantization noise."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 8)).astype(np.float32)
    w = rng.normal(size=(4, 3, 3)).astype(np.float32)
    fp64 = _conv1d_forward_fp64(
        x.astype(np.float64), w.astype(np.float64),
        stride=1, padding=1, dilation=1, groups=1,
    )
    fp32 = nn_functional.conv1d(x, w, padding=1)
    np.testing.assert_allclose(fp64, fp32, atol=5e-5, rtol=5e-5)


# ── basic VJP — stride/padding/dilation/groups/bias matrix ─────────────────


@pytest.mark.parametrize("stride,padding,dilation,groups,bias", [
    (1, 0, 1, 1, False),
    (1, 1, 1, 1, False),
    (2, 0, 1, 1, False),
    (1, 2, 2, 1, False),
    (1, 1, 1, 2, False),
    (1, 1, 1, 1, True),
    (2, 1, 2, 2, True),  # full combo
])
def test_conv1d_vjp_matches_numeric(stride, padding, dilation, groups, bias):
    rng = np.random.default_rng(stride * 100 + padding * 10 + dilation + groups)
    n, c_in, length = 2, 4, 12
    c_out, kernel = 6, 3
    x = rng.normal(size=(n, c_in, length)).astype(np.float64)
    w = rng.normal(size=(c_out, c_in // groups, kernel)).astype(np.float64) * 0.5
    b_arr = rng.normal(size=(c_out,)).astype(np.float64) if bias else None
    do_shape = nn_functional.conv1d(
        x, w, bias=b_arr, stride=stride, padding=padding,
        dilation=dilation, groups=groups,
    ).shape
    do = np.ones(do_shape, dtype=np.float64)

    grad_x, grad_w, grad_bias = get_vjp("conv1d")(
        do, x, w, b_arr, stride=stride, padding=padding,
        dilation=dilation, groups=groups,
    )

    # Compare against the fp64 forward so the FD reference doesn't pull
    # fp32 quantization noise into the comparison. (`nn_functional.conv1d`
    # is fp32-only by design; for analytic-gradient verification we use the
    # bit-exact fp64 helper that backs `_conv1d_forward_fp64` / the JVP.)
    def _fp64_forward(x_in, w_in, b_in):
        y = _conv1d_forward_fp64(
            x_in, w_in, stride=stride, padding=padding,
            dilation=dilation, groups=groups,
        )
        if b_in is not None:
            y = y + b_in.reshape(1, -1, 1)
        return y

    expected_x = _numeric_grad(lambda v: _fp64_forward(v, w, b_arr), x)
    expected_w = _numeric_grad(lambda v: _fp64_forward(x, v, b_arr), w)

    np.testing.assert_allclose(grad_x, expected_x, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(grad_w, expected_w, atol=1e-4, rtol=1e-4)

    if bias:
        # `grad_bias[oc]` should be the sum of `do[:, oc, :]` over batch+pos.
        np.testing.assert_allclose(grad_bias, do.sum(axis=(0, 2)))
    else:
        assert grad_bias is None


# ── focused correctness: smallest non-trivial case ─────────────────────────


def test_conv1d_vjp_no_padding_no_stride_simple_shape():
    """Verifies the bare loop math against an analytic computation."""
    # Forward: y[b,oc,p] = Σ_k x[b,0,p+k] * w[oc,0,k]
    x = np.array([[[1.0, 2.0, 3.0, 4.0]]])  # (1, 1, 4)
    w = np.array([[[0.5, -1.0, 2.0]]])  # (1, 1, 3)  -> kernel=3
    do = np.array([[[1.0, 1.0]]])  # out_len = 4 - 3 + 1 = 2
    grad_x, grad_w, grad_bias = get_vjp("conv1d")(do, x, w, None)

    # Analytic:
    #   y[0,0,0] = 0.5*1 + (-1)*2 + 2*3 = 4.5
    #   y[0,0,1] = 0.5*2 + (-1)*3 + 2*4 = 6
    #   grad_x[0,0,p] = Σ_{pos,k where pos+k==p} do[0,0,pos]*w[0,0,k]
    #     p=0: pos=0,k=0 -> 1*0.5 = 0.5
    #     p=1: pos=0,k=1 + pos=1,k=0 -> -1 + 0.5 = -0.5
    #     p=2: pos=0,k=2 + pos=1,k=1 -> 2 + -1 = 1.0
    #     p=3: pos=1,k=2 -> 2.0
    expected_grad_x = np.array([[[0.5, -0.5, 1.0, 2.0]]])
    np.testing.assert_allclose(grad_x, expected_grad_x)

    #   grad_w[0,0,k] = Σ_pos do[0,0,pos]*x[0,0,pos+k]
    #     k=0: 1*1 + 1*2 = 3
    #     k=1: 1*2 + 1*3 = 5
    #     k=2: 1*3 + 1*4 = 7
    expected_grad_w = np.array([[[3.0, 5.0, 7.0]]])
    np.testing.assert_allclose(grad_w, expected_grad_w)
    assert grad_bias is None


# ── JVP across the parameter matrix ────────────────────────────────────────


@pytest.mark.parametrize("stride,padding,dilation,groups", list(itertools.product(
    [1, 2], [0, 1], [1, 2], [1, 2],
)))
def test_conv1d_jvp_matches_finite_difference(stride, padding, dilation, groups):
    rng = np.random.default_rng(stride + padding * 7 + dilation * 13 + groups * 31)
    n, c_in, length = 1, 4, 10
    c_out, kernel = 4, 3
    if length + 2 * padding - dilation * (kernel - 1) - 1 < 0:
        pytest.skip("output would be empty")
    x = rng.normal(size=(n, c_in, length))
    w = rng.normal(size=(c_out, c_in // groups, kernel)) * 0.5
    dx = rng.normal(size=x.shape) * 0.05
    dW = rng.normal(size=w.shape) * 0.05

    primal, tangent = get_jvp("conv1d")(
        (x, w), (dx, dW),
        stride=stride, padding=padding, dilation=dilation, groups=groups,
    )

    expected_primal = _conv1d_forward_fp64(
        x, w, stride=stride, padding=padding, dilation=dilation, groups=groups,
    )
    np.testing.assert_allclose(primal, expected_primal, atol=1e-12)

    # FD reference against the fp64 helper — ground-truth comparison.
    kw = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    expected_tan = _numeric_jvp(
        lambda v: _conv1d_forward_fp64(v, w, **kw), x, dx,
    ) + _numeric_jvp(
        lambda v: _conv1d_forward_fp64(x, v, **kw), w, dW,
    )
    np.testing.assert_allclose(tangent, expected_tan, atol=1e-4, rtol=1e-4)


def test_conv1d_jvp_with_bias():
    """Bias tangent flows through the additive constant directly."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 2, 6))
    w = rng.normal(size=(3, 2, 3)) * 0.5
    b = rng.normal(size=(3,))
    dx = np.zeros_like(x)
    dW = np.zeros_like(w)
    db = np.array([1.0, 2.0, 3.0])

    primal, tangent = get_jvp("conv1d")(
        (x, w, b), (dx, dW, db), stride=1, padding=0, dilation=1, groups=1,
    )
    # With dx = dW = 0, the tangent is purely the bias broadcast.
    expected_tan = np.broadcast_to(db.reshape(1, -1, 1), tangent.shape).astype(np.float64)
    np.testing.assert_allclose(tangent, expected_tan)


# ── registry promotion ────────────────────────────────────────────────────


def test_conv1d_registry_entry_reports_vjp_and_jvp_complete():
    from tessera.compiler.primitive_coverage import coverage_for

    entry = coverage_for("conv1d")
    assert entry.contract_status["vjp"] == "complete", (
        "registering a VJP for conv1d must auto-flip the dashboard"
    )
    assert entry.contract_status["jvp"] == "complete"
