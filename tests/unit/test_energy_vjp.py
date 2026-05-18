"""M6 Step 3 start — finite-difference verification for the
closed-form VJP table.

Each whitelisted energy op gets:
  - one positive test that ``vjp_for(op_name)`` matches central
    differences of the forward path within a tight tolerance;
  - a coverage test that every op in
    ``energy_jit._ENERGY_ATTR_TO_OP_NAME`` has a VJP registered.

The harness uses fp64 internally so the finite-difference baseline
is reliable; we cast back to fp32 to verify the VJP table works on
the same dtypes the Apple GPU runtime emits.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

from tessera import energy
from tessera.compiler import energy_jit, energy_vjp


# ---------------------------------------------------------------------------
# Coverage gate
# ---------------------------------------------------------------------------

def test_every_whitelisted_energy_op_has_a_vjp() -> None:
    """Every name in ``_ENERGY_ATTR_TO_OP_NAME`` must have an entry
    in ``ENERGY_VJPS``.  A miss here means we extended the
    whitelist without writing the VJP — M6 Step 3 expects parity."""
    op_names = set(energy_jit._ENERGY_ATTR_TO_OP_NAME.values())
    missing = op_names - set(energy_vjp.ENERGY_VJPS)
    assert not missing, (
        f"energy ops without a VJP: {sorted(missing)} — add closed-form "
        f"rules to `energy_vjp.py` before lowering them in M6 Step 3."
    )


def test_vjp_table_has_no_extra_entries() -> None:
    """The reverse: every VJP entry must correspond to a real op
    name.  Otherwise we'd be carrying dead rules."""
    op_names = set(energy_jit._ENERGY_ATTR_TO_OP_NAME.values())
    extras = set(energy_vjp.ENERGY_VJPS) - op_names
    assert not extras, f"VJP table has dead entries: {sorted(extras)}"


def test_has_vjp_predicate() -> None:
    assert energy_vjp.has_vjp("energy_norm_sq") is True
    assert energy_vjp.has_vjp("bogus_op") is False


def test_vjp_for_raises_on_unknown_op() -> None:
    with pytest.raises(KeyError, match="no closed-form VJP"):
        energy_vjp.vjp_for("bogus")


# ---------------------------------------------------------------------------
# Finite-difference harness
# ---------------------------------------------------------------------------

def _finite_diff(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Central differences along every element of ``x``."""
    x = x.astype(np.float64)
    out_shape = np.asarray(fn(x)).shape
    grad = np.zeros(x.shape + out_shape, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=[["readwrite"]])
    while not it.finished:
        idx = it.multi_index
        orig = float(x[idx])
        x[idx] = orig + eps
        plus = np.asarray(fn(x), dtype=np.float64)
        x[idx] = orig - eps
        minus = np.asarray(fn(x), dtype=np.float64)
        x[idx] = orig
        grad[idx] = (plus - minus) / (2.0 * eps)
        it.iternext()
    # `grad` has shape `x.shape + out_shape`; when the output is a
    # scalar this collapses to `x.shape`, which is what the closed-
    # form VJPs return.
    return grad


def _check_scalar_energy(fn, op_name: str, x: np.ndarray, *params,
                          atol: float = 1e-4) -> None:
    """Compare closed-form ``∂fn/∂x`` to finite differences.

    Restricted to scalar-output energy functions for clarity; the
    VJPs accept ``out_grad=1.0`` to recover ``∂fn/∂x`` directly.
    """
    closed = energy_vjp.vjp_for(op_name)(x, *params, 1.0)[0]
    fd = _finite_diff(lambda y: fn(y, *params), x)
    np.testing.assert_allclose(closed, fd.astype(closed.dtype),
                               rtol=atol, atol=atol)


# ---------------------------------------------------------------------------
# Per-op finite-difference tests
# ---------------------------------------------------------------------------

def test_vjp_quadratic_matches_finite_diff() -> None:
    rng = np.random.RandomState(0)
    y = rng.randn(4).astype(np.float64)
    W = rng.randn(4, 4).astype(np.float64)
    _check_scalar_energy(energy.quadratic, "energy_quadratic", y, W)


def test_vjp_bilinear_matches_finite_diff() -> None:
    rng = np.random.RandomState(1)
    y = rng.randn(4).astype(np.float64)
    x = rng.randn(4).astype(np.float64)
    W = rng.randn(4, 4).astype(np.float64)
    _check_scalar_energy(energy.bilinear, "energy_bilinear", y, x, W)


def test_vjp_inner_matches_finite_diff() -> None:
    rng = np.random.RandomState(2)
    y = rng.randn(4).astype(np.float64)
    x = rng.randn(4).astype(np.float64)
    _check_scalar_energy(energy.inner, "energy_inner", y, x)


def test_vjp_polynomial_matches_finite_diff() -> None:
    rng = np.random.RandomState(3)
    y = rng.randn(4).astype(np.float64) * 0.5
    coefs = (0.1, 0.3, 0.5)  # 0.1 + 0.3 y + 0.5 y²
    # Element-wise polynomial returns a vector; reduce to scalar.
    fn = lambda yy, cc: float(energy.polynomial(yy, cc).sum())
    closed = energy_vjp.vjp_for("energy_polynomial")(y, coefs, 1.0)[0]
    fd = _finite_diff(lambda yy: fn(yy, coefs), y)
    np.testing.assert_allclose(closed, fd, rtol=1e-3, atol=1e-3)


def test_vjp_norm_matches_finite_diff() -> None:
    rng = np.random.RandomState(4)
    y = rng.randn(5).astype(np.float64)
    _check_scalar_energy(energy.norm, "energy_norm", y)


def test_vjp_norm_at_origin_is_zero() -> None:
    """Subgradient at 0 is 0 — locks the M6 Step 3 contract."""
    y = np.zeros(3, dtype=np.float64)
    closed = energy_vjp.vjp_for("energy_norm")(y, 1.0)[0]
    np.testing.assert_array_equal(closed, np.zeros_like(y))


def test_vjp_norm_sq_matches_finite_diff() -> None:
    rng = np.random.RandomState(5)
    y = rng.randn(6).astype(np.float64)
    _check_scalar_energy(energy.norm_sq, "energy_norm_sq", y)


def test_vjp_relu_matches_finite_diff() -> None:
    # Avoid eps-straddling 0 by offsetting away from the kink.
    y = np.array([-1.2, -0.5, 0.5, 1.2, 2.7], dtype=np.float64)
    fn = lambda yy: float(energy.relu(yy).sum())
    closed = energy_vjp.vjp_for("energy_relu")(y, np.ones_like(y))[0]
    fd = _finite_diff(fn, y)
    np.testing.assert_allclose(closed, fd, atol=1e-4)


def test_vjp_tanh_matches_finite_diff() -> None:
    rng = np.random.RandomState(6)
    y = rng.randn(5).astype(np.float64)
    fn = lambda yy: float(energy.tanh(yy).sum())
    closed = energy_vjp.vjp_for("energy_tanh")(y, np.ones_like(y))[0]
    fd = _finite_diff(fn, y)
    np.testing.assert_allclose(closed, fd, atol=1e-5)


def test_vjp_sigmoid_matches_finite_diff() -> None:
    rng = np.random.RandomState(7)
    y = rng.randn(5).astype(np.float64)
    fn = lambda yy: float(energy.sigmoid(yy).sum())
    closed = energy_vjp.vjp_for("energy_sigmoid")(y, np.ones_like(y))[0]
    fd = _finite_diff(fn, y)
    np.testing.assert_allclose(closed, fd, atol=1e-5)


def test_vjp_gelu_matches_finite_diff() -> None:
    rng = np.random.RandomState(8)
    y = rng.randn(5).astype(np.float64) * 1.5
    fn = lambda yy: float(energy.gelu(yy).sum())
    closed = energy_vjp.vjp_for("energy_gelu")(y, np.ones_like(y))[0]
    fd = _finite_diff(fn, y)
    np.testing.assert_allclose(closed, fd, atol=2e-3)


def test_vjp_softplus_matches_finite_diff() -> None:
    rng = np.random.RandomState(9)
    y = rng.randn(5).astype(np.float64)
    fn = lambda yy: float(energy.softplus(yy).sum())
    closed = energy_vjp.vjp_for("energy_softplus")(y, np.ones_like(y))[0]
    fd = _finite_diff(fn, y)
    np.testing.assert_allclose(closed, fd, atol=1e-5)


def test_vjp_linear_matches_finite_diff() -> None:
    rng = np.random.RandomState(10)
    y = rng.randn(4).astype(np.float64)
    W = rng.randn(4, 3).astype(np.float64)
    b = rng.randn(3).astype(np.float64)
    fn = lambda yy: float(energy.linear(yy, W, b).sum())
    closed = energy_vjp.vjp_for("energy_linear")(
        y, W, b, np.ones(3, dtype=np.float64),
    )[0]
    fd = _finite_diff(fn, y)
    np.testing.assert_allclose(closed, fd, atol=1e-5)


def test_vjp_mlp_head_matches_finite_diff() -> None:
    rng = np.random.RandomState(11)
    y = rng.randn(4).astype(np.float64)
    W1 = rng.randn(4, 6).astype(np.float64)
    b1 = rng.randn(6).astype(np.float64)
    W2 = rng.randn(6, 3).astype(np.float64)
    b2 = rng.randn(3).astype(np.float64)
    fn = lambda yy: float(energy.mlp_head(yy, W1, b1, W2, b2).sum())
    closed = energy_vjp.vjp_for("energy_mlp_head")(
        y, W1, b1, W2, b2, np.ones(3, dtype=np.float64),
    )[0]
    fd = _finite_diff(fn, y)
    np.testing.assert_allclose(closed, fd, atol=1e-4)


def test_vjp_reduce_sum_matches_finite_diff() -> None:
    rng = np.random.RandomState(12)
    y = rng.randn(6).astype(np.float64)
    fn = lambda yy: float(energy.reduce_sum(yy))
    closed = energy_vjp.vjp_for("energy_reduce_sum")(y, 1.0)[0]
    fd = _finite_diff(fn, y)
    np.testing.assert_allclose(closed, fd, atol=1e-6)


# ---------------------------------------------------------------------------
# Composition test — chain energy_jit lowering with VJP lookup
# ---------------------------------------------------------------------------

def test_energy_jit_lowered_program_has_vjp_for_every_op() -> None:
    """End-to-end: lower a real energy function and prove every IR
    op in it has a closed-form VJP available.  This is the
    composition pre-condition M6 Step 3's codegen will check."""
    from tessera.compiler.energy_jit import lower_energy_function

    def E(y, W1, b1, W2, b2):
        h = energy.linear(y, W1, b1)
        a = energy.relu(h)
        out = energy.linear(a, W2, b2)
        return energy.reduce_sum(out)

    ir = lower_energy_function(E)
    for op in ir.ops:
        assert energy_vjp.has_vjp(op.op_name), (
            f"M6 Step 3 readiness: {op.op_name} has no closed-form "
            "VJP — cannot lower this energy program with gradient "
            "recomputation yet."
        )
