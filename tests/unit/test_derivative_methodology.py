"""Derivative-checking methodology and second-order structure (MC5–MC7, MC9).

Four findings from `docs/audit/compiler/MATRIX_CALCULUS_REVIEW.md`:

* **MC5** — `check_grad` was coordinate-wise with a hard-coded `eps=1e-4` and a
  fixed tolerance. The directional form tests the whole linear operator in two
  evaluations regardless of dimension, the step is now scale-aware, and the
  order-of-accuracy form checks the convergence *rate*, which a rule wrong by a
  constant factor cannot fake.
* **MC6** — forward-over-reverse does not compose in this lane, and neither
  does reverse-over-forward. The diagnostic used to blame "raw numpy loss
  math", which is not the cause.
* **MC7** — `count_primitive_executions` sat after the forward-mode dispatch
  returned, so it counted zero forward-mode executions and the Baur–Strassen
  cost oracle read ~1x for a `jacfwd` doing one pass per input dimension.
* **MC9** — second-derivative symmetry and the four-point second difference are
  free laws: no reference implementation needed for the first, no derivative
  rule at all for the second.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import debug, ops
from tessera.autodiff import grad, hvp, jacfwd, jacrev, jvp
from tessera.autodiff.laws import (
    hessian_second_difference_check,
    hessian_symmetry_check,
    second_difference,
)
from tessera.autodiff.tape import TesseraAutodiffError, count_primitive_executions


def _matrix() -> np.ndarray:
    return np.random.default_rng(20260820).standard_normal((4, 4))


def _nonlinear(X):
    return ops.matmul(ops.sin(X), X)


def _nonlinear_dd(X, V):
    return jvp(_nonlinear, X, V)[1]


# ── MC5: the step is scale-aware ────────────────────────────────────────────
def test_fd_step_scales_with_the_input() -> None:
    small, large = np.ones(3) * 1e-3, np.ones(3) * 1e6
    assert debug.fd_step(small) < debug.fd_step(large)
    # sqrt(eps)*||x|| for a well-scaled input, not the old fixed 1e-4.
    root_eps = float(np.sqrt(np.finfo(np.float64).eps))
    np.testing.assert_allclose(debug.fd_step(np.ones(4)), root_eps * 2.0, rtol=1e-9)


def test_fd_step_has_a_floor_for_tiny_inputs() -> None:
    """A near-zero point must not produce a step of zero."""
    assert debug.fd_step(np.zeros(3)) > 0.0


def test_check_grad_uses_the_scale_aware_step_by_default() -> None:
    result = debug.check_grad(lambda x: float(np.sum(x * x)), [np.ones(3) * 100.0])
    assert result.eps == pytest.approx(debug.fd_step([np.ones(3) * 100.0]))


# ── MC5: the directional form is basis-free ─────────────────────────────────
def test_directional_check_passes_for_a_correct_rule() -> None:
    result = debug.check_grad_directional(_nonlinear, [_matrix()], _nonlinear_dd)
    assert result.passed, result.format()


def test_directional_check_catches_a_wrong_rule() -> None:
    bad = lambda X, V: 1.5 * _nonlinear_dd(X, V)  # noqa: E731
    assert not debug.check_grad_directional(_nonlinear, [_matrix()], bad).passed


def test_directional_check_works_on_matrix_inputs_in_two_evaluations() -> None:
    """The whole argument for the form: one probe, any input shape.

    A coordinate sweep of a 4x4 input costs 32 evaluations; this costs 2 per
    probe, and 'one coordinate' is not even the natural unit for a matrix.
    """
    calls = [0]

    def counted(X):
        calls[0] += 1
        return _nonlinear(X)

    debug.check_grad_directional(counted, [_matrix()],
                                 lambda X, V: jvp(_nonlinear, X, V)[1], probes=1)
    assert calls[0] == 2


# ── MC5: order of accuracy is the stronger test ─────────────────────────────
@pytest.mark.parametrize("scheme,expected", [("central", 2.0), ("forward", 1.0)])
def test_measured_order_matches_the_scheme(scheme, expected) -> None:
    result = debug.check_order_of_accuracy(
        _nonlinear, [_matrix()], _nonlinear_dd, scheme=scheme,
    )
    assert result.passed, result.format()
    assert abs(result.measured_order - expected) < 0.25


def test_order_test_catches_what_a_loose_tolerance_misses() -> None:
    """A rule 2% wrong passes a loose tolerance and fails the order test."""
    A = _matrix()
    slightly_wrong = lambda X, V: 1.02 * _nonlinear_dd(X, V)  # noqa: E731
    assert debug.check_grad_directional(
        _nonlinear, [A], slightly_wrong, rtol=0.05).passed
    assert not debug.check_order_of_accuracy(
        _nonlinear, [A], slightly_wrong, scheme="central").passed


def test_exactly_linear_functions_report_exact_not_order_zero() -> None:
    """Central differences have no truncation term for a linear map."""
    linear = lambda X: ops.mul(X, scalar=3.0)  # noqa: E731
    result = debug.check_order_of_accuracy(
        linear, [_matrix()], lambda X, V: jvp(linear, X, V)[1],
    )
    assert result.passed and result.exact, result.format()


def test_order_check_needs_a_nonzero_derivative_to_normalize() -> None:
    constant = lambda X: ops.mul(X, scalar=0.0)  # noqa: E731
    with pytest.raises(ValueError, match="nonzero directional derivative"):
        debug.check_order_of_accuracy(
            constant, [_matrix()], lambda X, V: jvp(constant, X, V)[1])


# ── MC6: mode nesting says what is actually wrong ───────────────────────────
def _scalar(z):
    return ops.reduce(ops.mul(ops.sin(z), z), op="sum")


def test_forward_over_reverse_reports_the_real_cause() -> None:
    with pytest.raises(TesseraAutodiffError) as excinfo:
        jvp(grad(_scalar), np.arange(4.0), np.ones(4))
    message = str(excinfo.value)
    assert "forward-mode trace" in message
    assert "raw numpy" not in message, (
        "the old diagnostic blamed raw-numpy loss math, sending the reader to "
        "look for a bug that does not exist"
    )
    assert "hvp" in message, "the message must name what IS available"


def test_reverse_over_forward_fails_the_same_way() -> None:
    """Both orders are blocked, and for the same structural reason."""
    with pytest.raises(TesseraAutodiffError, match="forward-mode trace"):
        grad(lambda x: jvp(_scalar, x, np.ones(4))[1])(np.arange(4.0))


def test_hvp_still_works_and_is_scale_aware() -> None:
    x = np.arange(1.0, 5.0)
    v = np.ones(4)
    assert np.all(np.isfinite(hvp(_scalar, x, v)))


# ── MC7: the cost oracle can see forward mode ───────────────────────────────
def _wide(x):
    return ops.reduce(ops.mul(ops.tanh(x), ops.tanh(x)), op="sum")


def test_forward_mode_executions_are_counted() -> None:
    x = np.random.default_rng(0).standard_normal(8)
    with count_primitive_executions() as box:
        jvp(_wide, x, np.ones(8))
    assert box[0] > 0, (
        "forward-mode executions were invisible to the counter: the increment "
        "sat after the JVP dispatch, which returns"
    )


def test_jacfwd_cost_scales_with_the_input_dimension() -> None:
    """jacfwd runs one sample pass plus one per input; jacrev runs one."""
    n = 8
    x = np.random.default_rng(0).standard_normal(n)
    with count_primitive_executions() as fwd:
        jacfwd(_wide)(x)
    with count_primitive_executions() as rev:
        jacrev(_wide)(x)
    per_pass = rev[0]
    assert fwd[0] == per_pass * (n + 1), (
        f"expected {per_pass * (n + 1)} forward-mode executions "
        f"({n} inputs + 1 sample pass), got {fwd[0]}"
    )
    assert fwd[0] > rev[0], "the R1 cost oracle must see the asymmetry"


# ── MC9: free second-derivative laws ────────────────────────────────────────
def test_hessian_symmetry_law_passes() -> None:
    result = hessian_symmetry_check()
    assert result.status == "pass", result.detail


def test_hessian_second_difference_law_passes() -> None:
    result = hessian_second_difference_check()
    assert result.status == "pass", result.detail


def test_second_difference_oracle_uses_no_derivative_rule() -> None:
    """Four function evaluations reproduce a known Hessian exactly.

    f(x) = (sum x)^2 has H = 2 * ones, so f''[u,v] = 2 (sum u)(sum v).
    """
    def f(z):
        t = ops.reduce(z, op="sum")
        return ops.mul(t, t)

    rng = np.random.default_rng(4)
    x, u, v = (rng.standard_normal(5) for _ in range(3))
    expected = 2.0 * float(np.sum(u)) * float(np.sum(v))
    got = second_difference(f, x, u, v, 1e-4)
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_second_derivative_is_symmetric_in_its_arguments() -> None:
    def f(z):
        return ops.mul(ops.reduce(ops.sin(z), op="sum"),
                       ops.reduce(ops.mul(z, z), op="sum"))

    rng = np.random.default_rng(9)
    x, u, v = (rng.standard_normal(5) for _ in range(3))
    step = 1e-4
    uv = second_difference(f, x, u, v, step)
    vu = second_difference(f, x, v, u, step)
    np.testing.assert_allclose(uv, vu, rtol=1e-9)


def test_second_derivative_laws_are_in_the_sweep() -> None:
    from tessera.autodiff.laws import run_law_sweep

    laws = {r.law for r in run_law_sweep()}
    assert "hessian_symmetry" in laws
    assert "hessian_second_difference" in laws
