"""MSW-2 — exact higher-order derivatives are reachable.

Three deliverables, checked against three different kinds of oracle so a
single mistake cannot satisfy all of them:

* `laplacian_exact` against **closed forms** (an analytic Laplacian per
  supported rank) and against an independently assembled Hessian trace.
* `jet_trace` against the **canonical forward** at order 0 and the
  **registered JVP** at order 1 — the anchoring obligation
  `test_jet_struct.py` already imposes on the hand-written jets, now
  extended to traced ones.
* `laplacian_exact` against `laplacian_estimate`, which is the only check
  that ties the deterministic path to the stochastic one.

The estimator comparison needs care to mean anything: for a SEPARABLE `f`
the Hessian is diagonal and Rademacher probes are exact at one sample, so
an agreement test built on `sum(exp(x*x))` would pass no matter what
either function did. Every estimator check here uses a coupled Hessian,
and one test pins that the coupling is real.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera
from tessera import autodiff as A
from tessera.autodiff import (
    TruncatedJet, jet_lift, jet_trace, laplacian_exact, laplacian_estimate,
)

ops = tessera.ops


def _seed(x, v, order=2):
    return TruncatedJet(order), jet_lift(TruncatedJet(order), x, v)


# --- laplacian_exact against closed forms, per rank -----------------------


def test_laplacian_exact_matches_the_analytic_laplacian_rank1():
    """f(x) = sum(exp(x*x)); Δf = Σ (2 + 4 x_i²) exp(x_i²)."""
    x = np.array([0.3, -1.1, 2.0])
    jf = jet_trace(lambda v: ops.sum(ops.exp(ops.mul(v, v))))
    analytic = float(np.sum((2.0 + 4.0 * x ** 2) * np.exp(x ** 2)))
    assert laplacian_exact(jf, x) == pytest.approx(analytic, rel=1e-12)


def test_laplacian_exact_matches_the_analytic_laplacian_rank2():
    """A rank-2 field: the Laplacian sums over EVERY element, not rows.

    Getting the flattening wrong is the obvious failure here and it stays
    finite, so the closed form is over all six entries.
    """
    x = np.array([[0.4, -0.2, 1.1], [0.0, 0.7, -1.3]])
    jf = jet_trace(lambda v: ops.sum(ops.exp(ops.mul(v, v))))
    analytic = float(np.sum((2.0 + 4.0 * x ** 2) * np.exp(x ** 2)))
    assert laplacian_exact(jf, x) == pytest.approx(analytic, rel=1e-12)


def test_laplacian_exact_matches_a_coupled_analytic_hessian_trace():
    """f(x) = sum(exp(M x)) — Hessian Mᵀ diag(exp(Mx)) M is NOT diagonal."""
    rs = np.random.default_rng(0)
    M = rs.standard_normal((4, 3))
    x = np.array([0.3, -1.1, 0.7])
    jf = jet_trace(lambda v: ops.sum(ops.exp(ops.matmul(M, v))))
    hessian = M.T @ np.diag(np.exp(M @ x)) @ M
    assert laplacian_exact(jf, x) == pytest.approx(float(np.trace(hessian)), rel=1e-10)


def test_laplacian_exact_is_deterministic():
    x = np.array([0.3, -1.1, 2.0])
    jf = jet_trace(lambda v: ops.sum(ops.exp(ops.mul(v, v))))
    assert laplacian_exact(jf, x) == laplacian_exact(jf, x)


def test_laplacian_exact_refuses_an_empty_field():
    jf = jet_trace(lambda v: ops.sum(ops.mul(v, v)))
    with pytest.raises(Exception, match="at least one input element"):
        laplacian_exact(jf, np.zeros(0))


def test_laplacian_exact_refuses_a_non_scalar_output():
    jf = jet_trace(lambda v: ops.mul(v, v))     # vector out
    with pytest.raises(Exception, match="scalar-output"):
        laplacian_exact(jf, np.array([1.0, 2.0]))


# --- the deterministic and stochastic paths agree ------------------------


COUPLED_M = np.random.default_rng(0).standard_normal((4, 3))
COUPLED_X = np.array([0.3, -1.1, 0.7])


def _coupled():
    return jet_trace(lambda v: ops.sum(ops.exp(ops.matmul(COUPLED_M, v))))


def test_the_coupled_fixture_really_is_coupled():
    """Without this the estimator agreement below would be vacuous.

    Rademacher probes give `vᵀHv = tr H` exactly when H is diagonal, so a
    separable f makes the estimator exact at one sample and the comparison
    proves nothing about either function.
    """
    hessian = COUPLED_M.T @ np.diag(np.exp(COUPLED_M @ COUPLED_X)) @ COUPLED_M
    off_diagonal = hessian - np.diag(np.diag(hessian))
    assert np.max(np.abs(off_diagonal)) > 0.1 * np.max(np.abs(hessian))


def test_estimator_converges_to_the_exact_laplacian():
    import tessera.rng as rng

    jf = _coupled()
    exact = laplacian_exact(jf, COUPLED_X)
    key = rng.RNGKey(7)
    coarse = laplacian_estimate(jf, COUPLED_X, key, samples=16)
    fine = laplacian_estimate(jf, COUPLED_X, key, samples=4096)
    err_coarse = abs(coarse - exact) / abs(exact)
    err_fine = abs(fine - exact) / abs(exact)
    assert err_fine < err_coarse, "more samples must not be worse"
    assert err_fine < 0.01, f"4096 samples still {err_fine:.4f} from exact"


def test_estimator_and_exact_agree_bitwise_on_a_diagonal_hessian():
    """Where Rademacher is exact, the two must agree to floating point.

    This is the complement of the test above: the separable case makes a
    LOOSE convergence check into an exact one, so it catches a constant
    factor (a missing 2, a mean-vs-sum slip) that convergence would hide.
    """
    import tessera.rng as rng

    x = np.array([0.3, -1.1, 2.0])
    jf = jet_trace(lambda v: ops.sum(ops.exp(ops.mul(v, v))))
    exact = laplacian_exact(jf, x)
    est = laplacian_estimate(jf, x, rng.RNGKey(3), samples=8)
    assert est == pytest.approx(exact, rel=1e-12)


# --- jet_trace anchoring: order 0 == forward, order 1 == registered JVP ---


PROGRAMS = {
    "pointwise": lambda v: ops.sum(ops.exp(ops.mul(v, v))),
    "chain": lambda v: ops.sum(ops.tanh(ops.add(v, ops.mul(v, v)))),
    "matmul": lambda v: ops.sum(ops.exp(ops.matmul(COUPLED_M, v))),
    "reduction": lambda v: ops.mean(ops.mul(ops.sin(v), ops.sin(v))),
}


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_jet_trace_order0_equals_the_canonical_forward(name):
    fn = PROGRAMS[name]
    x = np.array([0.3, -1.1, 0.7])
    W, seed = _seed(x, np.ones_like(x))
    order0 = float(np.asarray(jet_trace(fn)(W, seed)[0]))
    assert order0 == pytest.approx(float(np.asarray(fn(x))), rel=1e-12)


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_jet_trace_order1_equals_the_registered_jvp(name):
    fn = PROGRAMS[name]
    x = np.array([0.3, -1.1, 0.7])
    v = np.array([1.0, -0.5, 0.25])
    W, seed = _seed(x, v)
    order1 = float(np.asarray(jet_trace(fn)(W, seed)[1]))
    _, tangent = A.jvp(fn, (x,), (v,))
    assert order1 == pytest.approx(float(np.asarray(tangent)), rel=1e-10)


def test_jet_trace_order2_matches_a_central_difference():
    """Order 2 is the coefficient the estimators consume; anchor it too.

    `a₂` is ½ d²/dt² f(x + t v), so the comparison carries the factor.
    """
    fn = PROGRAMS["matmul"]
    x = np.array([0.3, -1.1, 0.7])
    v = np.array([1.0, -0.5, 0.25])
    W, seed = _seed(x, v)
    a2 = float(np.asarray(jet_trace(fn)(W, seed)[2]))
    h = 1e-4
    second = (float(np.asarray(fn(x + h * v))) - 2 * float(np.asarray(fn(x)))
              + float(np.asarray(fn(x - h * v)))) / h ** 2
    assert 2.0 * a2 == pytest.approx(second, rel=1e-5)


# --- fail-closed behaviour ------------------------------------------------


def test_jet_trace_refuses_an_op_with_no_jet_rule():
    """An op dropped to order 0 is a wrong derivative that looks right."""
    x = np.array([0.3, -1.1, 0.7])
    W, seed = _seed(x, np.ones_like(x))
    with pytest.raises(Exception, match="no jet rule for tessera.ops.clamp"):
        jet_trace(lambda v: ops.sum(ops.clamp(v, 0.0, 1.0)))(W, seed)


def test_jet_trace_refuses_a_program_that_never_touches_ops():
    """Raw numpy is invisible to the tape, so it would silently be constant."""
    x = np.array([0.3, -1.1, 0.7])
    W, seed = _seed(x, np.ones_like(x))
    with pytest.raises(Exception, match="recorded no tessera.ops"):
        jet_trace(lambda v: np.sum(v * v))(W, seed)


def test_jet_trace_refuses_a_result_it_cannot_tie_to_the_trace():
    x = np.array([0.3, -1.1, 0.7])
    W, seed = _seed(x, np.ones_like(x))
    with pytest.raises(Exception, match="could not tie the returned value"):
        jet_trace(lambda v: float(np.asarray(ops.sum(ops.mul(v, v)))) + 0.0)(W, seed)


def test_closed_over_constants_are_order_zero_not_ignored():
    """A captured constant has zero derivative — but must still be USED.

    Treating it as absent rather than as a constant would change the value,
    so this checks the forward as well as the derivative.
    """
    c = np.array([2.0, 3.0, 4.0])
    x = np.array([0.3, -1.1, 0.7])
    fn = lambda v: ops.sum(ops.mul(c, ops.mul(v, v)))
    jf = jet_trace(fn)
    W, seed = _seed(x, np.ones_like(x))
    coeffs = jf(W, seed)
    assert float(np.asarray(coeffs[0])) == pytest.approx(float(np.asarray(fn(x))), rel=1e-12)
    assert laplacian_exact(jf, x) == pytest.approx(float(2.0 * np.sum(c)), rel=1e-12)


def test_jet_trace_retraces_so_a_second_point_is_not_stale():
    """The record is captured at one point; reusing it elsewhere is wrong.

    Evaluating the same `jet_fn` at two points must give each point's own
    answer, not the first point's.
    """
    jf = jet_trace(lambda v: ops.sum(ops.exp(ops.mul(v, v))))
    a = np.array([0.3, -1.1, 2.0])
    b = np.array([1.7, 0.2, -0.4])
    la = laplacian_exact(jf, a)
    lb = laplacian_exact(jf, b)
    analytic_b = float(np.sum((2.0 + 4.0 * b ** 2) * np.exp(b ** 2)))
    assert lb == pytest.approx(analytic_b, rel=1e-12)
    assert la != pytest.approx(lb, rel=1e-3)


# --- gaps the first pass of this suite missed ----------------------------


@pytest.mark.parametrize("build,analytic_factor", [
    (lambda v: ops.sum(ops.mul(ops.mul(v, v), 3.0)), 2.0 * 3.0),
    (lambda v: ops.sum(ops.add(ops.mul(v, v), 1.0)), 2.0),
    (lambda v: ops.sum(ops.div(ops.mul(v, v), 4.0)), 2.0 / 4.0),
])
def test_python_scalar_literals_are_constants_in_w(build, analytic_factor):
    """Literal operands must ride through as order-0 constants.

    Every other test here passes arrays, so a literal reaching `resolve` was
    an untested path — and a literal mishandled as a zero jet would change
    the value, not just the derivative.
    """
    x = np.array([0.3, -1.1, 0.7])
    assert laplacian_exact(jet_trace(build), x) == pytest.approx(
        analytic_factor * x.size, rel=1e-12)


def test_laplacian_exact_accepts_a_hand_written_jet_program():
    """`laplacian_exact` consumes any jet_fn, not only a traced one.

    `jet_trace` is a convenience over the `jet_*` vocabulary, not a
    precondition — and since every other test in this file goes through it,
    nothing else would notice if the two became coupled.
    """
    from tessera.autodiff import jet_mul, jet_sum

    x = np.array([0.3, -1.1, 0.7])
    hand = lambda W, u: jet_sum(W, jet_mul(W, u, u))
    assert laplacian_exact(hand, x) == pytest.approx(2.0 * x.size, rel=1e-12)


def test_the_trace_cache_does_not_leak_across_points():
    """The cache is keyed on the primal point; a new point must re-trace.

    Cheap to get wrong in a way tests would miss: caching on the FUNCTION
    would return the first point's record forever, and every value would
    still look plausible.
    """
    jf = jet_trace(lambda v: ops.sum(ops.exp(ops.mul(v, v))))
    for x in (np.array([0.3, -1.1, 2.0]),
              np.array([1.7, 0.2, -0.4]),
              np.array([0.3, -1.1, 2.0])):     # back to the first: still right
        analytic = float(np.sum((2.0 + 4.0 * x ** 2) * np.exp(x ** 2)))
        assert laplacian_exact(jf, x) == pytest.approx(analytic, rel=1e-12)


def test_the_trace_is_taken_once_per_point_not_once_per_direction():
    """The cache's whole purpose, asserted behaviourally.

    `laplacian_exact` evaluates `d` seed directions at ONE point, and the
    tape record depends only on the point — so the program should execute
    `d` times fewer. Counting primitive executions says this without
    introspecting the closure: a two-op program over a 3-element field runs
    2 primitives if traced once and 6 if traced per direction.

    Measured before the cache: tracing was 40% of `laplacian_exact` at
    d=128, all of it redundant.
    """
    from tessera.autodiff.tape import count_primitive_executions

    jf = jet_trace(lambda v: ops.sum(ops.mul(v, v)))   # exactly 2 ops
    x = np.array([1.0, 2.0, 3.0])

    with count_primitive_executions() as box:
        laplacian_exact(jf, x)
    assert box[0] == 2, (
        f"traced {box[0] // 2} times for a 3-direction Laplacian; the record "
        "depends on the point, not the seed"
    )

    with count_primitive_executions() as box:
        laplacian_exact(jf, x)
    assert box[0] == 0, "a repeat at the same point must not re-trace"

    with count_primitive_executions() as box:
        laplacian_exact(jf, np.array([4.0, 5.0, 6.0]))
    assert box[0] == 2, "a NEW point must re-trace — a stale record is wrong"


# --- fused op OPTIONS, not just op names (review on #698) ----------------
#
# Failing closed on an unknown op NAME was never enough. `ops.matmul` also
# takes `bias`, `residual`, `activation` and `epilogue`; the first version of
# the replay evaluated `a[0] @ a[1]` and discarded the rest, so a program with
# a `gelu` activation replayed as a LINEAR function and reported a Laplacian
# of 0.0. A rule now declares the options it interprets and the replay refuses
# any other, which closes the class rather than these two instances.


BIAS = np.ones(4)
RESIDUAL = np.full(4, 0.5)


def _fd_laplacian(fn, x, h=1e-4):
    total = 0.0
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = 1.0
        total += (float(np.asarray(fn(x + h * e))) - 2 * float(np.asarray(fn(x)))
                  + float(np.asarray(fn(x - h * e)))) / h ** 2
    return total


@pytest.mark.parametrize("build", [
    lambda M: (lambda v: ops.sum(ops.exp(ops.matmul(M, v)))),
    lambda M: (lambda v: ops.sum(ops.exp(ops.matmul(M, v, BIAS)))),
    # residual WITHOUT bias: the tape records the omitted bias as a None
    # literal, so this is the case that catches an off-by-one in the operand
    # positions — it would otherwise add the residual as if it were the bias.
    lambda M: (lambda v: ops.sum(ops.exp(ops.matmul(M, v, None, RESIDUAL)))),
    lambda M: (lambda v: ops.sum(ops.exp(ops.matmul(M, v, BIAS, RESIDUAL)))),
])
def test_fused_bias_and_residual_are_honoured(build):
    """These lift into W exactly — they are additions — so they are supported."""
    fn = build(np.random.default_rng(0).standard_normal((4, 3)))
    x = np.array([0.3, -1.1, 0.7])
    assert laplacian_exact(jet_trace(fn), x) == pytest.approx(
        _fd_laplacian(fn, x), rel=1e-4)


def test_a_fused_activation_is_refused_not_dropped():
    """The reported defect: dropping it made a nonlinear program linear.

    `sum(matmul(M, x, activation="gelu"))` returned a Laplacian of 0.0 where
    the finite-difference truth is 2.24 — finite, plausible, and wrong. No
    activation the op accepts (relu/gelu/silu) is a registered holonomic
    recurrence, so the honest answer is to refuse.
    """
    M = np.random.default_rng(0).standard_normal((4, 3))
    fn = lambda v: ops.sum(ops.matmul(M, v, activation="gelu"))
    x = np.array([0.3, -1.1, 0.7])

    assert abs(_fd_laplacian(fn, x)) > 1.0, "fixture must be nonlinear, or this proves nothing"
    with pytest.raises(Exception, match="activation"):
        laplacian_exact(jet_trace(fn), x)


def test_a_fused_epilogue_is_refused():
    M = np.random.default_rng(0).standard_normal((4, 3))
    x = np.array([0.3, -1.1, 0.7])
    with pytest.raises(Exception, match="epilogue"):
        laplacian_exact(
            jet_trace(lambda v: ops.sum(ops.matmul(M, v, epilogue={"bias": BIAS}))), x)


@pytest.mark.parametrize("build,analytic_factor", [
    (lambda v: ops.sum(ops.mul(ops.mul(v, v), scalar=3.0)), 2.0 * 3.0),
    (lambda v: ops.sum(ops.add(ops.mul(v, v), scalar=1.0)), 2.0),
])
def test_scalar_keyword_forms_are_supported(build, analytic_factor):
    """`ops.mul(x, scalar=3.0)` is canonical and used to raise IndexError.

    The tape records ONE input and keeps the value in kwargs, so reading
    `a[1]` failed for a call that executes fine outside the transform.
    """
    x = np.array([0.3, -1.1, 0.7])
    assert laplacian_exact(jet_trace(build), x) == pytest.approx(
        analytic_factor * x.size, rel=1e-12)


def test_an_option_no_rule_interprets_is_refused_generically():
    """The class, not the instance: `sum` reads axis/keepdims and nothing else.

    Without a generic check, every future op option becomes another silent
    drop waiting to be found in review.
    """
    from tessera.autodiff.jet import _JET_REPLAY

    assert _JET_REPLAY["sum"].reads == frozenset({"axis", "keepdims"})
    assert _JET_REPLAY["exp"].reads == frozenset()
    assert "activation" in _JET_REPLAY["matmul"].reads


@pytest.mark.parametrize("spelling", [
    "bias_none_kw", "bias_none_residual_kw", "bias_kw", "residual_kw", "positional",
])
def test_every_spelling_of_the_fused_operands_replays_correctly(spelling):
    """Keyword-spelled operands, including an explicit `None` (review on #700).

    `promote_operand_kwargs` is not consistent about which keyword operands
    become positional inputs: `matmul(A, B, residual=r)` promotes to four
    inputs with a None bias filler, while
    `matmul(A, B, bias=None, residual=r)` leaves BOTH in kwargs and records
    only two. So the strict unknown-kwarg check refused `bias=None` -- a
    valid no-op that replayed fine before -- and a kwarg-spelled residual
    never reached the operand path at all.
    """
    M = np.random.default_rng(0).standard_normal((4, 3))
    bias, residual = np.ones(4), np.full(4, 0.5)
    builders = {
        "bias_none_kw": lambda v: ops.matmul(M, v, bias=None),
        "bias_none_residual_kw": lambda v: ops.matmul(M, v, bias=None, residual=residual),
        "bias_kw": lambda v: ops.matmul(M, v, bias=bias),
        "residual_kw": lambda v: ops.matmul(M, v, residual=residual),
        "positional": lambda v: ops.matmul(M, v, bias, residual),
    }
    fn = lambda v: ops.sum(ops.exp(builders[spelling](v)))
    x = np.array([0.3, -1.1, 0.7])
    assert laplacian_exact(jet_trace(fn), x) == pytest.approx(
        _fd_laplacian(fn, x), rel=1e-4)


def test_a_kwarg_operand_that_is_traced_keeps_its_derivative():
    """A residual computed FROM the input must not become a constant.

    The kwarg path resolves through the replay environment first. Reading it
    as a constant would drop its dependence on x -- a smaller, quieter
    version of the activation bug: the value stays finite and the derivative
    goes wrong.
    """
    M = np.random.default_rng(0).standard_normal((3, 3))
    x = np.array([0.3, -1.1, 0.7])

    def fn(v):
        traced_residual = ops.mul(v, v)          # depends on the input
        return ops.sum(ops.exp(ops.matmul(M, v, residual=traced_residual)))

    got = laplacian_exact(jet_trace(fn), x)
    assert got == pytest.approx(_fd_laplacian(fn, x), rel=1e-4)
    # and it must differ from the constant-residual reading, or the test is
    # blind to exactly the mistake it exists to catch
    frozen = np.asarray(ops.mul(x, x))
    const_fn = lambda v: ops.sum(ops.exp(ops.matmul(M, v, residual=frozen)))
    assert got != pytest.approx(laplacian_exact(jet_trace(const_fn), x), rel=1e-3)
