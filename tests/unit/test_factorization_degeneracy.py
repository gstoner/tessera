"""Enforcer for the declared `degeneracy_policy` semantic key (MC2).

Decision #29 says a declaration must have a consumer; this file is the consumer
for ``tessera.autodiff.degeneracy``. It probes each factorization rule *exactly
at* its degeneracy and asserts the declared policy is what happens — including
the negative fixtures whose correct output is the **diagnostic** rather than a
number (Decision #10a).

The defect this pins (MC2, ``docs/audit/compiler/MATRIX_CALCULUS_REVIEW.md``):
``vjp_svd`` built its coupling matrix as ``1/(s2[None,:] - s2[:,None] +
np.eye(n)*1e-12)`` and then zeroed the diagonal, so the ``eps`` guard covered
only the entries it immediately erased. A repeated singular value produced an
all-``NaN`` gradient behind bare numpy warnings; a near-repeated one produced a
huge wrong gradient with no warning at all.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tessera.autodiff import degeneracy as deg
from tessera.autodiff.jvp import _JVPS
from tessera.autodiff.vjp import _VJPS

CODE = "E_DEGENERATE_FACTORIZATION"


# ── fixtures ────────────────────────────────────────────────────────────────
def _degenerate_matrix() -> np.ndarray:
    """Singular values (2, 2, 2) — a single three-member degenerate cluster."""
    return np.eye(3) * 2.0


def _partly_degenerate_matrix() -> np.ndarray:
    """Singular values (3, 2, 2): one simple value plus a degenerate pair."""
    return np.diag([3.0, 2.0, 2.0])


def _regular_matrix() -> np.ndarray:
    rng = np.random.default_rng(20260820)
    return rng.standard_normal((4, 4))


def _full_cotangent(shape=(3, 3), n=3, value=0.1):
    return (np.full(shape, value), np.full(n, value), np.full(shape, value))


# ── registry structure ──────────────────────────────────────────────────────
def test_every_declared_op_has_a_rule_and_a_supported_default() -> None:
    for op, policy in deg.FACTORIZATION_DEGENERACY.items():
        assert op in _VJPS, f"{op} declares a degeneracy policy but has no VJP"
        supported = deg.SUPPORTED_POLICIES[op]
        assert policy in supported, (
            f"{op} declares default policy {policy!r}, which it does not implement"
        )


def test_default_policy_is_fail_closed_everywhere() -> None:
    # The whole point of the key: absence of an explicit choice must refuse,
    # never damp, never guess (Decision #21a).
    for op, policy in deg.FACTORIZATION_DEGENERACY.items():
        assert policy == deg.FAIL_CLOSED, f"{op} defaults to {policy!r}"


def test_undeclared_op_fails_closed() -> None:
    # `polar` has no factorization rule in-tree, so it has no declared policy.
    with pytest.raises(KeyError, match="no declared degeneracy policy"):
        deg.declared_policy("polar")


def test_unsupported_policy_is_rejected_not_downgraded() -> None:
    # `generalized` is implemented for svd only; qr must refuse it rather than
    # silently doing something else (Decision #29).
    with deg.degeneracy_policy("generalized"):
        with pytest.raises(deg.TesseraDegeneracyError, match="does not implement"):
            deg.active_policy("qr")


# ── policy parsing ──────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "spec, expected",
    [
        ("fail_closed", (deg.FAIL_CLOSED, None)),
        ("generalized", (deg.GENERALIZED, None)),
        ("unchecked", (deg.UNCHECKED, None)),
        ("damped:1e-6", (deg.DAMPED, 1e-6)),
    ],
)
def test_parse_policy_accepts_the_declared_vocabulary(spec, expected) -> None:
    assert deg.parse_policy(spec) == expected


@pytest.mark.parametrize(
    "spec",
    ["damped", "damped:", "damped:not-a-number", "damped:-1", "damped:0",
     "fail_closed:1e-6", "regularized", ""],
)
def test_parse_policy_refuses_everything_else(spec) -> None:
    with pytest.raises(deg.TesseraDegeneracyError, match=CODE):
        deg.parse_policy(spec)


def test_context_manager_validates_eagerly() -> None:
    # A typo must fail at the `with`, not silently at first use of a rule.
    with pytest.raises(deg.TesseraDegeneracyError, match=CODE):
        with deg.degeneracy_policy("damped"):  # missing :<tau>
            pass


def test_context_manager_restores_the_declared_default() -> None:
    assert deg.active_policy("svd")[0] == deg.FAIL_CLOSED
    with deg.degeneracy_policy("unchecked"):
        assert deg.active_policy("svd")[0] == deg.UNCHECKED
    assert deg.active_policy("svd")[0] == deg.FAIL_CLOSED


# ── cluster detection ───────────────────────────────────────────────────────
def test_clusters_report_only_genuinely_degenerate_groups() -> None:
    tol = deg.existence_tolerance(4)
    assert deg.singular_value_clusters(np.array([4.0, 3.0, 2.0, 1.0]), tol=tol) == []
    assert deg.singular_value_clusters(np.array([3.0, 2.0, 2.0]), tol=tol) == [(1, 2)]
    assert deg.singular_value_clusters(np.array([2.0, 2.0, 2.0]), tol=tol) == [(0, 1, 2)]
    # An all-zero matrix has no scale to be relative to; every value coincides.
    assert deg.singular_value_clusters(np.zeros(3), tol=tol) == [(0, 1, 2)]


# ── NEGATIVE FIXTURES: the correct output is the diagnostic ─────────────────
def test_svd_vjp_refuses_at_repeated_singular_values() -> None:
    with pytest.raises(deg.TesseraDegeneracyError) as excinfo:
        _VJPS["svd"](_full_cotangent(), _degenerate_matrix())
    message = str(excinfo.value)
    assert CODE in message
    assert "svd backward" in message
    assert "Singular values 0 and 1 coincide" in message      # the index pair
    assert "[(0, 1, 2)]" in message                            # the cluster


def test_svd_vjp_refuses_when_only_part_of_the_spectrum_is_degenerate() -> None:
    with pytest.raises(deg.TesseraDegeneracyError, match=r"Singular values 1 and 2"):
        _VJPS["svd"](np.ones(3), _partly_degenerate_matrix())


def test_svd_forward_mode_refuses_at_repeated_singular_values() -> None:
    with pytest.raises(deg.TesseraDegeneracyError, match="svd forward-mode"):
        _JVPS["svd"]((_degenerate_matrix(),), (np.ones((3, 3)),))


def test_qr_refuses_on_a_rank_deficient_factor() -> None:
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])            # rank 1
    for registry, mode in ((_VJPS, "backward"), (_JVPS, "forward-mode")):
        with pytest.raises(deg.TesseraDegeneracyError) as excinfo:
            if registry is _VJPS:
                registry["qr"](np.ones((2, 2)), singular)
            else:
                registry["qr"]((singular,), (np.ones((2, 2)),))
        message = str(excinfo.value)
        assert CODE in message and f"qr {mode}" in message
        assert "R factor is numerically rank deficient" in message


def test_tri_solve_refuses_on_a_singular_triangle() -> None:
    T = np.array([[0.0, 0.0], [1.0, 2.0]])                   # zero pivot
    with pytest.raises(deg.TesseraDegeneracyError, match="T factor"):
        _VJPS["tri_solve"](np.ones(2), T, np.ones(2))


def test_cholesky_converts_a_numpy_failure_into_a_tessera_diagnostic() -> None:
    not_pd = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(deg.TesseraDegeneracyError) as excinfo:
        _VJPS["cholesky"](np.ones((2, 2)), not_pd)
    message = str(excinfo.value)
    assert CODE in message
    assert "positive-definite" in message
    # The raw numpy error is preserved as the cause, not swallowed.
    assert isinstance(excinfo.value.__cause__, np.linalg.LinAlgError)


def test_no_checked_policy_ever_returns_nan() -> None:
    """The reported defect, pinned: NaN must never be the answer."""
    for policy in (deg.FAIL_CLOSED, deg.GENERALIZED, "damped:1e-6"):
        with deg.degeneracy_policy(policy):
            try:
                (grad,) = _VJPS["svd"](_full_cotangent(), _degenerate_matrix())
            except deg.TesseraDegeneracyError:
                continue          # refusing is a correct outcome
            assert np.all(np.isfinite(grad)), (
                f"policy {policy!r} returned a non-finite gradient"
            )


def test_degenerate_input_emits_no_bare_numpy_warning() -> None:
    """The old rule signalled only through numpy's own RuntimeWarnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(deg.TesseraDegeneracyError):
            _VJPS["svd"](_full_cotangent(), _degenerate_matrix())
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


# ── eigh: the same contract on eigenvalue gaps (notes §13.2.1) ──────────────
def _degenerate_symmetric() -> np.ndarray:
    """Eigenvalues (2, 2, 1) — a degenerate pair plus a simple value."""
    return np.diag([2.0, 2.0, 1.0])


def test_eigh_refuses_at_an_eigenvalue_crossing() -> None:
    with pytest.raises(deg.TesseraDegeneracyError) as excinfo:
        _VJPS["eigh"]((np.ones(3), np.ones((3, 3))), _degenerate_symmetric())
    message = str(excinfo.value)
    assert CODE in message
    assert "eigh backward" in message
    assert "Eigenvalues" in message and "coincide" in message


def test_eigh_forward_mode_refuses_at_a_crossing() -> None:
    with pytest.raises(deg.TesseraDegeneracyError, match="eigh forward-mode"):
        _JVPS["eigh"]((_degenerate_symmetric(),), (np.eye(3),))


def test_eigh_generalized_admits_the_eigenvalue_sum() -> None:
    """sum(eigenvalues) is the trace — differentiable at a crossing, gradient I.

    Individual eigenvalues are not differentiable inside a degenerate cluster,
    but any symmetric function of the cluster is, and the sum is the cleanest
    example: it equals `trace(S)`, whose gradient is the identity everywhere.
    """
    with deg.degeneracy_policy("generalized"):
        (grad,) = _VJPS["eigh"](np.ones(3), _degenerate_symmetric())
    np.testing.assert_allclose(grad, np.eye(3), atol=1e-12)


def test_eigh_generalized_refuses_individual_eigenvalues() -> None:
    # `eigh` returns ascending eigenvalues, so diag([2,2,1]) gives w = [1,2,2]
    # and the degenerate cluster is indices (1, 2). A cotangent that weights
    # those two differently asks for an individual eigenvalue inside the
    # cluster, which is exactly what does not exist.
    with deg.degeneracy_policy("generalized"):
        with pytest.raises(deg.TesseraDegeneracyError, match="varies across cluster"):
            _VJPS["eigh"](np.array([0.0, 1.0, 0.0]), _degenerate_symmetric())


def test_eigh_is_silent_on_a_well_separated_spectrum() -> None:
    S = np.diag([5.0, 3.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (grad,) = _VJPS["eigh"](np.ones(3), S)
    np.testing.assert_allclose(grad, np.eye(3), atol=1e-12)


# ── GENERALIZED: admit the rotation-invariant part, refuse the rest ─────────
def test_generalized_admits_the_nuclear_norm_gradient() -> None:
    """d(sum of singular values) exists at a degenerate full-rank A: it is U·Vᵀ."""
    A = _degenerate_matrix()
    U, _s, Vt = np.linalg.svd(A, full_matrices=False)
    with deg.degeneracy_policy("generalized"):
        (grad,) = _VJPS["svd"](np.ones(3), A)
    np.testing.assert_allclose(grad, U @ Vt, atol=1e-12)


def test_generalized_refuses_unequal_weights_across_a_cluster() -> None:
    # Individual s_i inside a cluster are not differentiable; only symmetric
    # functions of the cluster are.
    with deg.degeneracy_policy("generalized"):
        with pytest.raises(deg.TesseraDegeneracyError, match="varies across cluster"):
            _VJPS["svd"](np.array([1.0, 2.0, 3.0]), _degenerate_matrix())


def test_generalized_admits_equal_weights_within_each_cluster() -> None:
    # (3, 2, 2): the simple value may carry its own weight; the pair must share.
    A = _partly_degenerate_matrix()
    U, _s, Vt = np.linalg.svd(A, full_matrices=False)
    with deg.degeneracy_policy("generalized"):
        (grad,) = _VJPS["svd"](np.array([5.0, 1.0, 1.0]), A)
    np.testing.assert_allclose(grad, U @ np.diag([5.0, 1.0, 1.0]) @ Vt, atol=1e-12)


def test_generalized_tolerates_rounding_noise_in_the_cotangent() -> None:
    """Admissibility is judged against the cotangent's noise floor, not n*eps.

    A cotangent that arrived through a chain of ops carries rounding noise many
    orders above the existence threshold; judging it there would refuse every
    realistic call and make the policy unusable.
    """
    A = _degenerate_matrix()
    U, _s, Vt = np.linalg.svd(A, full_matrices=False)
    noisy = np.ones(3) + np.array([1e-13, -2e-13, 5e-14])
    with deg.degeneracy_policy("generalized"):
        (grad,) = _VJPS["svd"](noisy, A)
    np.testing.assert_allclose(grad, U @ np.diag(noisy) @ Vt, atol=1e-12)


def test_generalized_still_refuses_a_spread_above_the_noise_floor() -> None:
    # An order-1e-6 spread is far above sqrt(eps): a real request for a
    # derivative that does not exist, not rounding noise.
    with deg.degeneracy_policy("generalized"):
        with pytest.raises(deg.TesseraDegeneracyError, match="varies across cluster"):
            _VJPS["svd"](np.array([1.0, 1.0 + 1e-6, 1.0]), _degenerate_matrix())


def test_generalized_refuses_a_cotangent_that_rotates_within_a_cluster() -> None:
    A = _degenerate_matrix()
    # A cotangent on U with a nonzero antisymmetric part inside the cluster:
    # that asks for the derivative of an individual singular vector.
    dU = np.zeros((3, 3))
    dU[0, 1] = 1.0
    with deg.degeneracy_policy("generalized"):
        with pytest.raises(deg.TesseraDegeneracyError, match="couples indices"):
            _VJPS["svd"]((dU, np.ones(3), np.zeros((3, 3))), A)


# ── DAMPED and UNCHECKED ────────────────────────────────────────────────────
def test_damped_is_finite_and_converges_to_the_exact_rule() -> None:
    """On a *near*-degenerate spectrum, damping must vanish as tau -> 0."""
    A = np.diag([3.0, 2.0, 2.0 - 1e-4])
    cotangent = _full_cotangent()
    with deg.degeneracy_policy("unchecked"):
        (exact,) = _VJPS["svd"](cotangent, A)
    assert np.all(np.isfinite(exact))
    errors = []
    for tau in (1e-3, 1e-4, 1e-5, 1e-6):
        with deg.degeneracy_policy(f"damped:{tau}"):
            (damped,) = _VJPS["svd"](cotangent, A)
        assert np.all(np.isfinite(damped))
        errors.append(np.max(np.abs(damped - exact)))
    assert errors == sorted(errors, reverse=True), (
        f"damping error should shrink monotonically with tau, got {errors}"
    )


def test_unchecked_reproduces_the_pre_guard_behaviour() -> None:
    """The escape hatch exists and is reachable only by explicit request."""
    with deg.degeneracy_policy("unchecked"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (grad,) = _VJPS["svd"](_full_cotangent(), _degenerate_matrix())
    assert np.isnan(grad).any(), (
        "'unchecked' is documented as the old, unguarded rule; if this now "
        "returns a finite answer the documentation is wrong"
    )


# ── conditioning band: exists, but thin ─────────────────────────────────────
def test_ill_conditioned_but_defined_spectrum_warns_and_returns() -> None:
    # Gap well above `n*eps` (so the derivative exists) but below sqrt(eps).
    A = np.diag([1.0, 1.0 + 1e-12])
    with pytest.warns(deg.TesseraDegeneracyWarning, match="ill conditioned"):
        (grad,) = _VJPS["svd"]((np.zeros((2, 2)), np.ones(2), np.zeros((2, 2))), A)
    assert np.all(np.isfinite(grad))


def test_well_separated_spectrum_is_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # any warning fails the test
        (grad,) = _VJPS["svd"](np.ones(4), _regular_matrix())
    assert np.all(np.isfinite(grad))


# ── regression: the guard must not change a regular answer ──────────────────
def test_regular_svd_gradient_still_matches_finite_differences() -> None:
    """The guard is a gate, not a change of rule."""
    A = _regular_matrix()
    w = np.array([1.0, 0.5, 0.25, 0.125])

    def f(M):
        return float(w @ np.linalg.svd(M, compute_uv=False))

    (grad,) = _VJPS["svd"](w, A)
    step = 1e-6
    rng = np.random.default_rng(7)
    V = rng.standard_normal(A.shape)
    directional = (f(A + step * V) - f(A - step * V)) / (2 * step)
    np.testing.assert_allclose(float(np.sum(grad * V)), directional, rtol=1e-6)


@pytest.mark.parametrize("op", ["cholesky", "qr", "tri_solve"])
def test_regular_inputs_are_unaffected(op) -> None:
    rng = np.random.default_rng(11)
    B = rng.standard_normal((3, 3))
    spd = B @ B.T + 3.0 * np.eye(3)
    if op == "cholesky":
        (grad,) = _VJPS["cholesky"](np.ones((3, 3)), spd)
    elif op == "qr":
        (grad,) = _VJPS["qr"](np.ones((3, 3)), B)
    else:
        grad, _db = _VJPS["tri_solve"](np.ones(3), np.tril(spd), np.ones(3))
    assert np.all(np.isfinite(grad))


# ── every diagnostic carries the stable code ────────────────────────────────
def test_all_refusals_carry_the_registered_code() -> None:
    from tessera.compiler.diagnostic_codes import code_lookup

    assert code_lookup(CODE) is not None, f"{CODE} must be in the code registry"
    probes = [
        lambda: _VJPS["svd"](_full_cotangent(), _degenerate_matrix()),
        lambda: _VJPS["qr"](np.ones((2, 2)), np.array([[1.0, 2.0], [2.0, 4.0]])),
        lambda: _VJPS["cholesky"](np.ones((2, 2)), np.array([[1.0, 2.0], [2.0, 1.0]])),
        lambda: _VJPS["tri_solve"](np.ones(2), np.array([[0.0, 0.0], [1.0, 2.0]]),
                                   np.ones(2)),
    ]
    for probe in probes:
        with pytest.raises(deg.TesseraDegeneracyError) as excinfo:
            probe()
        assert str(excinfo.value).startswith(CODE)
