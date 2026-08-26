"""W4 / queue order 2 — the scan reverse rule and its numerical oracle.

`tessera.control_scan` is the fourth control primitive. The bounded symbol-body
form is normalized to SCF inside paired reverse mode; payload/dynamic/malformed
forms still fail closed with AUTODIFF_CONTROL_SCAN_UNSUPPORTED.

The bounded compiler implementation consumes this rule through canonical SCF
normalization. These rows remain its independent differential reference; they
also explain why broader dynamic/payload forms stay outside the first envelope.

── The rule ──

For ``(c_{t+1}, y_t) = body(c_t, x_t)``, outputs ``c_T`` and
``ys = [y_0..y_{T-1}]``, given cotangents ``cbar_T`` and ``ybar_t``:

    for t = T-1 .. 0:
        (cbar_t, xbar_t) = body_vjp(c_t, x_t; cbar_{t+1}, ybar_t)

returning ``cbar_0`` for the init operand and ``[xbar_0..xbar_{T-1}]`` for
``xs``. **The adjoint of a scan is a scan**, over reversed t, carrying the
carry cotangent and consuming ``(c_t, x_t, ybar_t)``.

── Why it is not on AdjointInterface ──

The reverse scan needs the BODY's paired backward (a companion function this
pass generates) and a residual tape of the intermediate carries, which the
forward scan does not stack. `buildAdjoint` receives only an OpBuilder at the
forward site and is contractually limited to emitting ops there, so it can
create neither. The rule belongs beside the scf region handling in the paired
pass, where companion functions and residual policies already live.

The carry tape is exactly the SAVE/RECOMPUTE/HYBRID choice that pass already
models: ``T x |carry|`` saved, or the forward scan replayed.
"""

from __future__ import annotations

import numpy as np
import pytest

T, N = 5, 4


def _body(c, x):
    """Deliberately NONLINEAR, so the Jacobian is not constant and a wrong
    rule cannot pass by accident."""
    n = np.tanh(c + x)
    return n, n * n


def _scan(c0, xs):
    c, ys, carries = c0, [], [c0]
    for x in xs:
        c, y = _body(c, x)
        ys.append(y)
        carries.append(c)
    return c, np.array(ys), carries


def _body_vjp(c, x, cbar_next, ybar):
    s = c + x
    n = np.tanh(s)
    dL_dn = cbar_next + ybar * (2.0 * n)      # next = n, y = n*n
    dL_ds = dL_dn * (1.0 - n * n)
    return dL_ds, dL_ds                        # s = c + x: both operands


def _scan_vjp(xs, cbar_T, ybars, carries):
    """The rule under test: a scan over reversed t."""
    cbar = cbar_T.copy()
    xbars = [None] * len(xs)
    for t in range(len(xs) - 1, -1, -1):
        cbar, xbars[t] = _body_vjp(carries[t], xs[t], cbar, ybars[t])
    return cbar, np.array(xbars)


@pytest.fixture(scope="module")
def case():
    rs = np.random.RandomState(0)
    return dict(c0=rs.randn(N), xs=rs.randn(T, N),
                cbar_T=rs.randn(N), ybars=rs.randn(T, N))


def _loss(case, c0, xs):
    c, ys, _ = _scan(c0, xs)
    return float(case["cbar_T"] @ c + (case["ybars"] * ys).sum())


def test_scan_vjp_matches_central_differences(case):
    """Both cotangents, against the definition of the derivative."""
    c0, xs = case["c0"], case["xs"]
    _, _, carries = _scan(c0, xs)
    gc0, gxs = _scan_vjp(xs, case["cbar_T"], case["ybars"], carries)

    eps = 1e-6
    basis = np.eye(N)
    fd_c0 = np.array([
        (_loss(case, c0 + eps * basis[i], xs)
         - _loss(case, c0 - eps * basis[i], xs)) / (2 * eps)
        for i in range(N)
    ])
    fd_xs = np.zeros((T, N))
    for t in range(T):
        for i in range(N):
            d = np.zeros((T, N))
            d[t, i] = eps
            fd_xs[t, i] = (_loss(case, c0, xs + d)
                           - _loss(case, c0, xs - d)) / (2 * eps)

    assert np.abs(gc0 - fd_c0).max() < 1e-8
    assert np.abs(gxs - fd_xs).max() < 1e-8


def test_the_finite_difference_check_can_fail(case):
    """A control. Central differences agree with almost anything if the
    perturbation never reaches the quantity under test, so a deliberately
    wrong rule must be caught."""
    c0, xs = case["c0"], case["xs"]
    _, _, carries = _scan(c0, xs)

    def wrong_vjp(xs_, cbar_T, ybars, carries_):
        """Forgets the per-step ybar contribution — the single most likely
        implementation slip, since the stacked output is the operand a
        for-loop rule does not have."""
        cbar = cbar_T.copy()
        xbars = [None] * len(xs_)
        for t in range(len(xs_) - 1, -1, -1):
            cbar, xbars[t] = _body_vjp(carries_[t], xs_[t], cbar,
                                       np.zeros_like(ybars[t]))
        return cbar, np.array(xbars)

    gc0, _ = wrong_vjp(xs, case["cbar_T"], case["ybars"], carries)
    eps = 1e-6
    basis = np.eye(N)
    fd_c0 = np.array([
        (_loss(case, c0 + eps * basis[i], xs)
         - _loss(case, c0 - eps * basis[i], xs)) / (2 * eps)
        for i in range(N)
    ])
    assert np.abs(gc0 - fd_c0).max() > 1e-3


def test_the_reverse_pass_needs_every_intermediate_carry(case):
    """Why a residual tape is required rather than optional.

    `body_vjp` is evaluated at `carry_t`, which is a different point for every
    t. Using the final carry throughout — the shape a rule would take if the
    tape were skipped — gives the wrong answer, so SAVE-or-RECOMPUTE is a
    correctness requirement, not a memory optimisation.
    """
    c0, xs = case["c0"], case["xs"]
    cT, _, carries = _scan(c0, xs)
    right, _ = _scan_vjp(xs, case["cbar_T"], case["ybars"], carries)
    tapeless = [cT] * len(carries)
    wrong, _ = _scan_vjp(xs, case["cbar_T"], case["ybars"], tapeless)
    assert np.abs(right - wrong).max() > 1e-3


def test_recomputing_the_carries_reproduces_the_saved_tape_exactly(case):
    """The other half of that choice: RECOMPUTE must be bit-identical to SAVE,
    or the two policies are not two implementations of one contract."""
    c0, xs = case["c0"], case["xs"]
    _, _, saved = _scan(c0, xs)
    _, _, replayed = _scan(c0, xs)
    for a, b in zip(saved, replayed):
        assert np.array_equal(a, b)
    lhs = _scan_vjp(xs, case["cbar_T"], case["ybars"], saved)
    rhs = _scan_vjp(xs, case["cbar_T"], case["ybars"], replayed)
    assert np.array_equal(lhs[0], rhs[0])
    assert np.array_equal(lhs[1], rhs[1])


def test_a_zero_trip_scan_returns_the_incoming_carry_cotangent(case):
    """The degenerate case the verifier already admits (trip = 0): the scan is
    the identity on the carry, so its adjoint is too."""
    empty = np.zeros((0, N))
    _, _, carries = _scan(case["c0"], empty)
    cbar, xbars = _scan_vjp(empty, case["cbar_T"], np.zeros((0, N)), carries)
    assert np.array_equal(cbar, case["cbar_T"])
    assert xbars.shape == (0,)
