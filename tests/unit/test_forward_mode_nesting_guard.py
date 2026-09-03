"""Forward-over-forward must not report a silent zero.

`jvp(jvp(f))` returned 0.0 for functions with perfectly good second
derivatives. `_ACTIVE_JVP` is a single contextvar, so an inner `jvp`
shadows the outer one completely: every `tessera.ops.*` call goes to the
inner trace, the outer trace never sees the output, and the scalar branch
of `jvp` filled the missing tangent with zeros.

This is the forward-mode counterpart of the reverse-over-reverse case
AD-WEIL-1/MSW-1 refuses — and it was the worse of the two, because reverse
mode at least raises. The tuple-output branch already refused it; only the
scalar branch failed open.

Found while scoping MSW-8: a PINN's loss needs a second derivative of the
network with respect to its inputs, and both routes to one were broken —
reverse-over-reverse fails closed (MSW-1), forward-over-forward returned a
plausible zero.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera
from tessera import autodiff as A

ops = tessera.ops


X = np.array([0.7])
ONE = np.array([1.0])


def _f(z):
    return ops.sum(ops.tanh(z))


def _d1(z):
    _, tangent = A.jvp(_f, (z,), (ONE,))
    return tangent


def test_first_order_forward_mode_is_unaffected():
    """The guard must not disturb the mode that works."""
    t = np.tanh(0.7)
    _, d1 = A.jvp(_f, (X,), (ONE,))
    assert float(np.asarray(d1)) == pytest.approx(1.0 - t * t, rel=1e-12)


def test_forward_over_forward_refuses_instead_of_returning_zero():
    """The defect: 0.0 where the analytic second derivative is -0.767."""
    analytic = -2.0 * np.tanh(0.7) * (1.0 - np.tanh(0.7) ** 2)
    assert abs(analytic) > 0.5, "fixture must have a real second derivative"
    with pytest.raises(Exception, match="nested forward-mode trace"):
        A.jvp(_d1, (X,), (ONE,))


def test_the_refusal_names_a_route_that_works():
    """A diagnostic that only says no costs the reader the next hour."""
    with pytest.raises(Exception) as excinfo:
        A.jvp(_d1, (X,), (ONE,))
    message = str(excinfo.value)
    assert "jet" in message, "must point at the surface that does compose"
    assert "hvp" in message


def test_a_genuinely_tangent_free_output_still_returns_zero():
    """An honest zero must survive the guard.

    MSW-1's refinement, applied here: a function that ignores its input has
    a real zero derivative, and turning that into a refusal would trade a
    false-negative class for a false-positive one.
    """
    constant = lambda z: ops.sum(ops.mul(np.array([3.0]), np.array([2.0])))
    _, tangent = A.jvp(constant, (X,), (ONE,))
    assert float(np.asarray(tangent)) == 0.0


def test_an_unrelated_inner_jvp_does_not_poison_an_honest_zero():
    """Per-value, not blanket — the distinction MSW-1 had to make twice.

    A function that opens an inner forward trace for something unrelated
    AND genuinely ignores its own input must still get its legitimate zero,
    not a refusal. Flagging on ENTRY (the mere fact a nested trace opened)
    would fail this; recording what the inner trace actually consumed does
    not.
    """
    def ignores_input_but_nests(z):
        other = np.array([0.25])
        A.jvp(_f, (other,), (ONE,))          # unrelated inner trace
        return ops.sum(ops.mul(np.array([3.0]), np.array([2.0])))

    _, tangent = A.jvp(ignores_input_but_nests, (X,), (ONE,))
    assert float(np.asarray(tangent)) == 0.0


def test_reverse_inside_forward_still_fails_closed():
    """The neighbouring case that was already correct stays correct."""
    with pytest.raises(Exception, match="reverse mode inside an active forward"):
        A.grad(_d1)(X)


def test_the_jet_surface_is_the_working_route():
    """What the diagnostic points at must actually work.

    A guard that redirects to a broken alternative is worse than none.
    """
    from tessera.autodiff import TruncatedJet, jet_lift, jet_trace

    W = TruncatedJet(2)
    coeffs = jet_trace(_f)(W, jet_lift(W, X, ONE))
    second = 2.0 * float(np.asarray(coeffs[2]))
    t = np.tanh(0.7)
    assert second == pytest.approx(-2.0 * t * (1.0 - t * t), rel=1e-9)
