"""Regression tests for the P1 autodiff-rule defects found by the 2026-08-29
review (`docs/audit/compiler/CODE_REVIEW_2026-08-29.md`).

Each rule is checked against a finite-difference reference rather than against
its own algebra, so a re-derivation that is self-consistently wrong still
fails. Where a rule cannot be correct at all, the test asserts it fails closed
instead of returning a silently wrong gradient.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.autodiff.jvp import _JVPS
from tessera.autodiff.vjp import _VJPS


def _fd(f, a, eps=1e-6):
    """Central-difference gradient of scalar `f` with respect to array `a`."""
    grad = np.zeros_like(a, dtype=np.float64)
    for idx in np.ndindex(a.shape):
        original = a[idx]
        a[idx] = original + eps
        high = f()
        a[idx] = original - eps
        low = f()
        a[idx] = original
        grad[idx] = (high - low) / (2 * eps)
    return grad


# ── flash_attn: the dropout mask must be replayed, or the rule must refuse ──

def test_flash_attn_vjp_replays_the_seeded_dropout_mask():
    """The forward applies a Bernoulli mask scaled by 1/(1-p). Differentiating
    the dropout-free function instead gave gradients ~70% wrong with no
    diagnostic; the mask is exactly reproducible when a seed is supplied."""
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((1, 4, 3))
    K = rng.standard_normal((1, 4, 3))
    V = rng.standard_normal((1, 4, 3))
    dout = rng.standard_normal((1, 4, 3))
    p, seed = 0.5, 7

    def loss():
        return float((ts.ops.flash_attn(Q, K, V, dropout_p=p, seed=seed) * dout).sum())

    dQ, dK, dV = _VJPS["flash_attn"](dout, Q, K, V, dropout_p=p, seed=seed)
    np.testing.assert_allclose(_fd(loss, Q), dQ, atol=1e-7)
    np.testing.assert_allclose(_fd(loss, K), dK, atol=1e-7)
    np.testing.assert_allclose(_fd(loss, V), dV, atol=1e-7)


def test_flash_attn_vjp_fails_closed_without_a_seed():
    """Without a seed the forward's mask is unreproducible. Returning the
    dropout-free gradient would be silently wrong, so the rule must refuse —
    the contract vjp_dropout already enforces."""
    from tessera.autodiff.tape import TesseraAutodiffError

    x = np.ones((1, 2, 2))
    with pytest.raises(TesseraAutodiffError, match="seed"):
        _VJPS["flash_attn"](x, x, x, x, dropout_p=0.5)


# ── pad: every mode that COPIES input must scatter its cotangent back ───────

@pytest.mark.parametrize("mode", ["constant", "edge", "reflect", "symmetric", "wrap"])
def test_pad_vjp_matches_finite_differences_for_every_gather_mode(mode):
    """vjp_pad ignored `mode` and always sliced, which is the constant-mode
    adjoint. For the copy modes that silently dropped the cotangent mass sitting
    in the padded region."""
    x = np.arange(5.0)
    pad_width = [(2, 2)]
    dout = np.random.default_rng(1).standard_normal(9)

    def loss():
        return float((ts.ops.pad(x, pad_width, mode=mode) * dout).sum())

    analytic = _VJPS["pad"](dout, x, pad_width=pad_width, mode=mode)[0]
    np.testing.assert_allclose(_fd(loss, x), analytic, atol=1e-7)


def test_pad_vjp_fails_closed_for_a_statistic_mode():
    """'mean' pads with a statistic of the input, not a copy of it, so there is
    no index to scatter back to. Refuse rather than return the slice."""
    from tessera.autodiff.tape import TesseraAutodiffError

    with pytest.raises(TesseraAutodiffError, match="no adjoint"):
        _VJPS["pad"](np.ones(9), np.arange(5.0), pad_width=[(2, 2)], mode="mean")


@pytest.mark.parametrize("mode", ["reflect", "edge", "symmetric", "wrap"])
def test_jvp_pad_does_not_crash_for_non_constant_modes(mode):
    """numpy rejects `constant_values` for any mode but 'constant', and jvp_pad
    passed it unconditionally — so every non-constant pad raised."""
    primal, tangent = _JVPS["pad"](
        (np.arange(3.0),), (np.ones(3),), pad_width=[(1, 1)], mode=mode)
    assert primal.shape == (5,)
    assert tangent.shape == (5,)


# ── reductions ─────────────────────────────────────────────────────────────

def test_amax_tangent_squeezes_only_the_reduced_axis():
    """`counts.squeeze()` dropped every size-1 axis, so an unrelated singleton
    dimension made the tangent broadcast to the wrong shape."""
    x = np.arange(15.0).reshape(3, 1, 5)
    dx = np.ones_like(x)
    primal, tangent = _JVPS["amax"]((x,), (dx,), axis=2, keepdims=False)
    assert tangent.shape == primal.shape == (3, 1)

    scalar_primal, scalar_tangent = _JVPS["amax"]((x,), (dx,), axis=None, keepdims=False)
    assert np.shape(scalar_tangent) == np.shape(scalar_primal) == ()


@pytest.mark.parametrize("values,index,expected", [
    ([1.5, 2.0, 3.0], 0, 6.0),   # no zeros — ratio form
    ([0.0, 2.0, 3.0], 0, 6.0),   # one zero — derivative at the zero is prod(rest)
    ([0.0, 2.0, 3.0], 1, 0.0),   # one zero — every other partial is 0
    ([0.0, 0.0, 3.0], 0, 0.0),   # two zeros — all partials are 0
])
def test_prod_tangent_is_correct_at_zero_valued_inputs(values, index, expected):
    """The ratio trick prod/x_i collapses when the slice contains a zero: the
    whole product is 0, so every term vanished — but the derivative with
    respect to the zero element is the product of the others."""
    x = np.array(values)
    dx = np.zeros(3)
    dx[index] = 1.0
    _, tangent = _JVPS["prod"]((x,), (dx,), axis=None)
    assert float(tangent) == pytest.approx(expected)


# ── optimizer adjoints ─────────────────────────────────────────────────────

def test_nesterov_velocity_cotangent_uses_momentum_squared():
    """velocity reaches look_ahead only through new_velocity, picking up one
    factor of momentum at each step, so d(params)/d(velocity) = -lr*m^2. The
    rule returned -lr*m*(1+m) — 2.11x too large at m=0.9 — while the native
    x86/ROCm backward helper already used the correct form."""
    lr, m = 0.1, 0.9
    params = np.array([1.0, 2.0])
    grads = np.array([0.3, -0.2])
    velocity = np.array([0.5, 0.1])
    dout = np.ones(2)

    _, _, d_state = _VJPS["nesterov"](
        dout, params, grads, {"velocity": velocity}, lr=lr, momentum=m)
    np.testing.assert_allclose(d_state["velocity"], -lr * m * m * dout, rtol=1e-12)


# ── schedules, clipping, losses ────────────────────────────────────────────

def test_inverse_sqrt_lr_peaks_at_init_value_and_warms_up():
    """The old form returned init_value*sqrt(warmup)/sqrt(step) at every step,
    so the rate EXCEEDED init_value for the whole warmup — 63x the nominal peak
    on step 1 at the usual warmup_steps=4000."""
    from tessera import optim

    init_value, warmup = 1e-3, 4000
    rates = [optim.inverse_sqrt_lr(s, init_value=init_value, warmup_steps=warmup)
             for s in range(1, 20001)]
    assert max(rates) <= init_value + 1e-18
    # Peak sits exactly at the end of warmup, and the schedule is continuous.
    assert optim.inverse_sqrt_lr(warmup, init_value=init_value,
                                 warmup_steps=warmup) == pytest.approx(init_value)
    assert optim.inverse_sqrt_lr(1, init_value=init_value,
                                 warmup_steps=warmup) < init_value
    # ...and still decays as sqrt after it.
    assert optim.inverse_sqrt_lr(4 * warmup, init_value=init_value,
                                 warmup_steps=warmup) == pytest.approx(init_value / 2)


def test_clip_grad_norm_honours_norm_type():
    """Every norm_type other than inf silently fell through to the L2 norm, so
    norm_type=1.0 clipped by — and reported — the wrong quantity."""
    from tessera import optim

    grads = [np.array([3.0]), np.array([4.0])]
    assert optim.clip_grad_norm(grads, 1.0, norm_type=2.0)[1] == pytest.approx(5.0)
    assert optim.clip_grad_norm(grads, 1.0, norm_type=1.0)[1] == pytest.approx(7.0)
    assert optim.clip_grad_norm(grads, 1.0, norm_type=float("inf"))[1] == pytest.approx(4.0)
    # A norm this cannot compute fails closed rather than substituting L2.
    with pytest.raises(ValueError):
        optim.clip_grad_norm(grads, 1.0, norm_type=0.0)


def test_focal_loss_honours_ignore_index_and_range_checks():
    """Targets were fancy-indexed unchecked, so numpy's negative wraparound
    turned the -100 padding convention that cross_entropy_loss documents into a
    real class index and averaged its probability into the loss."""
    from tessera import losses

    rng = np.random.default_rng(0)
    logits = rng.standard_normal((4, 10))
    padded = losses.focal_loss(logits, np.array([1, 2, -100, 3]))
    unpadded = losses.focal_loss(logits[[0, 1, 3]], np.array([1, 2, 3]))
    assert float(padded) == pytest.approx(float(unpadded))

    with pytest.raises(ValueError):
        losses.focal_loss(logits, np.array([1, 2, 99, 3]))


def test_kl_divergence_is_finite_when_p_has_zero_probability_entries():
    """p_log = -inf gives p = 0 and 0*(-inf) = NaN, which poisoned the whole
    reduction. The true contribution of a zero-probability term is 0. -inf
    arrives routinely from log_softmax over masked logits."""
    from tessera import losses

    logits = np.array([[1.0, 2.0, -np.inf]])
    p_log = logits - np.log(np.exp(logits[np.isfinite(logits)]).sum())
    q = np.array([[0.3, 0.6, 0.1]])

    value = float(losses.kl_divergence(p_log, q))
    assert np.isfinite(value)
    p = np.exp(p_log)
    expected = float(np.sum(np.where(p > 0, p * (p_log - np.log(q)), 0.0)))
    assert value == pytest.approx(expected)


# ── transform / tape plumbing ──────────────────────────────────────────────

def test_jacrev_refuses_an_output_its_tape_never_produced():
    """The structural-resolution branch conflated 'fn returned a constant' with
    'fn's tail ran in raw numpy on top of taped ops'. In the second case the
    gradient path is real but unreachable, and returning the pre-zeroed buffer
    reported a zero Jacobian for a function that is not constant."""
    from tessera import ops
    from tessera.autodiff import jacrev
    from tessera.autodiff.tape import TesseraAutodiffError

    # Still provable, still resolved structurally.
    np.testing.assert_allclose(jacrev(lambda x: x)(np.array([1.0, 2.0])), np.eye(2))
    np.testing.assert_allclose(
        jacrev(lambda x: ops.mul(x, 2.0))(np.array([1.0, 2.0])), 2.0 * np.eye(2))

    with pytest.raises(TesseraAutodiffError, match="outside the tape"):
        jacrev(lambda x: np.tanh(ops.matmul(x, np.eye(3))))(np.array([0.1, 0.2, 0.3]))


def test_record_custom_vjp_call_passes_scalar_positionals_through():
    """`_describe` coerces a python int to a float64 0-d array so ordinary ops
    can treat it as a numeric operand. That is wrong for this entry point, whose
    contract is that non-array positionals reach `forward` unchanged: an
    `axis=1` argument arrived as np.float64(1.0) and numpy refused to index
    with it."""
    from tessera.autodiff.tape import record_custom_vjp_call, tape

    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    def forward(a, axis):
        return np.sum(a, axis=axis)

    def rule(dout, a):  # only ARRAY inputs are recorded, per the contract
        return (np.broadcast_to(np.expand_dims(dout, 1), a.shape).copy(),)

    # Works with no tape active...
    np.testing.assert_allclose(
        record_custom_vjp_call("mysum", forward, rule, x, 1), [3.0, 7.0])

    # ...and identically inside one, with the gradient flowing.
    with tape() as t:
        out = record_custom_vjp_call("mysum", forward, rule, x, 1)
        np.testing.assert_allclose(out, [3.0, 7.0])
        t.backward(out, cotangent=np.ones(2))
    np.testing.assert_allclose(t.cotangent[id(x)], np.ones((2, 2)))
