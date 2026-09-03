"""A weighted mean must not depend on the scale of its weights.

The RL masked reductions divided by ``max(sum(mask), 1.0)``, which clamps
every mask sum below 1 — so a float mask like ``[0.1, 0.1]`` normalised by
1.0 instead of 0.2 and returned one fifth of the weighted mean. Rescaling
otherwise-equivalent weights moved the training loss.

Same defect PR #697 fixed for `seq2seq_loss`, in a different family. It
stayed latent because masks are usually 0/1 with at least one entry set,
and because every existing test used exactly that shape.

**Nothing restricted the mask to 0/1**, which is what makes this a bug
rather than an undocumented precondition: `PYTHON_API_SPEC` says only
"optional mask", the argument is typed `Any`, and `rl._reduce` casts it to
float64 and multiplies without validation.

The clamp's one defensible job — avoiding 0/0 on a fully masked input — is
kept explicitly, and returns the same 0.0 it produced before, so only the
fractional case changes.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera import rl


RNG = np.random.default_rng(0)
LOGP_NEW = RNG.standard_normal(4)
LOGP_OLD = RNG.standard_normal(4)
ADVANTAGES = RNG.standard_normal(4)
BASE_MASK = np.array([1.0, 1.0, 0.0, 1.0])
SCALES = (0.05, 0.1, 0.25, 0.5, 1.0, 4.0, 100.0)


def _ppo(mask, **kw):
    return float(rl.ppo_policy_loss(LOGP_NEW, LOGP_OLD, ADVANTAGES, mask=mask, **kw))


# --- the invariance itself ------------------------------------------------


@pytest.mark.parametrize("scale", SCALES)
def test_ppo_loss_is_invariant_to_the_scale_of_the_mask(scale):
    assert _ppo(BASE_MASK * scale) == pytest.approx(_ppo(BASE_MASK), rel=1e-12)


def test_the_sub_unit_scales_are_the_ones_that_used_to_break():
    """Guards the parametrisation above from going quietly vacuous.

    Only sums BELOW 1 hit the clamp, so a sweep of scales >= 1 would have
    passed against the old code too. Pin that this fixture actually reaches
    the broken regime.
    """
    sums = [float(np.sum(BASE_MASK * s)) for s in SCALES]
    assert any(s < 1.0 for s in sums), "no scale drives the mask sum below 1"
    assert any(s >= 1.0 for s in sums), "no scale stays in the previously-correct regime"


@pytest.mark.parametrize("scale", (0.1, 0.25, 1.0, 10.0))
def test_grpo_and_cispo_scale_exactly_linearly_with_the_mask(scale):
    """`_reduce` is shared, so the fix must reach every loss that calls it —
    but for these two the invariance takes a different form, and the
    difference is real rather than a tolerance issue.

    When `mask` is passed, grpo/cispo route it through
    `normalize_group_advantages`, which returns advantages already
    MULTIPLIED by the mask. The mask therefore enters twice: once weighting
    the advantages and once as the reduction denominator. The reduction
    contributes `1/s` and the advantages contribute `s`, so the loss scales
    exactly linearly — `loss(s·m) == s·loss(m)` — rather than being
    invariant. That is pre-existing advantage-weighting semantics, not the
    clamp.

    The linearity is still the thing the clamp broke. Clamping the
    denominator UP (0.4 -> 1.0) makes the loss SMALLER: measured here,
    `loss(0.1·m)` came out 0.01919 against the correct 0.04798 — 0.40x, the
    ratio of the true mask sum to the clamp. So this fails on the old code
    exactly where it matters.
    """
    logp_new = RNG.standard_normal((2, 3))
    logp_old = RNG.standard_normal((2, 3))
    rewards = RNG.standard_normal((2, 3))
    mask = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    for fn in (rl.grpo_policy_loss, rl.cispo_policy_loss):
        ref = float(fn(logp_new, logp_old, rewards, mask=mask))
        got = float(fn(logp_new, logp_old, rewards, mask=mask * scale))
        assert got == pytest.approx(scale * ref, rel=1e-12), (
            f"{fn.__name__} is not linear in the mask scale")


def test_group_advantage_normalisation_is_scale_invariant():
    """The third site the task did not name: a PER-GROUP denominator.

    `normalize_group_advantages` clamps each group's mask sum, so a group
    whose weights summed below 1 got a wrong mean AND variance — moving
    every normalised advantage in that group. The output is multiplied by
    the mask, so scaling the mask by `s` must scale the result by exactly
    `s` and change nothing else.
    """
    rewards = RNG.standard_normal((2, 3))
    mask = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    base = np.asarray(rl.normalize_group_advantages(rewards, mask=mask))
    for scale in (0.1, 0.25, 4.0):
        scaled = np.asarray(rl.normalize_group_advantages(rewards, mask=mask * scale))
        np.testing.assert_allclose(scaled, base * scale, rtol=1e-12, atol=0)


# --- the zero case the clamp existed for ---------------------------------


def test_a_fully_masked_input_is_zero_not_a_division_by_zero():
    out = _ppo(np.zeros(4))
    assert out == 0.0 and np.isfinite(out)


def test_a_fully_masked_group_stays_finite():
    rewards = RNG.standard_normal((2, 3))
    mask = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    out = np.asarray(rl.normalize_group_advantages(rewards, mask=mask))
    assert np.all(np.isfinite(out))
    assert np.all(out[0] == 0.0), "a fully masked group contributes nothing"


# --- the runtime executor is a SECOND forward and must agree --------------


def test_the_runtime_executor_matches_the_reference_on_a_fractional_mask():
    """`runtime._ppo_policy_loss_np` reimplements the same loss.

    Fixing one denominator and not the other would leave the compiled lane
    and the Python reference disagreeing — visible only for a fractional
    mask, which nothing tested.
    """
    from tessera import runtime as rt

    mask = BASE_MASK * 0.1
    runtime_value = float(rt._ppo_policy_loss_np(
        np, LOGP_NEW, LOGP_OLD, ADVANTAGES, mask=mask))
    assert runtime_value == pytest.approx(_ppo(mask), rel=1e-12)


# --- forward and its derivatives must move together ----------------------


def test_ppo_vjp_matches_the_fixed_forward_on_a_fractional_mask():
    """Finite differences against the forward, on the regime that changed.

    In #697 the seq2seq denominator was duplicated in the VJP and JVP, and
    fixing the forward alone would have traded a wrong loss for a wrong
    gradient. Here the RL rules are NUMERIC — they differentiate the
    forward by calling it — so they inherit the fix rather than needing
    one. That is a claim about the code, so it gets checked rather than
    asserted.
    """
    from tessera.autodiff import vjp as _vjp

    mask = BASE_MASK * 0.1
    rule = _vjp._VJPS["ppo_policy_loss"]
    grad = np.asarray(rule(1.0, LOGP_NEW, LOGP_OLD, ADVANTAGES, mask=mask)[0],
                      dtype=np.float64)

    h = 1e-6
    numeric = np.zeros_like(LOGP_NEW)
    for i in range(LOGP_NEW.size):
        up, dn = LOGP_NEW.copy(), LOGP_NEW.copy()
        up[i] += h
        dn[i] -= h
        numeric[i] = (float(rl.ppo_policy_loss(up, LOGP_OLD, ADVANTAGES, mask=mask))
                      - float(rl.ppo_policy_loss(dn, LOGP_OLD, ADVANTAGES, mask=mask))) / (2 * h)
    np.testing.assert_allclose(grad, numeric, rtol=1e-4, atol=1e-7)


def test_ppo_jvp_matches_the_fixed_forward_on_a_fractional_mask():
    import importlib

    _jvp = importlib.import_module("tessera.autodiff.jvp")
    mask = BASE_MASK * 0.1
    tangent = RNG.standard_normal(LOGP_NEW.shape)

    zero = np.zeros_like(LOGP_NEW)
    primal, tan = _jvp._JVPS["ppo_policy_loss"](
        (LOGP_NEW, LOGP_OLD, ADVANTAGES), (tangent, zero, zero), mask=mask)
    assert float(primal) == pytest.approx(_ppo(mask), rel=1e-9)

    h = 1e-6
    numeric = (float(rl.ppo_policy_loss(LOGP_NEW + h * tangent, LOGP_OLD, ADVANTAGES, mask=mask))
               - float(rl.ppo_policy_loss(LOGP_NEW - h * tangent, LOGP_OLD, ADVANTAGES, mask=mask))) / (2 * h)
    assert float(tan) == pytest.approx(numeric, rel=1e-4, abs=1e-7)


def test_the_gradient_is_also_scale_invariant():
    """The loss and its gradient must both stop moving with the weights."""
    from tessera.autodiff import vjp as _vjp

    rule = _vjp._VJPS["ppo_policy_loss"]
    base = np.asarray(rule(1.0, LOGP_NEW, LOGP_OLD, ADVANTAGES, mask=BASE_MASK)[0])
    for scale in (0.1, 0.25, 4.0):
        scaled = np.asarray(
            rule(1.0, LOGP_NEW, LOGP_OLD, ADVANTAGES, mask=BASE_MASK * scale)[0])
        np.testing.assert_allclose(scaled, base, rtol=1e-6, atol=1e-9)


# --- the FOURTH mirror: a native MPSGraph kernel -------------------------


@pytest.mark.hardware_apple_gpu
def test_the_native_ppo_kernel_agrees_with_the_reference_on_a_fractional_mask():
    """`mpsg_run_ppo_policy_loss_ex_f32` computes this denominator too.

    I first reported that no native kernel did — from a grep that found
    nothing, which is not evidence of absence. It does: the MPSGraph build
    had `maximum(sumMask, 1)`, so the Apple lane would have kept returning
    the old mis-scaled loss on exactly the inputs this change fixes.

    The availability probe is the enforcement: it compares the kernel against
    `_ppo_policy_loss_np`, so a disagreement disables the lane instead of
    silently mis-scaling a training loss. Verified by reverting the kernel
    and rebuilding — availability went False, and True again with the fix.
    """
    from tessera import runtime as rt

    if not rt._apple_gpu_ppo_policy_loss_ex_available():
        pytest.skip("native Apple PPO kernel unavailable on this host")

    rng = np.random.default_rng(11)
    logp_new = rng.standard_normal(4).astype(np.float32)
    logp_old = rng.standard_normal(4).astype(np.float32)
    advantages = rng.standard_normal(4).astype(np.float32)
    for scale in (0.05, 0.1, 0.5, 1.0):
        mask = (BASE_MASK * scale).astype(np.float32)
        reference = float(rl.ppo_policy_loss(logp_new, logp_old, advantages, mask=mask))
        runtime_value = float(rt._ppo_policy_loss_np(
            np, logp_new, logp_old, advantages, mask=mask))
        assert runtime_value == pytest.approx(reference, rel=1e-6)


def test_the_availability_probe_exercises_a_sub_unit_mask_sum():
    """Without this the probe cannot see the defect it now guards.

    Every original probe used a binary mask summing to 2 — above the clamp,
    so the native kernel could be wrong and still report healthy. Pin that a
    probe mask now sums below 1.
    """
    import inspect
    from tessera import runtime as rt

    source = inspect.getsource(rt._apple_gpu_ppo_policy_loss_ex_available)
    assert "mask_frac" in source, "the fractional probe is gone"
    assert "0.1, 0.0, 0.1" in source, (
        "the fractional probe mask changed; it must still sum below 1.0 or it "
        "cannot reach the denominator clamp"
    )
