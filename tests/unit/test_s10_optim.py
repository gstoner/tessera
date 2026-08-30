"""S10 optimizer, schedule, and gradient-transform coverage."""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera import optim
from tessera.state import tree_flatten, tree_unflatten


def test_sgd_momentum_nesterov_update_nested_trees():
    params = {"w": np.array([1.0, -2.0]), "b": np.array([0.5])}
    grads = {"w": np.array([0.25, -0.5]), "b": np.array([1.0])}

    out = ts.optim.sgd(params, grads, lr=0.1)
    np.testing.assert_allclose(out["w"], [0.975, -1.95])
    np.testing.assert_allclose(out["b"], [0.4])

    out_m, state_m = ts.optim.momentum(params, grads, lr=0.1, momentum=0.9)
    np.testing.assert_allclose(out_m["w"], out["w"])
    out_m2, _ = ts.optim.momentum(out_m, grads, state_m, lr=0.1, momentum=0.9)
    np.testing.assert_allclose(out_m2["w"], out_m["w"] - 0.1 * (0.9 * grads["w"] + grads["w"]))

    out_n, state_n = ts.optim.nesterov(params, grads, lr=0.1, momentum=0.9)
    assert state_n["velocity"]["w"].shape == params["w"].shape
    assert np.linalg.norm(out_n["w"] - params["w"]) > np.linalg.norm(out["w"] - params["w"])


def test_adamw_lion_lamb_and_adafactor_state_shapes():
    params = {"w": np.ones((2, 3), dtype=np.float32), "b": np.ones(3, dtype=np.float32)}
    grads = {"w": np.full((2, 3), 0.1, dtype=np.float32), "b": np.full(3, 0.2, dtype=np.float32)}

    adam_params, adam_state = ts.optim.adamw(params, grads, lr=0.01, weight_decay=0.1)
    assert adam_state["step"] == 1
    assert adam_params["w"].shape == params["w"].shape
    assert np.all(adam_params["w"] < params["w"])
    adam_plain, plain_state = ts.optim.adam(params, grads, lr=0.01)
    assert plain_state["step"] == 1
    assert np.all(adam_plain["w"] > adam_params["w"])

    ada_params, ada_state = ts.optim.adafactor(params, grads, lr=0.01)
    assert ada_state["v"]["w"]["factored"] is True
    assert ada_state["v"]["w"]["row"].shape == (2,)
    assert ada_state["v"]["w"]["col"].shape == (3,)
    assert ada_params["b"].shape == params["b"].shape

    lion_params, lion_state = ts.optim.lion(params, grads, lr=0.01)
    assert lion_state["step"] == 1
    np.testing.assert_allclose(lion_params["b"], np.full(3, 0.99))

    lamb_params, lamb_state = ts.optim.lamb(params, grads, lr=0.01)
    assert lamb_state["step"] == 1
    assert lamb_params["w"].shape == params["w"].shape


def test_muon_orthogonalizes_matrix_updates():
    params = {"w": np.ones((2, 2), dtype=np.float32)}
    grads = {"w": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
    out, state = ts.optim.muon(params, grads, lr=0.1)
    update = (params["w"] - out["w"]) / 0.1
    np.testing.assert_allclose(update @ update.T, np.eye(2), atol=1e-5)
    assert state["velocity"]["w"].shape == (2, 2)


def test_schedules_match_expected_values():
    assert ts.optim.constant_lr(0.3)(100) == 0.3
    assert ts.optim.cosine_lr(0, init_value=1.0, end_value=0.0, decay_steps=10) == 1.0
    assert ts.optim.cosine_lr(10, init_value=1.0, end_value=0.0, decay_steps=10) == 0.0
    assert ts.optim.cosine_warmup_lr(2, peak_value=1.0, warmup_steps=4, decay_steps=10) == 0.5
    assert ts.optim.linear_warmup_lr(5, peak_value=1.0, warmup_steps=10) == 0.5
    assert ts.optim.polynomial_lr(10, init_value=1.0, end_value=0.2, decay_steps=10) == 0.2
    np.testing.assert_allclose(ts.optim.inverse_sqrt_lr(4, init_value=2.0, warmup_steps=1), 1.0)
    assert ts.optim.cyclical_lr(5, base_value=0.1, max_value=1.0, step_size=5) == 1.0
    chained = ts.optim.chained_schedule(ts.optim.constant_lr(0.1), ts.optim.constant_lr(0.2))
    assert chained(7) == (0.1, 0.2)


def test_gradient_transforms_and_chain():
    grads = {"w": np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32), "b": np.array([10.0], dtype=np.float32)}
    clipped, total = ts.optim.clip_grad_norm(grads, max_norm=5.0)
    assert total > 5.0
    np.testing.assert_allclose(ts.optim.tree_l2_norm(clipped), 5.0, atol=1e-6)

    valued = ts.optim.clip_grad_value(grads, 2.0)
    assert valued["w"].max() == 2.0 and valued["b"][0] == 2.0

    centered = ts.optim.centralize_grad({"w": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)})
    np.testing.assert_allclose(centered["w"].mean(axis=0), [0.0, 0.0])

    params = {"w": np.ones((2, 2), dtype=np.float32)}
    decayed = ts.optim.add_decoupled_weight_decay({"w": np.zeros((2, 2), dtype=np.float32)}, params, 0.1)
    np.testing.assert_allclose(decayed["w"], 0.1)

    transform = ts.optim.chain(
        lambda updates, p: ts.optim.clip_grad_value(updates, 1.0),
        lambda updates, p: ts.optim.add_decoupled_weight_decay(updates, p, 0.5),
    )
    chained = transform({"w": np.full((2, 2), 3.0)}, params)
    np.testing.assert_allclose(chained["w"], 1.5)


def test_ema_polyak_and_optimizer_state_tree_round_trip():
    params = {"w": np.array([2.0, 4.0], dtype=np.float32)}
    ema = {"w": np.array([0.0, 0.0], dtype=np.float32)}
    np.testing.assert_allclose(ts.optim.ema_update(ema, params, decay=0.5)["w"], [1.0, 2.0])
    np.testing.assert_allclose(ts.optim.polyak_avg(ema, params, step=1)["w"], [1.0, 2.0])

    _, state = ts.optim.adamw(params, {"w": np.array([0.1, 0.2], dtype=np.float32)}, lr=0.01)
    leaves, treedef = tree_flatten(state)
    restored = tree_unflatten(treedef, leaves)
    assert restored["step"] == state["step"]
    np.testing.assert_allclose(restored["m"]["w"], state["m"]["w"])


# ─────────────────────────────────────────────────────────────────────────────
# Adafactor second-moment bias correction (CODE_REVIEW_2026-08-29,
# `python/tessera/optim.py:427`).  The tracked `step` used to be incremented and
# never read, so the zero-initialized second moment was biased low and the first
# ~1/(1-beta2) updates were inflated by 1/sqrt(1 - beta2**step) — 31.6x at the
# default beta2=0.999.  `optim.adafactor_decay` is the single shared correction;
# these tests pin it against an independently written reference so a regression
# in either the tree form, the flat compiler ABI, or the analytic VJP shows up.


def _uncorrected_adafactor_full(g, *, beta2, eps, steps):
    """Textbook Adafactor full-moment update with an EXPLICIT 1-beta2**t debias.

    Written directly from the definition (raw EMA, then divide by 1-beta2**t)
    rather than through `adafactor_decay`, so it is an independent check of the
    step-dependent-decay formulation rather than a restatement of it.
    """
    v = np.zeros_like(g[0], dtype=np.float64)
    updates = []
    for t in range(1, steps + 1):
        gt = np.asarray(g[t - 1], dtype=np.float64)
        v = beta2 * v + (1.0 - beta2) * gt * gt
        v_hat = v / (1.0 - beta2**t)
        updates.append(gt / (np.sqrt(np.maximum(v_hat, eps)) + eps))
    return updates


def test_adafactor_step_one_update_is_not_inflated_by_the_ema_bias():
    """A constant gradient must give a ~1.0 normalized update from step 1."""
    grad = np.full((4, 4), 0.1, dtype=np.float32)
    params, state = ts.optim.adafactor(
        {"w": np.ones((4, 4), dtype=np.float32)}, {"w": grad}, lr=1.0
    )
    magnitude = float(np.abs(1.0 - np.asarray(params["w"])).mean())
    # Pre-fix this was 1/sqrt(1-0.999) = 31.6227...
    assert magnitude == pytest.approx(1.0, rel=1e-4), magnitude
    assert state["step"] == 1


def test_adafactor_matches_explicit_bias_corrected_ema_across_steps():
    """Full-moment (rank-1) leaves must track the explicit 1-beta2**t debias."""
    rng = np.random.default_rng(20260830)
    beta2, eps, lr, steps = 0.999, 1e-30, 1.0, 6
    grads = [rng.normal(scale=0.3, size=(7,)).astype(np.float32) for _ in range(steps)]
    expected = _uncorrected_adafactor_full(grads, beta2=beta2, eps=eps, steps=steps)

    params, state = {"w": np.zeros(7, dtype=np.float32)}, None
    for index, grad in enumerate(grads):
        previous = np.asarray(params["w"], dtype=np.float64)
        params, state = ts.optim.adafactor(
            params, {"w": grad}, state, lr=lr, beta2=beta2, eps=eps
        )
        applied = (previous - np.asarray(params["w"], dtype=np.float64)) / lr
        np.testing.assert_allclose(applied, expected[index], rtol=2e-5, atol=2e-6)
        assert state["step"] == index + 1


def test_adafactor_decay_is_the_debiasing_recursion_and_fails_closed():
    beta2 = 0.9
    # t=1 must discard the (empty) prior so v_1 == g_1**2 exactly.
    assert ts.optim.adafactor_decay(beta2, 1) == 0.0
    # The recursion must reproduce EMA_t / (1 - beta2**t) exactly.
    raw, corrected = 0.0, 0.0
    for t in range(1, 40):
        g2 = float(t) ** 2
        raw = beta2 * raw + (1.0 - beta2) * g2
        decay = ts.optim.adafactor_decay(beta2, t)
        corrected = decay * corrected + (1.0 - decay) * g2
        assert corrected == pytest.approx(raw / (1.0 - beta2**t), rel=1e-12)
    # Asymptotically the caller's beta2 is preserved, not replaced.
    assert ts.optim.adafactor_decay(beta2, 500) == pytest.approx(beta2, rel=1e-9)
    # Semantic keys fail closed rather than defaulting (Decision #21a).
    for bad_step in (0, -1):
        with pytest.raises(ValueError, match="1-based"):
            ts.optim.adafactor_decay(beta2, bad_step)
    for bad_beta2 in (1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match=r"beta2"):
            ts.optim.adafactor_decay(bad_beta2, 3)


def test_flat_adafactor_abi_carries_the_step_like_flat_adam():
    """The flat compiler ABI must agree with the tree form at every step, not
    only at step 1 where the correction happens to zero the carried state."""
    rng = np.random.default_rng(90210)
    shape = (3, 5)
    kwargs = {"lr": 0.003, "beta2": 0.91, "eps": 1.0e-7}
    p = rng.normal(size=shape).astype(np.float32)
    row = np.zeros(shape[:-1], np.float32)
    col = np.zeros(shape[-1], np.float32)
    tree_params, tree_state = {"w": p}, None
    for step in range(1, 5):
        g = rng.normal(scale=0.3, size=shape).astype(np.float32)
        p, row, col = ts.ops.adafactor(p, g, row, col, step=step, **kwargs)
        tree_params, tree_state = ts.optim.adafactor(
            tree_params, {"w": g}, tree_state, **kwargs
        )
        assert tree_state["step"] == step
        np.testing.assert_allclose(p, tree_params["w"], rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(row, tree_state["v"]["w"]["row"], rtol=2e-6)
        np.testing.assert_allclose(col, tree_state["v"]["w"]["col"], rtol=2e-6)
    # A declared step that contradicts the tree state must not be silently
    # resolved in favour of either side.
    with pytest.raises(ValueError, match="disagrees with the carried state"):
        ts.ops.adafactor(tree_params, {"w": g}, tree_state, step=99, **kwargs)


def test_adafactor_vjp_differentiates_the_corrected_forward():
    """The analytic VJP must track the step-dependent decay, not nominal beta2.

    (A previous batch shipped a defect by fixing an eager path while its VJP
    kept the old behaviour; this pins the pair together.)"""
    from tessera.autodiff.vjp import get_vjp

    rng = np.random.default_rng(4242)
    shape = (17,)
    kwargs = {"lr": 0.003, "beta2": 0.91, "eps": 1.0e-7}
    p = rng.normal(size=shape).astype(np.float32)
    g = rng.normal(scale=0.2, size=shape).astype(np.float32)
    dy = rng.normal(size=shape).astype(np.float32)
    for carried in (0, 1, 5):
        state = {
            "v": {
                "v": rng.uniform(0.1, 0.3, size=shape).astype(np.float32),
                "factored": False,
            },
            "step": carried,
        }
        analytic = get_vjp("adafactor")(dy, p, g, state, **kwargs)

        def forward(gradient):
            return np.asarray(
                ts.optim.adafactor(p, gradient, state, **kwargs)[0],
                dtype=np.float64,
            )

        numeric = np.zeros(shape, dtype=np.float64)
        h = 1e-3
        for index in range(shape[0]):
            bump = np.zeros(shape, dtype=np.float64)
            bump[index] = h
            plus = forward((g + bump).astype(np.float32))
            minus = forward((g - bump).astype(np.float32))
            numeric[index] = float(
                np.sum(np.asarray(dy, dtype=np.float64) * (plus - minus)) / (2 * h)
            )
        # The forward computes in fp32, so central differences sit on a ~1e-4
        # noise floor (the same reason `jvp_adafactor` pins h=1e-3).  That is
        # still two orders tighter than the error an uncorrected decay causes:
        # see `test_adafactor_vjp_rejects_the_nominal_decay` below.
        np.testing.assert_allclose(
            np.asarray(analytic[1], dtype=np.float64), numeric, rtol=2e-2, atol=1e-4
        )


def test_adafactor_vjp_rejects_the_nominal_decay():
    """Discrimination check for the test above: had the VJP kept differentiating
    the *nominal* beta2 (the pre-fix behaviour), the gradient would be wrong by
    far more than the fp32 finite-difference noise floor."""
    from tessera.autodiff import vjp as vjp_module
    from tessera.autodiff.vjp import get_vjp

    rng = np.random.default_rng(4242)
    shape = (17,)
    kwargs = {"lr": 0.003, "beta2": 0.91, "eps": 1.0e-7}
    p = rng.normal(size=shape).astype(np.float32)
    g = rng.normal(scale=0.2, size=shape).astype(np.float32)
    dy = rng.normal(size=shape).astype(np.float32)
    state = {
        "v": {
            "v": rng.uniform(0.1, 0.3, size=shape).astype(np.float32),
            "factored": False,
        },
        "step": 1,
    }
    corrected = np.asarray(get_vjp("adafactor")(dy, p, g, state, **kwargs)[1])

    original = vjp_module.adafactor_decay if hasattr(vjp_module, "adafactor_decay") else None
    del original
    import tessera.optim as optim_module

    saved = optim_module.adafactor_decay
    try:  # pre-fix behaviour: a fixed, uncorrected decay
        optim_module.adafactor_decay = lambda beta2, step: float(beta2)
        uncorrected = np.asarray(get_vjp("adafactor")(dy, p, g, state, **kwargs)[1])
    finally:
        optim_module.adafactor_decay = saved
    relative = float(
        np.max(np.abs(uncorrected - corrected) / (np.abs(corrected) + 1e-12))
    )
    assert relative > 0.05, relative


# --- PR #644 review: the two Adafactor ABI/representation seams ---------------

def test_flat_adafactor_without_step_preserves_the_moments_it_was_given():
    """`adafactor_decay(b2, 1)` is exactly 0 -- right for a genuine first step,
    where v_1 = g^2. Defaulting an ABSENT step to 1 would therefore make every
    call of a stateful caller that never passes one discard the moments it just
    supplied, turning a stateful optimizer stateless with no diagnostic."""
    import tessera as ts

    assert optim.adafactor_decay(0.999, 1) == 0.0
    param = np.ones((4, 4), np.float32)
    grad = np.full((4, 4), 0.5, np.float32)
    carried = np.full((4, 4), 9.0, np.float32)

    no_step = np.asarray(ts.ops.adafactor(param, grad, carried)[1])
    first_step = np.asarray(ts.ops.adafactor(param, grad, carried, step=1)[1])

    # Omitting step keeps the carried EMA (legacy, uncorrected) ...
    assert float(no_step.ravel()[0]) > 8.0
    # ... while an explicit step 1 correctly restarts from g^2.
    assert float(first_step.ravel()[0]) == pytest.approx(0.25)


def test_unmarked_adafactor_state_warns_rather_than_being_misread():
    """`state["v"]` changed meaning: it now carries the debiased estimate, not
    the raw EMA. The two cannot be told apart from the values, so an unmarked
    state at step > 0 is flagged. It is deliberately NOT rescaled: auto-
    migrating would silently rewrite every hand-built state dict, which is a
    worse failure than the one it fixes."""
    params = {"w": np.ones((4, 4), np.float32)}
    grads = {"w": np.full((4, 4), 0.5, np.float32)}
    _, marked = optim.adafactor(params, grads)
    assert marked["v_representation"] == optim._ADAFACTOR_V_REPRESENTATION

    _, marked_2 = optim.adafactor(params, grads, marked)
    unmarked = {k: v for k, v in marked.items() if k != "v_representation"}
    with pytest.warns(RuntimeWarning, match="v_representation"):
        _, from_unmarked = optim.adafactor(params, grads, unmarked)
    # Warned, but the values are untouched -- identical to the marked run.
    np.testing.assert_array_equal(
        np.asarray(from_unmarked["v"]["w"]["row"]),
        np.asarray(marked_2["v"]["w"]["row"]),
    )


def test_explicit_migration_recovers_a_legacy_checkpoint_and_is_idempotent():
    params = {"w": np.ones((4, 4), np.float32)}
    grads = {"w": np.full((4, 4), 0.5, np.float32)}
    _, native_1 = optim.adafactor(params, grads)
    _, native_2 = optim.adafactor(params, grads, native_1)

    # Forge the pre-correction representation: same step, v scaled back by the
    # bias factor the old code left in.
    bias = 1.0 - 0.999 ** int(native_1["step"])
    legacy = {k: v for k, v in native_1.items() if k != "v_representation"}
    legacy["v"] = optim._adafactor_tree_map_unary(
        lambda slot: optim._adafactor_scale_state(slot, bias, state_dtype="fp32"),
        native_1["v"],
    )

    migrated = optim.migrate_adafactor_state(legacy, 0.999)
    _, resumed = optim.adafactor(params, grads, migrated)
    np.testing.assert_allclose(
        np.asarray(resumed["v"]["w"]["row"]),
        np.asarray(native_2["v"]["w"]["row"]),
        rtol=1e-5,
    )
    # Migrating an already-marked state is a no-op.
    assert optim.migrate_adafactor_state(migrated, 0.999) is migrated
