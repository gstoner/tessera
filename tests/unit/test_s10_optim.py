"""S10 optimizer, schedule, and gradient-transform coverage."""

from __future__ import annotations

import numpy as np

import tessera as ts
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
