"""Expanded S8 conformance using S10-S15 training-step pieces."""

from __future__ import annotations

import numpy as np

import tessera as ts


def test_tiny_training_step_with_data_loss_optimizer_and_checkpoint(tmp_path):
    xs = np.arange(6, dtype=np.float64).reshape(3, 2)
    true_w = np.array([[2.0], [-1.0]], dtype=np.float64)
    ys = xs @ true_w

    ds = ts.data.Dataset.from_tensor_slices({"x": xs, "y": ys}).batch(3)
    batch = ds.to_list()[0]

    def loss_fn(w):
        pred = ts.ops.matmul(batch["x"], w)
        err = ts.ops.sub(pred, batch["y"])
        return ts.ops.mean(ts.ops.mul(err, err))

    w0 = np.zeros((2, 1), dtype=np.float64)
    loss0, grad = ts.value_and_grad(loss_fn)(w0)
    assert loss0 > 0.0
    assert grad.shape == w0.shape

    params = {"w": w0}
    grads = {"w": grad}
    clipped, total_norm = ts.optim.clip_grad_norm(grads, max_norm=10.0)
    assert total_norm > 0.0
    params1 = ts.optim.sgd(params, clipped, lr=0.05)
    assert ts.losses.mse_loss(batch["x"] @ params1["w"], batch["y"]) < loss0

    state = {
        "params": params1,
        "optimizer_slots": {"grad_norm": np.array(total_norm)},
        "metrics": {"loss": np.asarray(loss0)},
        "rng_state": {"data": ts.rng.RNGKey.from_seed(3).to_state()},
    }
    ckpt = tmp_path / "tiny_training_state.npz"
    ts.checkpoint.save_state(state, ckpt)
    loaded = ts.checkpoint.load_state(ckpt, collections=("params", "metrics"))
    np.testing.assert_allclose(loaded["params"]["w"], params1["w"])
    np.testing.assert_allclose(loaded["metrics"]["loss"], loss0)


def test_tiny_model_family_smokes_with_s15_inputs():
    tokens = ts.data.Dataset.from_tensor_slices(np.arange(8, dtype=np.float32).reshape(4, 2)).batch(2).to_list()[0]
    W = np.ones((2, 2), dtype=np.float32) * 0.25
    linformer_like = ts.nn.linear_general(tokens, W)
    assert linformer_like.shape == (2, 2)

    seq = np.asarray(tokens, dtype=np.float64)

    def recurrent_loss(x):
        def step(carry, item):
            carry = ts.ops.add(carry, item)
            return carry, carry

        final, _ = ts.scan(step, np.zeros(2, dtype=np.float64), x)
        return ts.ops.mean(ts.ops.mul(final, final))

    grad = ts.autodiff.grad(recurrent_loss)(seq)
    assert grad.shape == seq.shape

    text = "hello world"
    vocab = {"hello": 1, "world": 2}
    ids = np.asarray(ts.data.tokenizer_bpe(vocab).encode(text), dtype=np.float32)
    assert ids.tolist() == [1.0, 2.0]
