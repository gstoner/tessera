"""S11 losses and S12 state serialization/checkpointing."""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


def test_regression_and_classification_losses():
    pred = np.array([1.0, 3.0, 5.0])
    target = np.array([1.0, 1.0, 2.0])
    assert ts.losses.mse_loss(pred, target) == pytest.approx((0.0 + 4.0 + 9.0) / 3.0)
    assert ts.losses.mae_loss(pred, target) == pytest.approx(5.0 / 3.0)
    assert ts.losses.huber_loss(pred, target, delta=1.0, reduction="sum") == pytest.approx(0.0 + 1.5 + 2.5)
    assert ts.losses.smooth_l1_loss(pred, target, beta=1.0, reduction="sum") == pytest.approx(0.0 + 1.5 + 2.5)
    assert ts.losses.log_cosh_loss(pred, target) >= 0.0

    logits = np.array([[2.0, 0.0, -1.0], [0.0, 3.0, -2.0]])
    targets = np.array([0, 1])
    ce = ts.losses.cross_entropy_loss(logits, targets)
    assert ce < 0.2
    assert ts.losses.binary_cross_entropy_loss(np.array([0.0]), np.array([1.0])) == pytest.approx(np.log(2.0))
    assert ts.losses.focal_loss(logits, targets) < ce
    assert ts.losses.label_smoothed_cross_entropy(logits, targets, smoothing=0.1) > ce


def test_distribution_contrastive_diffusion_and_sequence_losses():
    p = np.log(np.array([[0.75, 0.25]], dtype=np.float64))
    q = np.array([[0.5, 0.5]], dtype=np.float64)
    assert ts.losses.kl_divergence(p, q) > 0.0
    assert ts.losses.js_divergence(np.exp(p), q) > 0.0
    assert ts.losses.wasserstein_distance(np.array([[0.0, 2.0]]), np.array([[1.0, 3.0]])) == pytest.approx(1.0)

    anchor = np.array([[0.0, 0.0]])
    positive = np.array([[0.1, 0.0]])
    negative = np.array([[2.0, 0.0]])
    assert ts.losses.triplet_loss(anchor, positive, negative, margin=0.5) == pytest.approx(0.0)
    assert ts.losses.contrastive_loss(anchor, positive, np.array([1.0])) > 0.0
    assert ts.losses.cosine_embedding_loss(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([1.0])) == pytest.approx(0.0, abs=1e-10)

    query = np.array([[1.0, 0.0]])
    pos = np.array([[1.0, 0.0]])
    neg = np.array([[[0.0, 1.0], [-1.0, 0.0]]])
    assert ts.losses.info_nce_loss(query, pos, neg) < 1e-3
    embeddings = np.eye(4, dtype=np.float64)
    labels = np.array([0, 0, 1, 1])
    assert np.isfinite(ts.losses.nt_xent_loss(embeddings, labels))

    assert ts.losses.ddpm_noise_pred_loss(np.array([1.0]), np.array([0.0])) == pytest.approx(1.0)
    assert ts.losses.score_matching_loss(np.array([1.0]), np.array([0.0])) == pytest.approx(0.5)
    assert ts.losses.vlb_loss(np.array([1.0, 2.0]), reduction="sum") == pytest.approx(3.0)

    logits = np.array([[[3.0, 0.0], [0.0, 3.0]]])
    targets = np.array([[0, 1]])
    mask = np.array([[1.0, 0.0]])
    assert ts.losses.seq2seq_loss(logits, targets, mask=mask) < 0.1


def test_ctc_loss_single_batch_reference():
    probs = np.array([
        [[0.1, 0.8, 0.1]],
        [[0.7, 0.2, 0.1]],
        [[0.1, 0.1, 0.8]],
    ], dtype=np.float64)
    log_probs = np.log(probs)
    loss = ts.losses.ctc_loss(log_probs, np.array([[1, 2]]), np.array([3]), np.array([2]), blank=0)
    assert np.isfinite(loss)
    assert loss < 1.0


def test_save_load_state_partial_and_migration(tmp_path):
    state = {
        "params": {"w": np.arange(6, dtype=np.float32).reshape(2, 3)},
        "optimizer_slots": {"m": np.ones(3, dtype=np.float32)},
        "metrics": {"step": np.array(3, dtype=np.int64)},
    }
    path = tmp_path / "state.tessera.npz"
    ts.checkpoint.save_state(state, path, version=1, metadata={"model": "tiny"})
    loaded = ts.checkpoint.load_state(path)
    np.testing.assert_array_equal(loaded["params"]["w"], state["params"]["w"])
    np.testing.assert_array_equal(loaded["optimizer_slots"]["m"], state["optimizer_slots"]["m"])

    partial = ts.checkpoint.load_state(path, collections=("params",))
    assert set(partial) == {"params"}

    @ts.checkpoint.state_migration(1, 2)
    def add_version(tree):
        tree = dict(tree)
        tree["metrics"] = dict(tree["metrics"])
        tree["metrics"]["version"] = np.array(2, dtype=np.int64)
        return tree

    migrated = ts.checkpoint.load_state(path, target_version=2)
    assert migrated["metrics"]["version"] == 2


def test_save_load_state_detects_checksum_mismatch(tmp_path):
    state = {"params": {"w": np.array([1.0, 2.0], dtype=np.float32)}}
    path = tmp_path / "state.tessera.npz"
    ts.checkpoint.save_state(state, path)
    with np.load(path, allow_pickle=False) as data:
        payload = {k: np.array(data[k]) for k in data.files}
    payload["leaf_0"][0] = 99.0
    with path.open("wb") as f:
        np.savez(f, **payload)
    with pytest.raises(ts.checkpoint.CheckpointError, match="checksum mismatch"):
        ts.checkpoint.load_state(path)


def test_save_load_sharded_state_round_trip(tmp_path):
    mesh = ts.NamedMesh(("dp",), (2,))
    state = {"params": {"w": np.array([[1.0, 2.0]], dtype=np.float32)}}
    root = ts.checkpoint.save_sharded(state, tmp_path / "sharded", mesh)
    loaded = ts.checkpoint.load_sharded(root, mesh)
    np.testing.assert_array_equal(loaded["params"]["w"], state["params"]["w"])
