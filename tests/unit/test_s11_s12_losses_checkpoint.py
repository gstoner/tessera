"""S11 losses and S12 state serialization/checkpointing."""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
import tessera.losses as losses_module


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


def test_log_cosh_loss_is_even_in_the_error():
    """log(cosh(e)) is even, so same-magnitude errors must cost the same.

    The old ``e + log1p(exp(-2e))`` form is the |e|-branch identity: for
    e < -354.9 it overflowed float64 and returned inf.
    """
    assert ts.losses.log_cosh_loss(0.0, 400.0) == pytest.approx(
        ts.losses.log_cosh_loss(400.0, 0.0))
    assert ts.losses.log_cosh_loss(0.0, 400.0) == pytest.approx(
        400.0 - np.log(2.0))
    pred = np.array([-1000.0, -3.0, 0.0, 3.0, 1000.0])
    assert np.all(np.isfinite(
        ts.losses.log_cosh_loss(pred, 0.0, reduction="none")))


def test_cross_entropy_probability_targets_reject_dropped_semantic_keys():
    """label_smoothing / ignore_index are semantic keys with no meaning against
    a distribution target — they must be rejected, not silently ignored."""
    logits = np.array([[2.0, 1.0, 0.1]])
    soft = np.array([[1.0, 0.0, 0.0]])
    baseline = ts.losses.cross_entropy_loss(logits, soft)
    for smoothing in (0.5, 7.0):
        with pytest.raises(ValueError, match="label_smoothing"):
            ts.losses.cross_entropy_loss(logits, soft, label_smoothing=smoothing)
    with pytest.raises(ValueError, match="ignore_index"):
        ts.losses.cross_entropy_loss(logits, soft, ignore_index=0)
    # Out-of-range label_smoothing is now rejected on the integer path too,
    # from the same hoisted check.
    with pytest.raises(ValueError, match="label_smoothing"):
        ts.losses.cross_entropy_loss(logits, np.array([0]), label_smoothing=7.0)
    assert ts.losses.cross_entropy_loss(
        logits, soft, ignore_index=-100) == pytest.approx(baseline)


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
    loaded = ts.checkpoint.load_state(path, trust_treedef=True)
    np.testing.assert_array_equal(loaded["params"]["w"], state["params"]["w"])
    np.testing.assert_array_equal(loaded["optimizer_slots"]["m"], state["optimizer_slots"]["m"])

    partial = ts.checkpoint.load_state(path, collections=("params",), trust_treedef=True)
    assert set(partial) == {"params"}

    @ts.checkpoint.state_migration(1, 2)
    def add_version(tree):
        tree = dict(tree)
        tree["metrics"] = dict(tree["metrics"])
        tree["metrics"]["version"] = np.array(2, dtype=np.int64)
        return tree

    migrated = ts.checkpoint.load_state(path, target_version=2, trust_treedef=True)
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
        ts.checkpoint.load_state(path, trust_treedef=True)


def test_save_load_sharded_state_round_trip(tmp_path):
    mesh = ts.NamedMesh(("dp",), (2,))
    state = {"params": {"w": np.array([[1.0, 2.0]], dtype=np.float32)}}
    root = ts.checkpoint.save_sharded(state, tmp_path / "sharded", mesh)
    loaded = ts.checkpoint.load_sharded(root, mesh, trust_treedef=True)
    np.testing.assert_array_equal(loaded["params"]["w"], state["params"]["w"])


# --- P2 code review 2026-08-29: rules must differentiate the eager loss -------
#
# Both defects were found while fixing the eager paths in the same batch: a
# loss and its rule can drift apart silently, because nothing forces the rule
# to be the derivative of the function the forward actually computes.

def _central_difference(fn, x, h: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        up, down = x.copy(), x.copy()
        up[i] += h
        down[i] -= h
        grad[i] = (fn(up) - fn(down)) / (2.0 * h)
        it.iternext()
    return grad


def test_log_cosh_jvp_primal_matches_the_eager_loss_in_the_tail():
    """log_cosh is even, so the two errors below must give the same loss. The
    JVP rule recomputed the primal on the raw error, where exp(-2e) overflows
    to inf for e < -354.9 — it returned inf while the eager loss returned
    399.3."""
    from tessera.autodiff.jvp import _JVPS

    for err in (400.0, -400.0):
        pred, target = np.array([0.0]), np.array([err])
        eager = losses_module.log_cosh_loss(pred, target)
        primal, _ = _JVPS["log_cosh_loss"](
            (pred, target), (np.zeros(1), np.zeros(1)))
        assert np.isfinite(primal).all()
        assert float(np.ravel(primal)[0]) == pytest.approx(float(eager), rel=1e-12)


@pytest.mark.parametrize(
    "kwargs,ignore_row,shape",
    [
        ({}, None, (4, 5)),
        ({"label_smoothing": 0.3}, None, (4, 5)),
        ({}, 1, (4, 5)),
        ({"label_smoothing": 0.2}, 1, (4, 5)),
        ({"axis": 0}, None, (5, 4)),
    ],
)
def test_cross_entropy_vjp_differentiates_the_loss_it_claims_to(
        kwargs, ignore_row, shape):
    """The rule swallowed label_smoothing, ignore_index and axis in **_, so a
    smoothed loss returned the unsmoothed gradient — bit-identical to the
    label_smoothing=0.0 answer and 0.15 away from the true derivative."""
    from tessera.autodiff.vjp import _VJPS

    rng = np.random.default_rng(0)
    logits = rng.normal(size=shape)
    n_positions = shape[1] if kwargs.get("axis") == 0 else shape[0]
    targets = rng.integers(0, 5, size=(n_positions,))
    if ignore_row is not None:
        targets = targets.copy()
        targets[ignore_row] = -100

    analytic = _VJPS["cross_entropy_loss"](1.0, logits, targets, **kwargs)[0]
    numeric = _central_difference(
        lambda L: losses_module.cross_entropy_loss(L, targets, **kwargs),
        logits)
    np.testing.assert_allclose(analytic, numeric, atol=1e-7)


def test_cross_entropy_vjp_actually_responds_to_label_smoothing():
    from tessera.autodiff.vjp import _VJPS

    rng = np.random.default_rng(1)
    logits = rng.normal(size=(4, 5))
    targets = rng.integers(0, 5, size=(4,))
    plain = _VJPS["cross_entropy_loss"](1.0, logits, targets)[0]
    smoothed = _VJPS["cross_entropy_loss"](
        1.0, logits, targets, label_smoothing=0.3)[0]
    assert not np.allclose(plain, smoothed)
