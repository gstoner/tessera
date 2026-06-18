"""Bebop-style e2e TV loss tests for MTP training."""

from __future__ import annotations

import numpy as np

from tessera.losses import mtp_e2e_tv_loss


def _logits_from_probs(p):
    return np.log(np.asarray(p, dtype=np.float64))


def test_identical_distributions_have_zero_loss_and_full_acceptance():
    probs = np.array([[[[0.7, 0.2, 0.1], [0.2, 0.6, 0.2]]]])
    logits = _logits_from_probs(probs)
    loss, metrics = mtp_e2e_tv_loss(logits, logits, return_metrics=True)
    assert loss == 0.0
    np.testing.assert_allclose(metrics["per_step_tv"], 0.0)
    np.testing.assert_allclose(metrics["per_step_alpha"], 1.0)
    np.testing.assert_allclose(metrics["expected_accept_len"], 2.0)


def test_e2e_tv_weights_early_mismatch_more_than_late_mismatch():
    target = _logits_from_probs(np.array([[[[0.9, 0.1], [0.9, 0.1]]]]))
    early_bad = _logits_from_probs(np.array([[[[0.1, 0.9], [0.9, 0.1]]]]))
    late_bad = _logits_from_probs(np.array([[[[0.9, 0.1], [0.1, 0.9]]]]))
    assert mtp_e2e_tv_loss(target, early_bad) > mtp_e2e_tv_loss(target, late_bad)


def test_better_overlap_improves_expected_accept_length_on_high_entropy_target():
    target = _logits_from_probs(np.array([[[[0.34, 0.33, 0.33], [0.34, 0.33, 0.33]]]]))
    poor = _logits_from_probs(np.array([[[[0.90, 0.05, 0.05], [0.05, 0.90, 0.05]]]]))
    close = _logits_from_probs(np.array([[[[0.40, 0.30, 0.30], [0.30, 0.40, 0.30]]]]))
    poor_loss, poor_m = mtp_e2e_tv_loss(target, poor, return_metrics=True)
    close_loss, close_m = mtp_e2e_tv_loss(target, close, return_metrics=True)
    assert close_loss < poor_loss
    assert float(close_m["expected_accept_len"].mean()) > float(poor_m["expected_accept_len"].mean())


def test_masked_mean_uses_unmasked_positions_only():
    target = _logits_from_probs(np.array([
        [[[0.8, 0.2]], [[0.8, 0.2]]],
    ]))
    draft = _logits_from_probs(np.array([
        [[[0.8, 0.2]], [[0.2, 0.8]]],
    ]))
    masked = mtp_e2e_tv_loss(target, draft, mask=np.array([[1.0, 0.0]]))
    assert masked == 0.0


def test_mask_shape_is_position_level_not_step_level():
    target = _logits_from_probs(np.array([[[[0.8, 0.2], [0.8, 0.2]]]]))
    draft = _logits_from_probs(np.array([[[[0.8, 0.2], [0.2, 0.8]]]]))
    try:
        mtp_e2e_tv_loss(target, draft, mask=np.ones((1, 1, 2)))
    except ValueError as exc:
        assert "mask must have shape" in str(exc)
    else:
        raise AssertionError("step-level mask should be rejected until semantics are explicit")
