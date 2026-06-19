from __future__ import annotations

import numpy as np
import pytest

from tessera.models import jepa


def test_jepa_masks_are_deterministic_and_nontrivial():
    mask1 = jepa.mask_blocks_2d((8, 8), block_size=2, mask_ratio=0.5, seed=1)
    mask2 = jepa.mask_blocks_2d((8, 8), block_size=2, mask_ratio=0.5, seed=1)
    tube = jepa.mask_tubes_3d((4, 3, 3), tube_size=2, mask_ratio=0.4, seed=2)

    np.testing.assert_array_equal(mask1, mask2)
    assert mask1.shape == (4, 4)
    assert 0 < mask1.sum() < mask1.size
    assert tube.shape == (2, 3, 3)
    assert 0 < tube.sum() < tube.size


def test_jepa_step_predicts_target_latents_not_logits():
    cfg = jepa.JEPAConfig(input_dim=6, latent_dim=4, predictor_hidden_size=8, mask_ratio=0.5)
    weights = jepa.synthetic_weights(cfg, seed=3)
    tokens = np.arange(12 * cfg.input_dim, dtype=np.float64).reshape(12, cfg.input_dim) / 10.0
    mask = np.zeros(12, dtype=bool)
    mask[[1, 3, 5, 7]] = True

    result = jepa.run_jepa_step(tokens, mask, weights)

    assert result.context_latents.shape == (8, cfg.latent_dim)
    assert result.target_latents.shape == (4, cfg.latent_dim)
    assert result.predictions.shape == result.target_latents.shape
    assert result.loss >= 0.0
    np.testing.assert_allclose(result.target_latents, jepa.encode_target(tokens[mask], weights))


def test_jepa_training_step_is_deterministic_and_loss_matches_reference():
    cfg = jepa.JEPAConfig(input_dim=4, latent_dim=3, predictor_hidden_size=5, mask_ratio=0.5)
    weights = jepa.synthetic_weights(cfg, seed=33)
    tokens = np.linspace(-1.0, 1.0, 16 * cfg.input_dim, dtype=np.float64).reshape(16, cfg.input_dim)
    mask = jepa.mask_blocks_2d((4, 4), block_size=1, mask_ratio=cfg.mask_ratio, seed=123).reshape(-1)

    first = jepa.run_jepa_step(tokens, mask, weights)
    second = jepa.run_jepa_step(tokens, mask, weights)
    expected_loss = jepa.jepa_l2_loss(first.predictions, first.target_latents)
    target_state = {"proj": np.zeros_like(weights.target_proj)}
    online_state = {"proj": weights.context_proj}
    ema = jepa.ema_update(target_state, online_state, decay=0.9)

    np.testing.assert_array_equal(first.mask, second.mask)
    np.testing.assert_allclose(first.predictions, second.predictions)
    assert first.loss == pytest.approx(expected_loss)
    np.testing.assert_allclose(ema["proj"], 0.1 * weights.context_proj)


def test_jepa_rejects_empty_context_or_targets():
    cfg = jepa.JEPAConfig(input_dim=3, latent_dim=2, predictor_hidden_size=4)
    weights = jepa.synthetic_weights(cfg, seed=4)
    tokens = np.ones((4, 3))

    with pytest.raises(jepa.JEPAContractError, match="context and one target"):
        jepa.run_jepa_step(tokens, np.ones(4, dtype=bool), weights)
    with pytest.raises(jepa.JEPAContractError, match="context and one target"):
        jepa.run_jepa_step(tokens, np.zeros(4, dtype=bool), weights)


def test_jepa_ema_update_handles_arrays_and_state_dicts():
    target = {"w": np.array([1.0, 3.0]), "b": np.array([0.0])}
    source = {"w": np.array([3.0, 7.0]), "b": np.array([2.0])}

    updated = jepa.ema_update(target, source, decay=0.75)

    np.testing.assert_allclose(updated["w"], [1.5, 4.0])
    np.testing.assert_allclose(updated["b"], [0.5])
    with pytest.raises(jepa.JEPAContractError, match="keys"):
        jepa.ema_update({"w": np.ones(1)}, {"x": np.ones(1)}, decay=0.5)


def test_jepa_multimodal_latents_share_one_contract():
    cfg = jepa.JEPAConfig(input_dim=5, latent_dim=3, predictor_hidden_size=6, modalities=("image", "video", "text"))
    weights = jepa.synthetic_weights(cfg, seed=5)
    latents = jepa.encode_multimodal_latents(
        {
            "image": np.ones((4, 5)),
            "video": np.ones((6, 5)) * 2,
            "text": np.ones((3, 5)) * 3,
        },
        cfg=cfg,
        weights=weights,
    )

    assert set(latents) == {"image", "video", "text"}
    assert latents["image"].shape == (4, cfg.latent_dim)
    assert latents["video"].shape == (6, cfg.latent_dim)
    assert latents["text"].shape == (3, cfg.latent_dim)
    with pytest.raises(jepa.JEPAContractError, match="unsupported modality"):
        jepa.encode_multimodal_latents({"audio": np.ones((2, 5))}, cfg=cfg, weights=weights)


def test_jepa_selective_decode_is_downstream_optional():
    latents = np.array([[0.1, 0.0], [3.0, 0.0], [0.0, 4.0]], dtype=np.float64)
    decoder_w = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float64)

    result = jepa.selective_decode(latents, decoder_w, threshold=1.0)

    assert result.decode_mask.tolist() == [False, True, True]
    assert result.logits.shape == (2, 3)
    assert result.decoded_token_ids.tolist() == [2, 1]

    skipped = jepa.selective_decode(latents, decoder_w, threshold=10.0)
    assert skipped.decode_mask.tolist() == [False, False, False]
    assert skipped.logits.shape == (0, 3)
    assert skipped.decoded_token_ids.shape == (0,)
