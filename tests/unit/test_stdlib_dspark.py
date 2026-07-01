import numpy as np

from tessera.stdlib import dspark


def _weights(vocab=7, hidden=5, seed=0):
    rng = np.random.default_rng(seed)
    return dspark.DSparkWeights(
        token_embedding=rng.standard_normal((vocab, hidden)).astype(np.float32) * 0.2,
        hidden_proj=rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.15,
        token_proj=rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.15,
        out_proj=rng.standard_normal((hidden, vocab)).astype(np.float32) * 0.2,
        confidence_proj=rng.standard_normal(hidden).astype(np.float32) * 0.1,
        markov=rng.standard_normal((vocab, hidden)).astype(np.float32) * 0.05,
    )


def test_anchor_sampling_and_masks_pin_dspark_contract():
    scores = np.array([0.1, 4.0, 1.0, 3.0, 2.0, 9.0], dtype=np.float32)
    anchors = dspark.sample_anchors(scores, 2, block_size=2, mode="topk")
    # Position 5 is invalid for a two-token block, so top-k over valid starts is 1,3.
    np.testing.assert_array_equal(anchors, np.array([1, 3]))

    cand = dspark.anchor_candidate_mask(anchors, seq_len=6, block_size=2)
    assert cand.shape == (2, 6)
    np.testing.assert_array_equal(cand[0], [False, True, True, False, False, False])
    np.testing.assert_array_equal(cand[1], [False, False, False, True, True, False])

    attn = dspark.anchor_block_attention_mask(anchors, context_len=6, block_size=2)
    assert attn.shape == (2, 2, 8)
    # Anchor 1, step 0 attends context tokens <=1 and no draft tokens.
    np.testing.assert_array_equal(attn[0, 0], [True, True, False, False, False, False, False, False])
    # Anchor 1, step 1 attends context tokens <=2 and the previous draft slot.
    np.testing.assert_array_equal(attn[0, 1], [True, True, True, False, False, False, True, False])


def test_draft_block_forward_shapes_are_deterministic():
    cfg = dspark.DSparkConfig(num_anchors=3, block_size=4, vocab_size=7)
    weights = _weights()
    rng = np.random.default_rng(1)
    hidden = rng.standard_normal((2, 8, 5)).astype(np.float32)
    prev = np.array([2, 4], dtype=np.int64)
    anchors = dspark.sample_anchors(8, 3, block_size=4, mode="uniform")

    out_a = dspark.draft_block_forward(hidden, prev, anchors, weights, cfg)
    out_b = dspark.draft_block_forward(hidden, prev, anchors, weights, cfg)

    assert out_a.logits.shape == (2, 3, 4, 7)
    assert out_a.confidence_logits.shape == (2, 3, 4)
    assert out_a.tokens.shape == (2, 3, 4)
    assert out_a.hidden.shape == (2, 3, 4, 5)
    np.testing.assert_allclose(out_a.logits, out_b.logits)
    np.testing.assert_array_equal(out_a.tokens, np.argmax(out_a.logits, axis=-1))


def test_confident_prefix_and_proposal_select_longest_anchor():
    draft = dspark.DSparkDraftOutput(
        logits=np.zeros((2, 3, 4, 5), dtype=np.float32),
        confidence_logits=np.array([
            [[4.0, 4.0, -4.0, 4.0], [3.0, 3.0, 3.0, -3.0], [2.0, -2.0, 2.0, 2.0]],
            [[-3.0, 3.0, 3.0, 3.0], [2.0, 2.0, -2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
        ], dtype=np.float32),
        tokens=np.arange(2 * 3 * 4, dtype=np.int64).reshape(2, 3, 4),
        hidden=np.zeros((2, 3, 4, 6), dtype=np.float32),
    )
    lengths = dspark.confident_prefix(draft.confidence_logits, 0.5)
    np.testing.assert_array_equal(lengths, [[2, 3, 1], [0, 2, 4]])

    prop = dspark.select_proposal(draft, np.array([0, 4, 8]), threshold=0.5)
    np.testing.assert_array_equal(prop.anchor_index, [1, 2])
    np.testing.assert_array_equal(prop.anchor_position, [4, 8])
    np.testing.assert_array_equal(prop.prefix_length, [3, 4])
    np.testing.assert_array_equal(prop.tokens[0], draft.tokens[0, 1])
    np.testing.assert_array_equal(prop.tokens[1], draft.tokens[1, 2])


def test_dspark_losses_cover_ce_l1_probability_and_confidence():
    logits = np.array([[[[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]]]], dtype=np.float32)
    target_ids = np.array([[[0, 1]]], dtype=np.int64)
    target_logits = logits + np.array([[[[0.1, -0.1, 0.0], [0.2, 0.0, -0.2]]]], dtype=np.float32)
    conf_logits = np.array([[[2.0, -2.0]]], dtype=np.float32)
    conf_targets = np.array([[[1.0, 0.0]]], dtype=np.float32)

    losses = dspark.dspark_losses(
        logits,
        target_ids,
        target_logits=target_logits,
        confidence_logits=conf_logits,
        confidence_targets=conf_targets,
        l1_weight=0.5,
        prob_weight=0.25,
    )

    assert set(losses) == {"ce", "l1", "prob", "confidence_bce", "total"}
    assert losses["ce"] > 0.0
    assert losses["l1"] > 0.0
    assert losses["prob"] > 0.0
    assert losses["confidence_bce"] < 0.2
    expected = losses["ce"] + 0.5 * losses["l1"] + 0.25 * losses["prob"] + losses["confidence_bce"]
    np.testing.assert_allclose(losses["total"], expected)


def test_dspark_reference_proposal_feeds_spec_accept_shape():
    cfg = dspark.DSparkConfig(num_anchors=2, block_size=3, vocab_size=6,
                              confidence_threshold=0.5)
    weights = _weights(vocab=6, hidden=4, seed=3)
    rng = np.random.default_rng(4)
    hidden = rng.standard_normal((1, 5, 4)).astype(np.float32)
    prev = np.array([1], dtype=np.int64)
    anchors = np.array([0, 2], dtype=np.int64)

    draft = dspark.draft_block_forward(hidden, prev, anchors, weights, cfg)
    proposal = dspark.select_proposal(
        draft,
        anchors,
        threshold=cfg.confidence_threshold,
    )

    assert proposal.tokens.shape == (1, cfg.block_size)
    assert proposal.prefix_length.shape == (1,)
    # The selected row is directly consumable as a linear speculative draft path.
    selected_logits = draft.logits[0, proposal.anchor_index[0]]
    assert selected_logits.shape == (cfg.block_size, cfg.vocab_size)
    assert proposal.tokens[0].shape == (cfg.block_size,)
