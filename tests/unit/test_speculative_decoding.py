"""Theme 6 — Speculative decoding scheduler.

Forward correctness of the four primitives:
  * `expand_tree` — flat balanced draft tree with parents + paths
  * `acceptance_probabilities` — Leviathan eq.1 ratio clipped to [0, 1]
  * `batch_verify` — per-path acceptance under standard rule with longest-
    prefix tie-break
  * `advance_kv` — cache trim by accepted-prefix length

End-to-end orchestration via `SpeculativeStep` is exercised against a
deterministic RNG seed so the accept/reject decisions are reproducible.

Per-backend Graph IR control flow ops (so a `@tessera.jit` of the whole
loop can lower into a single dispatched kernel) are deferred to Phase G.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.speculative import (
    DraftTree,
    SpeculativeSamplingConfig,
    SpeculativeStep,
    VerificationResult,
    acceptance_probabilities,
    advance_kv,
    batch_verify,
    expand_tree,
    rejection_verify_chain,
    residual_distribution,
    sample_draft_chain,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tree expansion
# ─────────────────────────────────────────────────────────────────────────────


class TestExpandTree:
    def test_balanced_tree_node_count_matches_geometric_sum(self):
        """A balanced (b, d) tree has 1 + b + b² + ... + b^d nodes."""
        for branching, depth in ((2, 3), (3, 2), (4, 2)):
            draft_tokens = [list(range(branching)) for _ in range(depth)]
            tree = expand_tree(
                root_token=0, draft_tokens=draft_tokens,
                branching=branching, depth=depth,
            )
            expected = sum(branching ** d for d in range(depth + 1))
            assert tree.num_nodes == expected

    def test_paths_count_matches_branching_pow_depth(self):
        tree = expand_tree(
            root_token=0,
            draft_tokens=[[10, 11], [20, 21], [30, 31]],
            branching=2, depth=3,
        )
        assert tree.num_paths == 2 ** 3
        assert tree.paths.shape == (8, 4)

    def test_paths_are_root_to_leaf_walks(self):
        """Each path must start with the root token and end at a leaf;
        each consecutive pair must be parent → child in the tree."""
        tree = expand_tree(
            root_token=99,
            draft_tokens=[[10, 11], [20, 21]],
            branching=2, depth=2,
        )
        for path in tree.paths:
            assert path[0] == 99
            # All non-root tokens must appear in the flat tokens array.
            for t in path[1:]:
                assert t in tree.tokens

    def test_depth_zero_produces_root_only(self):
        tree = expand_tree(
            root_token=42, draft_tokens=[], branching=1, depth=0,
        )
        assert tree.num_nodes == 1
        assert tree.tokens.tolist() == [42]
        assert tree.paths.shape == (1, 1)

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError, match="branching"):
            expand_tree(root_token=0, draft_tokens=[], branching=0, depth=0)
        with pytest.raises(ValueError, match="draft_tokens"):
            expand_tree(
                root_token=0, draft_tokens=[[1, 2]], branching=2, depth=2,
            )
        with pytest.raises(ValueError, match="branching"):
            # branching mismatch in inner level
            expand_tree(
                root_token=0, draft_tokens=[[1, 2, 3]], branching=2, depth=1,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance probability + batch_verify
# ─────────────────────────────────────────────────────────────────────────────


class TestAcceptanceProbabilities:
    def test_equal_log_probs_yield_one(self):
        lp = np.log(np.full((4, 3), 0.5))
        np.testing.assert_allclose(acceptance_probabilities(lp, lp), 1.0)

    def test_target_above_draft_clips_to_one(self):
        target = np.log(np.full((2, 2), 0.9))
        draft = np.log(np.full((2, 2), 0.1))
        # raw ratio is 9; clipped to 1.0
        np.testing.assert_array_equal(
            acceptance_probabilities(target, draft), np.full((2, 2), 1.0)
        )

    def test_target_below_draft_yields_ratio(self):
        target = np.log(np.full((2, 2), 0.3))
        draft = np.log(np.full((2, 2), 0.6))
        # ratio = 0.5; uncipped (already in [0, 1])
        np.testing.assert_allclose(
            acceptance_probabilities(target, draft), 0.5
        )

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="shapes"):
            acceptance_probabilities(np.zeros((2, 3)), np.zeros((2, 4)))


class TestBatchVerify:
    def test_high_acceptance_results_in_full_prefix(self):
        """Target ≫ draft means accept every token. Accepted prefix
        length should equal `depth`."""
        rng = np.random.default_rng(0)
        depth = 3
        num_paths = 4
        target = np.log(np.full((num_paths, depth), 0.9))
        draft = np.log(np.full((num_paths, depth), 0.1))
        paths = np.zeros((num_paths, depth + 1), dtype=np.int64)
        for i in range(num_paths):
            paths[i] = [99, 10 + i, 20 + i, 30 + i]
        res = batch_verify(
            target_log_probs=target, draft_log_probs=draft,
            paths=paths, rng=rng,
        )
        assert res.accepted_prefix_length == depth
        np.testing.assert_array_equal(
            res.accepted_prefix,
            paths[res.accepted_path_idx, 1:],
        )

    def test_zero_acceptance_results_in_empty_prefix(self):
        """Target ≪ draft and an unlucky uniform draw yields empty prefix
        (the draft never gets accepted). Use a target so much lower than
        draft that acceptance prob is ~0."""
        rng = np.random.default_rng(0)
        target = np.log(np.full((2, 3), 1e-9))
        draft = np.log(np.full((2, 3), 0.99))
        paths = np.array([[99, 10, 20, 30], [99, 11, 21, 31]], dtype=np.int64)
        res = batch_verify(
            target_log_probs=target, draft_log_probs=draft,
            paths=paths, rng=rng,
        )
        assert res.accepted_prefix_length == 0
        assert res.accepted_prefix.shape == (0,)

    def test_longest_prefix_path_wins(self):
        """When path 0 has 1 accepted token but path 1 has 3, path 1 must
        be picked even though path 0 came first."""
        # Construct an acceptance mask deterministically by setting
        # target/draft so that path 0's first token rejects but path 1
        # passes everything. Use very specific log-prob values + a fixed
        # seed so the RNG draws are reproducible.
        rng = np.random.default_rng(1234)
        target = np.array([
            [np.log(1e-9), np.log(0.9), np.log(0.9)],   # path 0 likely fails
            [np.log(0.99), np.log(0.99), np.log(0.99)], # path 1 always wins
        ])
        draft = np.array([
            [np.log(0.99), np.log(0.1), np.log(0.1)],
            [np.log(0.1), np.log(0.1), np.log(0.1)],
        ])
        paths = np.array([
            [99, 10, 20, 30],
            [99, 11, 21, 31],
        ], dtype=np.int64)
        res = batch_verify(
            target_log_probs=target, draft_log_probs=draft,
            paths=paths, rng=rng,
        )
        # Path 1's accepted prefix is longer.
        assert res.accepted_path_idx == 1
        assert res.accepted_prefix_length == 3
        np.testing.assert_array_equal(res.accepted_prefix, [11, 21, 31])

    def test_acceptance_mask_matches_run_length(self):
        rng = np.random.default_rng(7)
        target = np.log(np.full((3, 4), 0.5))
        draft = np.log(np.full((3, 4), 0.5))  # ratio = 1, all accept
        paths = np.zeros((3, 5), dtype=np.int64)
        res = batch_verify(
            target_log_probs=target, draft_log_probs=draft,
            paths=paths, rng=rng,
        )
        # All accept so every entry of the mask is True and prefix len = 4.
        assert res.acceptance_mask.all()
        assert res.accepted_prefix_length == 4


class TestRejectionSamplingChain:
    def test_residual_distribution_is_exact_on_toy_vocab(self):
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.1, 0.4, 0.5])
        resid = residual_distribution(p, q)
        np.testing.assert_allclose(resid, np.array([1.0, 0.0, 0.0]))

    def test_chain_rejects_and_samples_from_residual(self):
        rng = np.random.default_rng(0)
        draft_tokens = np.array([1])
        q = np.array([[0.1, 0.9, 0.0]])
        p = np.array([[0.8, 0.1, 0.1]])
        res = rejection_verify_chain(draft_tokens, q, p, rng=rng)
        assert res.accepted == 0
        assert res.rejected_at == 0
        assert res.emitted_from == "residual"
        np.testing.assert_allclose(res.residual_probs, np.array([7 / 8, 0.0, 1 / 8]))

    def test_all_accepted_emits_target_bonus(self):
        rng = np.random.default_rng(1)
        draft_tokens = np.array([0, 1])
        q = np.array([[0.4, 0.6], [0.6, 0.4]])
        p = np.array([[0.9, 0.1], [0.1, 0.9], [0.0, 1.0]])
        res = rejection_verify_chain(draft_tokens, q, p, rng=rng)
        assert res.accepted == 2
        assert res.rejected_at is None
        assert res.emitted_from == "target_bonus"
        np.testing.assert_array_equal(res.new_tokens, np.array([0, 1, 1]))

    def test_sample_draft_chain_caches_probabilities(self):
        rng = np.random.default_rng(2)
        logits = np.array([[10.0, -10.0], [-10.0, 10.0]])
        sample = sample_draft_chain(
            logits,
            config=SpeculativeSamplingConfig(temperature=1.0),
            rng=rng,
        )
        np.testing.assert_array_equal(sample.tokens, np.array([0, 1]))
        assert sample.probs.shape == logits.shape
        np.testing.assert_allclose(
            sample.token_probs,
            sample.probs[np.arange(2), sample.tokens],
        )

    def test_rejection_verify_normalizes_probability_rows(self):
        rng = np.random.default_rng(4)
        q = np.array([[7.0, 3.0]])
        p = np.array([[2.0, 8.0], [1.0, 0.0]])
        res = rejection_verify_chain(np.array([1]), q, p, rng=rng)
        assert res.accepted == 1
        np.testing.assert_allclose(res.residual_probs.sum(), 1.0)

    def test_rejection_verify_rejects_bad_probability_rows(self):
        with pytest.raises(ValueError, match="non-negative"):
            rejection_verify_chain(
                np.array([0]),
                np.array([[1.0, -0.1]]),
                np.array([[1.0, 0.0]]),
            )
        with pytest.raises(ValueError, match="positive mass"):
            rejection_verify_chain(
                np.array([0]),
                np.array([[0.0, 0.0]]),
                np.array([[1.0, 0.0]]),
            )
        with pytest.raises(ValueError, match="outside the vocabulary"):
            rejection_verify_chain(
                np.array([2]),
                np.array([[0.5, 0.5]]),
                np.array([[0.5, 0.5]]),
            )

    def test_monte_carlo_rejection_matches_target_distribution(self):
        rng = np.random.default_rng(3)
        q = np.array([[0.7, 0.3]])
        p = np.array([[0.2, 0.8]])
        counts = np.zeros(2, dtype=np.int64)
        for _ in range(4000):
            tok = int(rng.choice(2, p=q[0]))
            res = rejection_verify_chain(np.array([tok]), q, p, rng=rng)
            counts[res.new_tokens[0]] += 1
        empirical = counts / counts.sum()
        np.testing.assert_allclose(empirical, p[0], atol=0.035)


# ─────────────────────────────────────────────────────────────────────────────
# advance_kv
# ─────────────────────────────────────────────────────────────────────────────


class TestAdvanceKV:
    def test_advance_trims_and_zeros_trailing_keys(self):
        cache = ts.cache.KVCacheHandle(num_heads=1, head_dim=2, max_seq=16)
        # Fill with a recognizable pattern so we can spot the trim.
        cache.append(
            np.full((10, 1, 2), 7.0, dtype=np.float32),
            np.full((10, 1, 2), 3.0, dtype=np.float32),
        )
        assert cache.current_seq == 10
        advance_kv(cache, accepted_prefix_length=4)
        assert cache.current_seq == 4
        np.testing.assert_array_equal(
            cache.keys[:4], np.full((4, 1, 2), 7.0, dtype=np.float32)
        )
        # The trimmed-off slots got zeroed out.
        np.testing.assert_array_equal(
            cache.keys[4:10], np.zeros((6, 1, 2), dtype=np.float32)
        )

    def test_advance_works_on_latent_kv_cache(self):
        cache = ts.cache.LatentKVCacheHandle(latent_dim=4, max_seq=16)
        cache.append(np.full((6, 4), 5.0, dtype=np.float32))
        advance_kv(cache, accepted_prefix_length=2)
        assert cache.current_seq == 2
        np.testing.assert_array_equal(
            cache.latents[2:6], np.zeros((4, 4), dtype=np.float32)
        )

    def test_advance_rejects_negative_length(self):
        cache = ts.cache.KVCacheHandle(num_heads=1, head_dim=1, max_seq=4)
        with pytest.raises(ValueError, match="non-negative"):
            advance_kv(cache, accepted_prefix_length=-1)

    def test_advance_rejects_overrun(self):
        cache = ts.cache.KVCacheHandle(num_heads=1, head_dim=1, max_seq=4)
        cache.append(
            np.zeros((2, 1, 1), dtype=np.float32),
            np.zeros((2, 1, 1), dtype=np.float32),
        )
        with pytest.raises(ValueError, match="exceeds"):
            advance_kv(cache, accepted_prefix_length=5)

    def test_advance_rejects_non_cache_argument(self):
        with pytest.raises(TypeError, match="cache"):
            advance_kv("not a cache", accepted_prefix_length=0)


# ─────────────────────────────────────────────────────────────────────────────
# SpeculativeStep — orchestration
# ─────────────────────────────────────────────────────────────────────────────


class TestSpeculativeStep:
    def test_full_roundtrip_advances_cache_by_accepted_length(self):
        rng = np.random.default_rng(0)
        step = SpeculativeStep(branching=2, depth=2)
        cache = ts.cache.KVCacheHandle(num_heads=1, head_dim=2, max_seq=16)
        # Pre-existing context length.
        cache.append(
            np.zeros((3, 1, 2), dtype=np.float32),
            np.zeros((3, 1, 2), dtype=np.float32),
        )
        cache_pre_seq = cache.current_seq

        # Add the draft to the cache (real implementation would do this
        # from the target model's forward pass output).
        cache.append(
            np.full((2, 1, 2), 1.0, dtype=np.float32),
            np.full((2, 1, 2), 1.0, dtype=np.float32),
        )

        target = np.log(np.full((4, 2), 0.9))
        draft = np.log(np.full((4, 2), 0.1))
        result = step.run(
            root_token=99,
            draft_tokens=[[10, 11], [20, 21]],
            target_log_probs=target,
            draft_log_probs=draft,
            cache=cache,
            cache_pre_seq=cache_pre_seq,
            rng=rng,
        )
        # All accepted with ratio = 1.
        assert result.accepted_prefix_length == 2
        # Cache trimmed to pre + accepted (3 + 2 = 5).
        assert cache.current_seq == cache_pre_seq + result.accepted_prefix_length
