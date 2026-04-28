"""
Phase 4 — test_moe.py

Tests for MoEConfig, route_tokens(), and plan_all_to_all().
"""
import pytest
import numpy as np
from tessera.distributed.moe import (
    MoEConfig, RoutingResult, route_tokens, plan_all_to_all, AllToAllPlan
)


class TestMoEConfig:
    def test_basic_creation(self):
        cfg = MoEConfig(num_experts=8, top_k=2)
        assert cfg.num_experts == 8
        assert cfg.top_k == 2

    def test_invalid_num_experts_raises(self):
        with pytest.raises(ValueError):
            MoEConfig(num_experts=0)

    def test_top_k_exceeds_experts_raises(self):
        with pytest.raises(ValueError):
            MoEConfig(num_experts=4, top_k=5)

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError):
            MoEConfig(num_experts=8, top_k=0)

    def test_negative_capacity_factor_raises(self):
        with pytest.raises(ValueError):
            MoEConfig(num_experts=8, capacity_factor=-1.0)

    def test_compute_capacity_auto(self):
        cfg = MoEConfig(num_experts=8, capacity_factor=1.25)
        # 128 tokens / (8 experts) * 1.25 = 20
        cap = cfg.compute_capacity(num_tokens=128)
        assert cap == int(np.ceil(1.25 * 128 / 8))

    def test_explicit_capacity_override(self):
        cfg = MoEConfig(num_experts=8, expert_capacity=50)
        assert cfg.compute_capacity(num_tokens=128) == 50


class TestRouteTokens:
    def _make_scores(self, num_tokens, num_experts, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((num_tokens, num_experts)).astype(np.float32)

    def test_assignment_shape(self):
        scores = self._make_scores(128, 8)
        result = route_tokens(scores, MoEConfig(num_experts=8, top_k=2))
        assert result.assignment.shape == (128, 2)

    def test_weights_shape(self):
        scores = self._make_scores(128, 8)
        result = route_tokens(scores, MoEConfig(num_experts=8, top_k=2))
        assert result.weights.shape == (128, 2)

    def test_weights_sum_to_one_top2(self):
        scores = self._make_scores(64, 8)
        result = route_tokens(scores, MoEConfig(num_experts=8, top_k=2,
                                                 normalize_weights=True))
        # Each row (except capacity-dropped) should sum to ~1
        row_sums = result.weights.sum(axis=1)
        # Filter out dropped rows (assignment == -1)
        valid = (result.assignment >= 0).all(axis=1)
        assert np.allclose(row_sums[valid], 1.0, atol=1e-5)

    def test_top1_routing_valid_expert_indices(self):
        scores = self._make_scores(32, 8)
        result = route_tokens(scores, MoEConfig(num_experts=8, top_k=1))
        valid_mask = result.assignment[:, 0] >= 0
        assert ((result.assignment[valid_mask, 0] >= 0) &
                (result.assignment[valid_mask, 0] < 8)).all()

    def test_load_sum_equals_routed_tokens(self):
        scores = self._make_scores(64, 4)
        cfg = MoEConfig(num_experts=4, top_k=1)
        result = route_tokens(scores, cfg)
        # load sums to num_tokens * top_k (minus overflow)
        total_routed = (result.assignment >= 0).sum()
        assert result.load.sum() <= 64 * 1  # at most top_k per token

    def test_wrong_expert_dim_raises(self):
        scores = np.zeros((32, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="num_experts"):
            route_tokens(scores, MoEConfig(num_experts=8, top_k=1))

    def test_non_2d_scores_raises(self):
        scores = np.zeros((32,), dtype=np.float32)
        with pytest.raises(ValueError, match="2D"):
            route_tokens(scores, MoEConfig(num_experts=1, top_k=1))

    def test_overflow_counted(self):
        # Tiny capacity → force overflow
        scores = np.ones((16, 4), dtype=np.float32)  # all routes to expert 0/1/2/3
        cfg = MoEConfig(num_experts=4, top_k=1, expert_capacity=1)
        result = route_tokens(scores, cfg, capacity=1)
        # 16 tokens, 4 experts, capacity=1 → most overflow
        assert result.overflow > 0


class TestPlanAllToAll:
    def _simple_result(self, num_tokens=64, num_experts=8, seed=0):
        scores = np.random.default_rng(seed).standard_normal(
            (num_tokens, num_experts)).astype(np.float32)
        return route_tokens(scores, MoEConfig(num_experts=num_experts, top_k=1))

    def test_plan_returns_all_to_all_plan(self):
        result = self._simple_result()
        plan = plan_all_to_all(result, num_experts=8, num_ranks=4)
        assert isinstance(plan, AllToAllPlan)

    def test_send_counts_shape(self):
        result = self._simple_result()
        plan = plan_all_to_all(result, num_experts=8, num_ranks=4)
        assert plan.send_counts.shape == (4, 4)

    def test_recv_counts_is_transpose(self):
        result = self._simple_result()
        plan = plan_all_to_all(result, num_experts=8, num_ranks=4)
        assert np.array_equal(plan.recv_counts, plan.send_counts.T)

    def test_experts_per_rank(self):
        result = self._simple_result(num_experts=8)
        plan = plan_all_to_all(result, num_experts=8, num_ranks=4)
        assert plan.experts_per_rank == 2

    def test_indivisible_raises(self):
        result = self._simple_result(num_experts=7)
        with pytest.raises(ValueError, match="divisible"):
            plan_all_to_all(result, num_experts=7, num_ranks=4)

    def test_to_ir_attr_string(self):
        result = self._simple_result()
        plan = plan_all_to_all(result, num_experts=8, num_ranks=4)
        attr = plan.to_ir_attr()
        assert "tessera.moe_a2a" in attr
        assert "num_ranks = 4" in attr
