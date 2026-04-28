"""
Phase 5 — test_rng_legalize.py

Tests for SolverConfig RNG stream assignment and RNGStreamPlan.
"""
import pytest
from tessera.compiler.solver_config import (
    SolverConfig, RNGStreamPlan, RNGBackend,
)


class TestRNGStreamId:
    """Test SolverConfig.rng_stream_id()."""

    def test_rank_zero_seed_zero(self):
        cfg = SolverConfig(global_seed=0, num_ranks=4)
        assert cfg.rng_stream_id(0) == 0

    def test_rank_one_seed_zero(self):
        cfg = SolverConfig(global_seed=0, num_ranks=4)
        assert cfg.rng_stream_id(1) == 1

    def test_rank_with_nonzero_seed(self):
        # stream_id = seed * num_ranks + rank
        cfg = SolverConfig(global_seed=5, num_ranks=4)
        assert cfg.rng_stream_id(0) == 5 * 4 + 0  # 20
        assert cfg.rng_stream_id(3) == 5 * 4 + 3  # 23

    def test_all_ranks_unique(self):
        cfg = SolverConfig(global_seed=42, num_ranks=8)
        ids = [cfg.rng_stream_id(r) for r in range(8)]
        assert len(set(ids)) == 8

    def test_invalid_rank_negative(self):
        cfg = SolverConfig(num_ranks=4)
        with pytest.raises(ValueError):
            cfg.rng_stream_id(-1)

    def test_invalid_rank_too_large(self):
        cfg = SolverConfig(num_ranks=4)
        with pytest.raises(ValueError):
            cfg.rng_stream_id(4)

    def test_single_rank(self):
        cfg = SolverConfig(global_seed=99, num_ranks=1)
        assert cfg.rng_stream_id(0) == 99

    def test_stream_plan_returned(self):
        cfg = SolverConfig(global_seed=7, num_ranks=2)
        plan = cfg.rng_stream_plan()
        assert isinstance(plan, RNGStreamPlan)
        assert plan.global_seed == 7
        assert plan.num_ranks == 2


class TestRNGStreamPlan:
    def test_basic_creation(self):
        plan = RNGStreamPlan(backend=RNGBackend.PHILOX, global_seed=0,
                             num_ranks=4)
        assert plan.num_ranks == 4

    def test_stream_id_formula(self):
        plan = RNGStreamPlan(global_seed=3, num_ranks=4)
        assert plan.stream_id(0) == 12
        assert plan.stream_id(1) == 13

    def test_all_stream_ids_length(self):
        plan = RNGStreamPlan(num_ranks=8)
        assert len(plan.all_stream_ids()) == 8

    def test_streams_are_unique(self):
        plan = RNGStreamPlan(global_seed=0, num_ranks=4)
        assert plan.streams_are_unique()

    def test_streams_unique_nonzero_seed(self):
        plan = RNGStreamPlan(global_seed=100, num_ranks=16)
        assert plan.streams_are_unique()

    def test_invalid_num_ranks_zero(self):
        with pytest.raises(ValueError):
            RNGStreamPlan(num_ranks=0)

    def test_invalid_rank_out_of_range(self):
        plan = RNGStreamPlan(num_ranks=4)
        with pytest.raises(ValueError):
            plan.stream_id(4)

    def test_to_ir_attr_contains_backend(self):
        plan = RNGStreamPlan(backend=RNGBackend.PHILOX, num_ranks=2)
        attr = plan.to_ir_attr()
        assert "philox" in attr

    def test_to_ir_attr_contains_num_ranks(self):
        plan = RNGStreamPlan(num_ranks=8)
        attr = plan.to_ir_attr()
        assert "num_ranks = 8" in attr

    def test_threefry_backend(self):
        plan = RNGStreamPlan(backend=RNGBackend.THREEFRY, num_ranks=4)
        assert plan.backend == RNGBackend.THREEFRY
        assert "threefry" in plan.to_ir_attr()

    def test_repr(self):
        plan = RNGStreamPlan(backend=RNGBackend.PHILOX, global_seed=1,
                             num_ranks=2)
        r = repr(plan)
        assert "philox" in r
        assert "num_ranks=2" in r
