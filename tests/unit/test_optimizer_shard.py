"""
Phase 5 — test_optimizer_shard.py

Tests for ZeROConfig — Python-layer mirror of OptimizerShardPass.
"""
import pytest
from tessera.compiler.solver_config import ZeROConfig


class TestZeROConfigBasic:
    def test_default_stage(self):
        cfg = ZeROConfig()
        assert cfg.stage == 2

    def test_default_dp_axis(self):
        cfg = ZeROConfig()
        assert cfg.dp_axis == "dp"

    def test_stage_1_valid(self):
        cfg = ZeROConfig(stage=1)
        assert cfg.stage == 1

    def test_stage_3_valid(self):
        cfg = ZeROConfig(stage=3, partition_parameters=True)
        assert cfg.stage == 3

    def test_invalid_stage_zero(self):
        with pytest.raises(ValueError):
            ZeROConfig(stage=0)

    def test_invalid_stage_four(self):
        with pytest.raises(ValueError):
            ZeROConfig(stage=4)

    def test_invalid_num_dp_ranks(self):
        with pytest.raises(ValueError):
            ZeROConfig(num_dp_ranks=0)

    def test_stage2_partition_params_raises(self):
        with pytest.raises(ValueError):
            ZeROConfig(stage=2, partition_parameters=True)


class TestZeROConfigPartitioning:
    def test_partitioned_param_count_exact_division(self):
        cfg = ZeROConfig(num_dp_ranks=4)
        assert cfg.partitioned_param_count(400) == 100

    def test_partitioned_param_count_ceiling(self):
        cfg = ZeROConfig(num_dp_ranks=3)
        # ceil(10 / 3) = 4
        assert cfg.partitioned_param_count(10) == 4

    def test_memory_reduction_factor(self):
        cfg = ZeROConfig(num_dp_ranks=4)
        assert abs(cfg.memory_reduction_factor() - 0.25) < 1e-9

    def test_memory_reduction_single_rank(self):
        cfg = ZeROConfig(num_dp_ranks=1)
        assert cfg.memory_reduction_factor() == 1.0

    def test_to_ir_attr_contains_stage(self):
        cfg = ZeROConfig(stage=2)
        attr = cfg.to_ir_attr()
        assert "stage = 2" in attr

    def test_to_ir_attr_contains_dp_axis(self):
        cfg = ZeROConfig(dp_axis="data")
        attr = cfg.to_ir_attr()
        assert '"data"' in attr

    def test_to_ir_attr_contains_num_ranks(self):
        cfg = ZeROConfig(num_dp_ranks=8)
        attr = cfg.to_ir_attr()
        assert "num_ranks = 8" in attr

    def test_repr_contains_stage(self):
        cfg = ZeROConfig(stage=2)
        assert "stage=2" in repr(cfg)

    def test_zero3_partition_both_params_and_optimizer(self):
        cfg = ZeROConfig(stage=3, num_dp_ranks=4,
                         partition_optimizer_states=True,
                         partition_gradients=True,
                         partition_parameters=True)
        assert cfg.partition_parameters is True
        assert cfg.stage == 3
