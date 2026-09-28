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


class TestZeROConfigReachesOptimizerShardPass:
    """TILE-LATENT-DEFECTS-2026-09-27: producer and consumer share one contract.

    ``to_ir_attr()`` emits ``tessera_sr.zero_config``; ``OptimizerShardPass``
    used to read ``tessera.num_dp_ranks`` / ``tessera.dp_axis`` instead, so the
    configured values never arrived. The lit fixture
    ``tests/tessera-ir/phase5/optimizer_shard_zero_config.mlir`` runs the pass
    over this exact attribute text and checks that stage 2 / axis ``data`` /
    8 ranks come out; this test keeps the fixture's text byte-equal to what
    Python emits, so the two cannot drift apart.
    """

    FIXTURE = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tessera-ir" / "phase5" / "optimizer_shard_zero_config.mlir"
    )

    def test_fixture_carries_the_python_emitted_attribute(self):
        attr = ZeROConfig(stage=2, dp_axis="data", num_dp_ranks=8).to_ir_attr()
        assert attr == (
            '{tessera_sr.zero_config = {stage = 2, dp_axis = "data", '
            'num_ranks = 8}}'
        )
        assert f"module attributes {attr} {{" in self.FIXTURE.read_text()

    def test_pass_reads_the_emitted_key_not_the_stale_spelling(self):
        pass_src = (
            __import__("pathlib").Path(__file__).resolve().parents[2]
            / "src/solvers/scaling_resilience/lib/sr/passes/OptimizerShardPass.cpp"
        ).read_text()
        assert '"tessera_sr.zero_config"' in pass_src
        for key in ('"stage"', '"dp_axis"', '"num_ranks"'):
            assert key in pass_src, key
        assert '"tessera.num_dp_ranks"' not in pass_src
        assert '"tessera.dp_axis"' not in pass_src
