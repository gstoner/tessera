"""
Phase 5 — test_resilience_restart.py

Tests for ResilienceConfig — Python-layer mirror of ResilienceRestartPass.
"""
import pytest
from tessera.compiler.solver_config import ResilienceConfig


class TestResilienceConfigBasic:
    def test_default_interval(self):
        cfg = ResilienceConfig()
        assert cfg.checkpoint_interval == 100

    def test_default_policy(self):
        cfg = ResilienceConfig()
        assert cfg.restart_policy == "last"

    def test_default_max_restarts(self):
        cfg = ResilienceConfig()
        assert cfg.max_restarts == 3

    def test_policy_best(self):
        cfg = ResilienceConfig(restart_policy="best")
        assert cfg.restart_policy == "best"

    def test_policy_epoch(self):
        cfg = ResilienceConfig(restart_policy="epoch")
        assert cfg.restart_policy == "epoch"

    def test_invalid_policy(self):
        with pytest.raises(ValueError, match="restart_policy"):
            ResilienceConfig(restart_policy="random")

    def test_invalid_interval_zero(self):
        with pytest.raises(ValueError):
            ResilienceConfig(checkpoint_interval=0)

    def test_invalid_max_restarts_negative(self):
        with pytest.raises(ValueError):
            ResilienceConfig(max_restarts=-1)

    def test_max_restarts_zero_valid(self):
        cfg = ResilienceConfig(max_restarts=0)
        assert cfg.max_restarts == 0


class TestResilienceConfigSerialization:
    def test_to_ir_attr_is_string(self):
        cfg = ResilienceConfig()
        attr = cfg.to_ir_attr()
        assert isinstance(attr, str)

    def test_to_ir_attr_contains_interval(self):
        cfg = ResilienceConfig(checkpoint_interval=50)
        assert "interval = 50" in cfg.to_ir_attr()

    def test_to_ir_attr_contains_policy(self):
        cfg = ResilienceConfig(restart_policy="best")
        assert '"best"' in cfg.to_ir_attr()

    def test_to_ir_attr_contains_max_restarts(self):
        cfg = ResilienceConfig(max_restarts=5)
        assert "max_restarts = 5" in cfg.to_ir_attr()

    def test_to_ir_attr_contains_resilience_prefix(self):
        cfg = ResilienceConfig()
        assert "tessera_sr.resilience" in cfg.to_ir_attr()

    def test_repr_contains_interval(self):
        cfg = ResilienceConfig(checkpoint_interval=200)
        assert "200" in repr(cfg)

    def test_repr_contains_policy(self):
        cfg = ResilienceConfig(restart_policy="epoch")
        assert "epoch" in repr(cfg)

    def test_fault_barrier_enabled_default(self):
        cfg = ResilienceConfig()
        assert cfg.fault_barrier_enabled is True

    def test_save_dir_default(self):
        cfg = ResilienceConfig()
        assert "/tmp" in cfg.save_dir or cfg.save_dir != ""

    def test_large_interval(self):
        cfg = ResilienceConfig(checkpoint_interval=10_000)
        assert cfg.checkpoint_interval == 10_000
        attr = cfg.to_ir_attr()
        assert "10000" in attr
