"""
tests/test_configs.py

Tests for TransformerConfig, MDLMConfig, ContinuousDiffusionConfig,
and FlowMatchingConfig.
"""

import pytest
from tessera_diffusion_llm.configs import (
    TransformerConfig,
    MDLMConfig,
    ContinuousDiffusionConfig,
    FlowMatchingConfig,
)


# ---------------------------------------------------------------------------
# TransformerConfig
# ---------------------------------------------------------------------------

class TestTransformerConfig:

    def test_default_head_dim_computed(self):
        cfg = TransformerConfig(hidden_size=768, num_attention_heads=12)
        assert cfg.head_dim == 64

    def test_explicit_head_dim_preserved(self):
        cfg = TransformerConfig(hidden_size=768, num_attention_heads=12, head_dim=128)
        assert cfg.head_dim == 128

    def test_gqa_valid(self):
        cfg = TransformerConfig(
            hidden_size=256, num_attention_heads=8, num_kv_heads=2
        )
        assert cfg.groups == 4

    def test_gqa_invalid_ratio_raises(self):
        with pytest.raises(AssertionError):
            TransformerConfig(
                hidden_size=256, num_attention_heads=8, num_kv_heads=3
            )

    def test_q_dim(self):
        cfg = TransformerConfig(hidden_size=768, num_attention_heads=12)
        assert cfg.q_dim == 768

    def test_kv_dim_gqa(self):
        cfg = TransformerConfig(
            hidden_size=256, num_attention_heads=8, num_kv_heads=2, head_dim=32
        )
        assert cfg.kv_dim == 2 * 32

    def test_small_factory(self):
        cfg = TransformerConfig.small()
        assert cfg.hidden_size == 768
        assert cfg.num_hidden_layers == 12

    def test_medium_factory(self):
        cfg = TransformerConfig.medium()
        assert cfg.hidden_size == 1024
        assert cfg.num_hidden_layers == 24

    def test_large_factory(self):
        cfg = TransformerConfig.large()
        assert cfg.hidden_size == 1280

    def test_debug_tiny_factory(self):
        cfg = TransformerConfig.debug_tiny()
        assert cfg.hidden_size == 256
        assert cfg.num_hidden_layers == 4
        assert cfg.dropout_p == 0.0

    def test_debug_tiny_head_dim_correct(self):
        cfg = TransformerConfig.debug_tiny()
        assert cfg.head_dim == cfg.hidden_size // cfg.num_attention_heads


# ---------------------------------------------------------------------------
# MDLMConfig
# ---------------------------------------------------------------------------

class TestMDLMConfig:

    def test_default_fields(self):
        cfg = MDLMConfig()
        assert cfg.num_timesteps == 1_000
        assert cfg.mask_schedule == "cosine"
        assert cfg.self_condition is True
        assert cfg.reweight_loss is True

    def test_debug_tiny(self):
        cfg = MDLMConfig.debug_tiny()
        assert cfg.num_timesteps == 100
        assert cfg.mask_schedule == "linear"
        assert cfg.transformer.hidden_size == 256

    def test_transformer_field_is_config(self):
        cfg = MDLMConfig.debug_tiny()
        assert isinstance(cfg.transformer, TransformerConfig)

    def test_self_cond_prob_range(self):
        cfg = MDLMConfig()
        assert 0.0 <= cfg.self_cond_prob <= 1.0


# ---------------------------------------------------------------------------
# ContinuousDiffusionConfig
# ---------------------------------------------------------------------------

class TestContinuousDiffusionConfig:

    def test_default_fields(self):
        cfg = ContinuousDiffusionConfig()
        assert cfg.num_timesteps == 2_000
        assert cfg.beta_schedule == "cosine"
        assert cfg.prediction_type == "epsilon"
        assert cfg.learned_variance is True

    def test_debug_tiny(self):
        cfg = ContinuousDiffusionConfig.debug_tiny()
        assert cfg.num_timesteps == 200
        assert cfg.learned_variance is False
        assert cfg.transformer.hidden_size == 256

    def test_valid_prediction_types(self):
        for pt in ("epsilon", "x_start", "v"):
            cfg = ContinuousDiffusionConfig(prediction_type=pt)
            assert cfg.prediction_type == pt

    def test_valid_beta_schedules(self):
        for sched in ("linear", "cosine", "sqrt"):
            cfg = ContinuousDiffusionConfig(beta_schedule=sched)
            assert cfg.beta_schedule == sched


# ---------------------------------------------------------------------------
# FlowMatchingConfig
# ---------------------------------------------------------------------------

class TestFlowMatchingConfig:

    def test_default_fields(self):
        cfg = FlowMatchingConfig()
        assert cfg.interpolation == "linear"
        assert cfg.num_sampling_steps == 50
        assert cfg.solver == "euler"

    def test_debug_tiny(self):
        cfg = FlowMatchingConfig.debug_tiny()
        assert cfg.num_sampling_steps == 10
        assert cfg.transformer.hidden_size == 256

    def test_valid_solvers(self):
        for solver in ("euler", "midpoint", "rk4"):
            cfg = FlowMatchingConfig(solver=solver)
            assert cfg.solver == solver

    def test_t_distribution_options(self):
        for dist in ("uniform", "logit_normal"):
            cfg = FlowMatchingConfig(t_distribution=dist)
            assert cfg.t_distribution == dist
