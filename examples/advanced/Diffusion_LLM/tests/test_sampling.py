"""
tests/test_sampling.py

End-to-end generation tests and inference wrapper tests.
Uses debug_tiny configs on CPU.
"""

import pytest
import torch

from tessera_diffusion_llm import (
    MDLM, MDLMConfig,
    ContinuousDiffusionLLM, ContinuousDiffusionConfig,
    FlowMatchingLLM, FlowMatchingConfig,
)
from tessera_diffusion_llm.inference import DiffusionGenerator, GeneratorConfig
from tessera_diffusion_llm.utils import count_parameters, param_summary, tokens_to_human


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mdlm():
    return MDLM(MDLMConfig.debug_tiny()).eval()

@pytest.fixture
def contdiff():
    return ContinuousDiffusionLLM(ContinuousDiffusionConfig.debug_tiny()).eval()

@pytest.fixture
def flowmatch():
    return FlowMatchingLLM(FlowMatchingConfig.debug_tiny()).eval()


# ---------------------------------------------------------------------------
# DiffusionGenerator — MDLM
# ---------------------------------------------------------------------------

class TestGeneratorMDLM:

    def test_generate_returns_correct_shape(self, mdlm):
        gen = DiffusionGenerator(mdlm, GeneratorConfig(num_steps=5))
        tokens = gen.generate(batch_size=3, seq_len=16)
        assert tokens.shape == (3, 16)
        assert tokens.dtype == torch.long

    def test_generate_no_mask_in_output(self, mdlm):
        gen = DiffusionGenerator(mdlm, GeneratorConfig(num_steps=10))
        tokens = gen.generate(batch_size=2, seq_len=16)
        assert (tokens != mdlm.mask_token_id).all()

    def test_generate_large_batch_split(self, mdlm):
        """max_batch splits into multiple sub-calls."""
        gen = DiffusionGenerator(mdlm, GeneratorConfig(num_steps=3, max_batch=2))
        tokens = gen.generate(batch_size=5, seq_len=8)
        assert tokens.shape == (5, 8)

    def test_generate_with_prompt(self, mdlm):
        prompt = torch.randint(0, 100, (1, 4))
        gen    = DiffusionGenerator(mdlm, GeneratorConfig(num_steps=5))
        tokens = gen.generate(batch_size=2, seq_len=12, prompt_ids=prompt)
        # First 4 tokens must match prompt
        assert torch.equal(tokens[:, :4], prompt.expand(2, -1))

    def test_score_returns_scalar_per_sequence(self, mdlm):
        gen   = DiffusionGenerator(mdlm)
        ids   = torch.randint(0, 100, (3, 16))
        score = gen.score(ids)
        assert score.shape == (3,) or score.dim() == 0   # scalar or (B,)


# ---------------------------------------------------------------------------
# DiffusionGenerator — ContinuousDiffusion
# ---------------------------------------------------------------------------

class TestGeneratorContinuous:

    def test_generate_shape(self, contdiff):
        gen = DiffusionGenerator(contdiff, GeneratorConfig(num_steps=3))
        tokens = gen.generate(batch_size=2, seq_len=8)
        assert tokens.shape == (2, 8)

    def test_ddim_sampler(self, contdiff):
        gen = DiffusionGenerator(
            contdiff, GeneratorConfig(num_steps=3, sampler="ddim", eta=0.0)
        )
        tokens = gen.generate(batch_size=1, seq_len=8)
        assert tokens.shape == (1, 8)

    def test_top_k_sampling(self, contdiff):
        gen = DiffusionGenerator(
            contdiff, GeneratorConfig(num_steps=3, top_k=5)
        )
        tokens = gen.generate(batch_size=1, seq_len=8)
        assert tokens.shape == (1, 8)


# ---------------------------------------------------------------------------
# DiffusionGenerator — FlowMatching
# ---------------------------------------------------------------------------

class TestGeneratorFlowMatching:

    def test_generate_shape(self, flowmatch):
        gen = DiffusionGenerator(flowmatch, GeneratorConfig(num_steps=3))
        tokens = gen.generate(batch_size=2, seq_len=8)
        assert tokens.shape == (2, 8)

    def test_midpoint_solver(self, flowmatch):
        gen = DiffusionGenerator(
            flowmatch, GeneratorConfig(num_steps=3, solver="midpoint")
        )
        tokens = gen.generate(batch_size=1, seq_len=8)
        assert tokens.shape == (1, 8)

    def test_temperature_scaling(self, flowmatch):
        gen = DiffusionGenerator(
            flowmatch, GeneratorConfig(num_steps=3, temperature=0.5)
        )
        tokens = gen.generate(batch_size=1, seq_len=8)
        assert tokens.shape == (1, 8)


# ---------------------------------------------------------------------------
# GeneratorConfig defaults
# ---------------------------------------------------------------------------

class TestGeneratorConfig:

    def test_defaults(self):
        cfg = GeneratorConfig()
        assert cfg.temperature == 1.0
        assert cfg.top_k == 0
        assert cfg.sampler == "ddpm"
        assert cfg.solver == "euler"
        assert cfg.max_batch == 64

    def test_custom_values(self):
        cfg = GeneratorConfig(temperature=0.8, top_k=50, num_steps=20)
        assert cfg.temperature == 0.8
        assert cfg.top_k == 50
        assert cfg.num_steps == 20


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

class TestUtils:

    def test_count_parameters_positive(self, mdlm):
        n = count_parameters(mdlm)
        assert n > 0

    def test_count_trainable(self, mdlm):
        n_all = count_parameters(mdlm, trainable_only=False)
        n_tr  = count_parameters(mdlm, trainable_only=True)
        assert n_tr <= n_all

    def test_param_summary_keys(self, mdlm):
        s = param_summary(mdlm)
        assert "total" in s and "trainable" in s and "frozen" in s
        assert s["total"] == s["trainable"] + s["frozen"]

    def test_tokens_to_human_B(self):
        assert "B" in tokens_to_human(1_500_000_000)

    def test_tokens_to_human_M(self):
        assert "M" in tokens_to_human(117_000_000)

    def test_tokens_to_human_K(self):
        assert "K" in tokens_to_human(12_000)

    def test_tokens_to_human_small(self):
        assert tokens_to_human(500) == "500"


# ---------------------------------------------------------------------------
# Package-level imports (smoke test for __init__.py)
# ---------------------------------------------------------------------------

class TestPackageImports:

    def test_model_imports(self):
        from tessera_diffusion_llm import MDLM, ContinuousDiffusionLLM, FlowMatchingLLM
        assert MDLM is not None

    def test_config_imports(self):
        from tessera_diffusion_llm import MDLMConfig, ContinuousDiffusionConfig, FlowMatchingConfig
        assert MDLMConfig is not None

    def test_schedule_imports(self):
        from tessera_diffusion_llm import (
            NoiseSchedule, MaskSchedule,
            ddpm_sample, ddim_sample, mdlm_sample, flow_ode_sample
        )
        assert NoiseSchedule is not None

    def test_training_imports(self):
        from tessera_diffusion_llm import DiffusionTrainer, TrainerConfig
        assert DiffusionTrainer is not None

    def test_inference_imports(self):
        from tessera_diffusion_llm import DiffusionGenerator, GeneratorConfig
        assert DiffusionGenerator is not None

    def test_version(self):
        import tessera_diffusion_llm
        assert hasattr(tessera_diffusion_llm, "__version__")
