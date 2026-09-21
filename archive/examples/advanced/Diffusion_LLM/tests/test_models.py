"""
tests/test_models.py

Integration tests for all three model variants (forward pass, loss, generate).
All tests use debug_tiny configs and run on CPU.
"""

import pytest
import torch

from tessera_diffusion_llm.configs import (
    MDLMConfig,
    ContinuousDiffusionConfig,
    FlowMatchingConfig,
)
from tessera_diffusion_llm.models import (
    MDLM,
    ContinuousDiffusionLLM,
    FlowMatchingLLM,
    DiffusionTransformer,
)


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


def _make_ids(B=2, T=16, vocab=32000, mask_id=31999):
    """Random token ids (no masking) for testing."""
    return torch.randint(0, vocab - 1, (B, T))


# ---------------------------------------------------------------------------
# DiffusionTransformer backbone
# ---------------------------------------------------------------------------

class TestDiffusionTransformer:

    def test_forward_shape(self):
        from tessera_diffusion_llm.configs import TransformerConfig
        cfg = TransformerConfig.debug_tiny()
        backbone = DiffusionTransformer(cfg)
        B, T = 2, 16
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        t   = torch.randint(1, 100, (B,))
        out = backbone(token_ids=ids, t=t)
        assert out.shape == (B, T, cfg.hidden_size)

    def test_x_emb_input(self):
        from tessera_diffusion_llm.configs import TransformerConfig
        cfg = TransformerConfig.debug_tiny()
        backbone = DiffusionTransformer(cfg)
        B, T, H = 2, 16, cfg.hidden_size
        x_emb   = torch.randn(B, T, H)
        t       = torch.randint(1, 100, (B,))
        out     = backbone(token_ids=None, t=t, x_emb=x_emb)
        assert out.shape == (B, T, H)

    def test_num_parameters(self):
        from tessera_diffusion_llm.configs import TransformerConfig
        cfg = TransformerConfig.debug_tiny()
        backbone = DiffusionTransformer(cfg)
        n = backbone.num_parameters()
        assert n > 0


# ---------------------------------------------------------------------------
# MDLM
# ---------------------------------------------------------------------------

class TestMDLM:

    def test_forward_shape(self, mdlm):
        B, T = 2, 16
        cfg  = mdlm.cfg.transformer
        x_t  = torch.randint(0, cfg.vocab_size, (B, T))
        t    = torch.randint(1, 100, (B,))
        logits = mdlm.forward(x_t, t)
        assert logits.shape == (B, T, cfg.vocab_size)

    def test_q_sample_shape(self, mdlm):
        B, T = 2, 16
        x_0  = _make_ids(B, T)
        t    = torch.tensor([50, 80])
        x_t  = mdlm.q_sample(x_0, t)
        assert x_t.shape == (B, T)
        assert x_t.dtype == torch.long

    def test_q_sample_high_t_mostly_masked(self, mdlm):
        """At t=T, nearly all tokens should be masked."""
        B, T = 1, 512
        x_0  = _make_ids(B, T)
        t    = torch.tensor([100])   # max t for debug_tiny (100 steps)
        x_t  = mdlm.q_sample(x_0, t)
        frac_masked = (x_t == mdlm.mask_token_id).float().mean().item()
        assert frac_masked > 0.95

    def test_q_sample_t0_no_masking(self, mdlm):
        """At t=0, mask_prob=0 → no tokens should be masked."""
        B, T = 1, 64
        x_0  = _make_ids(B, T)
        t    = torch.zeros(B, dtype=torch.long)
        x_t  = mdlm.q_sample(x_0, t)
        assert torch.equal(x_t, x_0)

    def test_compute_loss_scalar(self, mdlm):
        x_0  = _make_ids()
        loss = mdlm.compute_loss(x_0)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_compute_loss_backward(self, mdlm):
        mdlm.train()
        x_0  = _make_ids()
        loss = mdlm.compute_loss(x_0)
        loss.backward()   # must not raise

    def test_compute_loss_with_attn_mask(self, mdlm):
        B, T = 2, 16
        x_0  = _make_ids(B, T)
        mask = torch.ones(B, T)
        mask[0, 10:] = 0   # pad last 6 positions of first sequence
        loss = mdlm.compute_loss(x_0, attn_mask=mask)
        assert loss.item() >= 0

    def test_generate_shape(self, mdlm):
        tokens = mdlm.generate(batch_size=2, seq_len=16, num_steps=5)
        assert tokens.shape == (2, 16)
        assert tokens.dtype == torch.long

    def test_generate_no_mask_tokens(self, mdlm):
        tokens = mdlm.generate(batch_size=2, seq_len=16, num_steps=10)
        assert (tokens != mdlm.mask_token_id).all()

    def test_generate_with_prompt(self, mdlm):
        prompt = torch.randint(0, 100, (1, 4))
        tokens = mdlm.generate(batch_size=2, seq_len=16, num_steps=5,
                                prompt_ids=prompt)
        assert tokens.shape == (2, 16)
        # Prompt prefix should be preserved
        assert torch.equal(tokens[:, :4], prompt.expand(2, -1))

    def test_self_conditioning_forward(self, mdlm):
        """Self-conditioning path should run without errors."""
        B, T = 2, 16
        cfg  = mdlm.cfg.transformer
        x_t  = torch.randint(0, cfg.vocab_size, (B, T))
        t    = torch.randint(1, 100, (B,))
        x_sc = torch.softmax(torch.randn(B, T, cfg.vocab_size), dim=-1)
        logits = mdlm.forward(x_t, t, x_self_cond=x_sc)
        assert logits.shape == (B, T, cfg.vocab_size)

    def test_lm_head_weight_tied(self, mdlm):
        assert mdlm.lm_head.weight.data_ptr() == \
               mdlm.backbone.embed_tokens.weight.data_ptr()


# ---------------------------------------------------------------------------
# ContinuousDiffusionLLM
# ---------------------------------------------------------------------------

class TestContinuousDiffusionLLM:

    def test_forward_shape(self, contdiff):
        cfg = contdiff.cfg.transformer
        B, T, H = 2, 16, cfg.hidden_size
        x_t   = torch.randn(B, T, H)
        t     = torch.randint(1, 200, (B,))
        pred, log_var = contdiff.forward(x_t, t)
        assert pred.shape == (B, T, H)
        # learned_variance=False in debug_tiny → log_var is None
        assert log_var is None

    def test_forward_learned_var(self):
        from tessera_diffusion_llm.configs import ContinuousDiffusionConfig, TransformerConfig
        cfg = ContinuousDiffusionConfig(
            transformer=TransformerConfig.debug_tiny(),
            num_timesteps=50,
            learned_variance=True,
        )
        model = ContinuousDiffusionLLM(cfg).eval()
        B, T, H = 2, 8, cfg.transformer.hidden_size
        x_t = torch.randn(B, T, H)
        t   = torch.randint(1, 50, (B,))
        pred, log_var = model.forward(x_t, t)
        assert pred.shape == (B, T, H)
        assert log_var is not None
        assert log_var.shape == (B, T, H)

    def test_q_sample_shape(self, contdiff):
        cfg = contdiff.cfg.transformer
        B, T, H = 2, 16, cfg.hidden_size
        x_0 = torch.randn(B, T, H)
        t   = torch.randint(1, 200, (B,))
        x_t, noise = contdiff.q_sample(x_0, t)
        assert x_t.shape == (B, T, H)
        assert noise.shape == (B, T, H)

    def test_compute_loss_scalar(self, contdiff):
        contdiff.train()
        x_0 = _make_ids()
        loss = contdiff.compute_loss(x_0)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_compute_loss_backward(self, contdiff):
        contdiff.train()
        x_0 = _make_ids()
        loss = contdiff.compute_loss(x_0)
        loss.backward()

    def test_generate_shape(self, contdiff):
        tokens = contdiff.generate(batch_size=2, seq_len=8, num_steps=5)
        assert tokens.shape == (2, 8)
        assert tokens.dtype == torch.long

    def test_generate_ddim(self, contdiff):
        tokens = contdiff.generate(batch_size=1, seq_len=8, num_steps=5,
                                    sampler="ddim", eta=0.0)
        assert tokens.shape == (1, 8)

    def test_generate_top_k(self, contdiff):
        tokens = contdiff.generate(batch_size=1, seq_len=8, num_steps=3,
                                    top_k=10)
        assert tokens.shape == (1, 8)

    def test_generate_with_prompt(self, contdiff):
        prompt = torch.randint(0, 100, (1, 3))
        tokens = contdiff.generate(batch_size=2, seq_len=12, num_steps=3,
                                    prompt_ids=prompt)
        assert tokens.shape == (2, 12)

    def test_predict_logits_shape(self, contdiff):
        cfg = contdiff.cfg.transformer
        B, T, H = 2, 8, cfg.hidden_size
        x_t   = torch.randn(B, T, H)
        t     = torch.randint(1, 50, (B,))
        logits = contdiff.predict_logits(x_t, t)
        assert logits.shape == (B, T, cfg.vocab_size)


# ---------------------------------------------------------------------------
# FlowMatchingLLM
# ---------------------------------------------------------------------------

class TestFlowMatchingLLM:

    def test_forward_shape(self, flowmatch):
        cfg = flowmatch.cfg.transformer
        B, T, H = 2, 16, cfg.hidden_size
        x_t = torch.randn(B, T, H)
        t   = torch.rand(B)           # float in [0,1]
        vel = flowmatch.forward(x_t, t)
        assert vel.shape == (B, T, H)

    def test_interpolate_endpoints(self, flowmatch):
        B, T, H = 2, 8, 16
        x_0 = torch.zeros(B, T, H)
        x_1 = torch.ones(B, T, H)
        # t=0 → x_0
        t0  = torch.zeros(B)
        assert torch.allclose(flowmatch.interpolate(x_0, x_1, t0), x_0)
        # t=1 → x_1
        t1  = torch.ones(B)
        assert torch.allclose(flowmatch.interpolate(x_0, x_1, t1), x_1)

    def test_interpolate_midpoint(self, flowmatch):
        B, T, H = 1, 4, 8
        x_0 = torch.zeros(B, T, H)
        x_1 = torch.ones(B, T, H) * 2.0
        t_half = torch.full((B,), 0.5)
        mid = flowmatch.interpolate(x_0, x_1, t_half)
        assert torch.allclose(mid, torch.ones(B, T, H))

    def test_compute_loss_scalar(self, flowmatch):
        flowmatch.train()
        x_0 = _make_ids()
        loss = flowmatch.compute_loss(x_0)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_compute_loss_backward(self, flowmatch):
        flowmatch.train()
        x_0 = _make_ids()
        loss = flowmatch.compute_loss(x_0)
        loss.backward()

    def test_generate_shape(self, flowmatch):
        tokens = flowmatch.generate(batch_size=2, seq_len=8, num_steps=3)
        assert tokens.shape == (2, 8)
        assert tokens.dtype == torch.long

    def test_generate_midpoint_solver(self, flowmatch):
        tokens = flowmatch.generate(batch_size=1, seq_len=8, num_steps=3,
                                     solver="midpoint")
        assert tokens.shape == (1, 8)

    def test_generate_with_prompt(self, flowmatch):
        prompt = torch.randint(0, 100, (1, 3))
        tokens = flowmatch.generate(batch_size=2, seq_len=12, num_steps=3,
                                     prompt_ids=prompt)
        assert tokens.shape == (2, 12)

    def test_lm_head_weight_tied(self, flowmatch):
        assert flowmatch.lm_head.weight.data_ptr() == \
               flowmatch.backbone.embed_tokens.weight.data_ptr()
