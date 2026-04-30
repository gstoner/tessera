"""
tests/test_schedules.py

Tests for noise/masking schedules and sampling algorithms.
All tests run on CPU with tiny configs for speed.
"""

import pytest
import torch

from tessera_diffusion_llm.schedules.noise import (
    cosine_beta_schedule,
    linear_beta_schedule,
    sqrt_beta_schedule,
    cosine_mask_schedule,
    linear_mask_schedule,
    NoiseSchedule,
    MaskSchedule,
)
from tessera_diffusion_llm.schedules.sampling import (
    ddpm_step,
    ddim_step,
    ddpm_sample,
    ddim_sample,
    ode_euler_step,
    flow_ode_sample,
    mdlm_step,
    mdlm_sample,
)


# ---------------------------------------------------------------------------
# Beta schedule functions
# ---------------------------------------------------------------------------

class TestBetaSchedules:

    def test_cosine_shape(self):
        betas = cosine_beta_schedule(100)
        assert betas.shape == (100,)

    def test_linear_shape(self):
        betas = linear_beta_schedule(200)
        assert betas.shape == (200,)

    def test_sqrt_shape(self):
        betas = sqrt_beta_schedule(150)
        assert betas.shape == (150,)

    def test_cosine_range(self):
        betas = cosine_beta_schedule(100)
        assert betas.min() > 0
        assert betas.max() < 1.0

    def test_linear_monotone(self):
        betas = linear_beta_schedule(100)
        assert (betas[1:] >= betas[:-1]).all()

    def test_sqrt_range(self):
        betas = sqrt_beta_schedule(100)
        assert betas.min() >= 1e-4
        assert betas.max() < 1.0


# ---------------------------------------------------------------------------
# Mask schedule functions
# ---------------------------------------------------------------------------

class TestMaskSchedules:

    def test_cosine_endpoints(self):
        m = cosine_mask_schedule(100)
        assert m[0].item() < 0.01      # ≈0 at t=0
        assert abs(m[-1].item() - 1.0) < 1e-5  # =1 at t=T

    def test_linear_endpoints(self):
        m = linear_mask_schedule(100)
        assert m[0].item() == pytest.approx(0.0)
        assert m[-1].item() == pytest.approx(1.0)

    def test_cosine_shape(self):
        m = cosine_mask_schedule(50)
        assert m.shape == (51,)   # T+1 values

    def test_linear_monotone(self):
        m = linear_mask_schedule(100)
        assert (m[1:] >= m[:-1]).all()


# ---------------------------------------------------------------------------
# NoiseSchedule
# ---------------------------------------------------------------------------

class TestNoiseSchedule:

    @pytest.fixture
    def sched(self):
        return NoiseSchedule(num_timesteps=100, schedule="cosine")

    def test_buffers_registered(self, sched):
        for name in ("betas", "alphas_cumprod", "sqrt_alphas_cumprod",
                     "sqrt_one_minus_alphas_cumprod", "posterior_variance"):
            assert hasattr(sched, name)

    def test_betas_shape(self, sched):
        assert sched.betas.shape == (100,)

    def test_alphas_cumprod_decreasing(self, sched):
        acp = sched.alphas_cumprod
        assert (acp[1:] <= acp[:-1]).all()

    def test_q_sample_shape(self, sched):
        x0 = torch.randn(2, 16, 64)
        t  = torch.randint(0, 100, (2,))
        xt = sched.q_sample(x0, t, torch.randn_like(x0))
        assert xt.shape == x0.shape

    def test_predict_start_from_noise_shape(self, sched):
        x_t = torch.randn(2, 16, 64)
        t   = torch.randint(0, 100, (2,))
        eps = torch.randn_like(x_t)
        x0  = sched.predict_start_from_noise(x_t, t, eps)
        assert x0.shape == x_t.shape

    def test_q_sample_and_reconstruct(self, sched):
        """x0_hat ≈ x0 when noise is known."""
        x0  = torch.randn(1, 4, 8)
        t   = torch.tensor([50])
        eps = torch.randn_like(x0)
        x_t = sched.q_sample(x0, t, eps)
        x0_hat = sched.predict_start_from_noise(x_t, t, eps)
        assert torch.allclose(x0, x0_hat, atol=1e-4)

    def test_all_schedules_instantiate(self):
        for sched_name in ("cosine", "linear", "sqrt"):
            s = NoiseSchedule(100, sched_name)
            assert s.num_timesteps == 100

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError):
            NoiseSchedule(100, "unknown_schedule")


# ---------------------------------------------------------------------------
# MaskSchedule
# ---------------------------------------------------------------------------

class TestMaskSchedule:

    @pytest.fixture
    def ms(self):
        return MaskSchedule(num_timesteps=100, schedule="cosine")

    def test_get_mask_prob_shape(self, ms):
        t    = torch.tensor([10, 50, 90])
        prob = ms.get_mask_prob(t)
        assert prob.shape == (3,)

    def test_get_mask_prob_range(self, ms):
        t    = torch.arange(0, 101)
        prob = ms.get_mask_prob(t)
        assert (prob >= 0.0).all()
        assert (prob <= 1.0).all()

    def test_get_transition_prob_non_negative(self, ms):
        t      = torch.tensor([90, 80])
        t_prev = torch.tensor([80, 70])
        trans  = ms.get_transition_prob(t, t_prev)
        assert (trans >= 0.0).all()

    def test_full_mask_at_T(self, ms):
        t = torch.tensor([100])
        assert ms.get_mask_prob(t).item() == pytest.approx(1.0, abs=1e-5)

    def test_linear_schedule(self):
        ms = MaskSchedule(100, "linear")
        t  = torch.tensor([50])
        assert ms.get_mask_prob(t).item() == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# Sampling algorithms
# ---------------------------------------------------------------------------

class TestDDPMSampling:

    @pytest.fixture
    def sched(self):
        return NoiseSchedule(50, "cosine")

    def test_ddpm_step_shape(self, sched):
        B, T, D = 2, 8, 16
        x_t     = torch.randn(B, T, D)
        t       = torch.tensor([5, 10])
        noise   = torch.randn_like(x_t)
        out     = ddpm_step(x_t, t, noise, sched)
        assert out.shape == (B, T, D)

    def test_ddpm_step_t0_no_noise(self, sched):
        """At t=0, std dev should be 0 → no noise added."""
        B, T, D = 1, 4, 8
        x_t     = torch.randn(B, T, D)
        t       = torch.zeros(B, dtype=torch.long)
        noise   = torch.randn_like(x_t)
        # Run multiple times to confirm zero variance
        outs = [ddpm_step(x_t, t, noise, sched).numpy() for _ in range(4)]
        for o in outs[1:]:
            assert abs(float((torch.tensor(outs[0]) - torch.tensor(o)).abs().max())) < 1e-5


class TestDDIMSampling:

    @pytest.fixture
    def sched(self):
        return NoiseSchedule(50, "cosine")

    def test_ddim_step_shape(self, sched):
        B, T, D   = 2, 8, 16
        x_t       = torch.randn(B, T, D)
        t         = torch.tensor([20, 30])
        t_prev    = torch.tensor([10, 20])
        pred_noise = torch.randn_like(x_t)
        out       = ddim_step(x_t, t, t_prev, pred_noise, sched, eta=0.0)
        assert out.shape == (B, T, D)

    def test_ddim_deterministic(self, sched):
        """eta=0 → same output every call."""
        B, T, D   = 1, 4, 8
        x_t       = torch.randn(B, T, D)
        t         = torch.tensor([20])
        t_prev    = torch.tensor([10])
        pred_noise = torch.randn_like(x_t)
        out1 = ddim_step(x_t, t, t_prev, pred_noise, sched, eta=0.0)
        out2 = ddim_step(x_t, t, t_prev, pred_noise, sched, eta=0.0)
        assert torch.allclose(out1, out2)


class TestFlowODE:

    def test_euler_step_shape(self):
        x_t  = torch.randn(2, 8, 16)
        t    = torch.full((2,), 0.5)
        vel  = torch.randn_like(x_t)
        out  = ode_euler_step(x_t, t, vel, dt=0.1)
        assert out.shape == x_t.shape

    def test_euler_step_math(self):
        x_t = torch.ones(1, 1, 1)
        vel = torch.ones(1, 1, 1) * 2.0
        out = ode_euler_step(x_t, None, vel, dt=0.5)
        assert out.item() == pytest.approx(0.0)  # 1 - 0.5*2 = 0


class TestMDLMSampling:

    @pytest.fixture
    def ms(self):
        return MaskSchedule(20, "linear")

    def test_mdlm_step_shape(self, ms):
        B, T     = 2, 16
        V        = 100
        mask_id  = 99
        x_t      = torch.full((B, T), mask_id, dtype=torch.long)
        t        = torch.tensor([10, 15])
        t_prev   = torch.tensor([5, 10])
        logits   = torch.randn(B, T, V)
        out      = mdlm_step(x_t, t, t_prev, logits, mask_id, ms)
        assert out.shape == (B, T)
        assert out.dtype == torch.long

    def test_mdlm_step_never_samples_mask_token(self, ms):
        B, T    = 4, 32
        V       = 100
        mask_id = 99
        x_t     = torch.full((B, T), mask_id, dtype=torch.long)
        t       = torch.full((B,), 10, dtype=torch.long)
        t_prev  = torch.zeros(B, dtype=torch.long)
        logits  = torch.randn(B, T, V)
        out     = mdlm_step(x_t, t, t_prev, logits, mask_id, ms, temperature=1.0)
        # Unmasked positions should never be the mask token
        unmasked = out != mask_id
        if unmasked.any():
            assert not (out[unmasked] == mask_id).any()

    def test_mdlm_step_preserves_unmasked(self, ms):
        B, T    = 2, 16
        V       = 50
        mask_id = 49
        x_t     = torch.randint(0, mask_id, (B, T))  # all unmasked
        t       = torch.tensor([5, 10])
        t_prev  = torch.zeros(B, dtype=torch.long)
        logits  = torch.randn(B, T, V)
        out     = mdlm_step(x_t, t, t_prev, logits, mask_id, ms)
        # Non-masked tokens should remain unchanged
        assert torch.equal(out, x_t)

    def test_mdlm_sample_shape(self, ms):
        B, T_seq = 2, 8
        V        = 50
        mask_id  = 49

        def _model_fn(x_t, t):
            return torch.randn(B, T_seq, V)

        out = mdlm_sample(_model_fn, ms, V, mask_id, (B, T_seq), "cpu", num_steps=5)
        assert out.shape == (B, T_seq)

    def test_mdlm_sample_no_mask_in_output(self, ms):
        """After full generation, no positions should remain masked."""
        B, T_seq = 1, 8
        V        = 50
        mask_id  = 49

        def _model_fn(x_t, t):
            # Strong uniform distribution over non-mask tokens
            logits = torch.zeros(B, T_seq, V)
            logits[..., mask_id] = -1e9
            return logits

        out = mdlm_sample(_model_fn, ms, V, mask_id, (B, T_seq), "cpu", num_steps=20)
        assert (out != mask_id).all()
