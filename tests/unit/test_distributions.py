"""Item 1 — `tessera.distributions.*` (Normal, Beta, kl_divergence).

Forward correctness of sample / log_prob / entropy plus closed-form
and Monte-Carlo KL divergence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import tessera as ts


# ─────────────────────────────────────────────────────────────────────────────
# Normal
# ─────────────────────────────────────────────────────────────────────────────


class TestNormal:
    def test_sample_seeded_is_reproducible(self):
        d = ts.distributions.Normal(loc=0.0, scale=1.0)
        a = d.sample((4, 8), seed=42)
        b = d.sample((4, 8), seed=42)
        np.testing.assert_array_equal(a, b)

    def test_sample_shape_includes_batch(self):
        d = ts.distributions.Normal(
            loc=np.zeros((3,), dtype=np.float32),
            scale=np.ones((3,), dtype=np.float32),
        )
        out = d.sample((4,), seed=0)
        assert out.shape == (4, 3)

    def test_log_prob_matches_analytical(self):
        x = np.linspace(-2, 2, 5).astype(np.float64)
        d = ts.distributions.Normal(0.0, 1.0)
        ref = -0.5 * (x ** 2 + np.log(2 * np.pi))
        np.testing.assert_allclose(d.log_prob(x), ref, rtol=1e-9, atol=1e-9)

    def test_log_prob_matches_at_nonstandard_params(self):
        d = ts.distributions.Normal(loc=2.0, scale=0.5)
        x = np.array([1.0, 2.0, 3.0])
        ref = -0.5 * (((x - 2.0) ** 2) / 0.25 + np.log(2 * np.pi * 0.25))
        np.testing.assert_allclose(d.log_prob(x), ref, rtol=1e-12)

    def test_entropy_matches_closed_form(self):
        d = ts.distributions.Normal(0.0, 2.0)
        ref = 0.5 * math.log(2 * math.pi * math.e * 4.0)
        np.testing.assert_allclose(d.entropy(), ref, rtol=1e-12)

    def test_negative_scale_rejected(self):
        with pytest.raises(ValueError, match="scale"):
            ts.distributions.Normal(0.0, -1.0)

    def test_empirical_mean_close_to_loc(self):
        d = ts.distributions.Normal(loc=3.0, scale=1.0)
        s = d.sample((4096,), seed=7)
        assert abs(float(s.mean()) - 3.0) < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Beta
# ─────────────────────────────────────────────────────────────────────────────


class TestBeta:
    def test_sample_in_open_interval(self):
        d = ts.distributions.Beta(alpha=2.0, beta=5.0)
        s = d.sample((1024,), seed=11)
        assert (s > 0).all() and (s < 1).all()

    def test_empirical_mean_close_to_a_over_a_plus_b(self):
        d = ts.distributions.Beta(alpha=2.0, beta=5.0)
        s = d.sample((4096,), seed=13)
        assert abs(float(s.mean()) - 2.0 / 7.0) < 0.05

    def test_log_prob_matches_analytical(self):
        d = ts.distributions.Beta(alpha=3.0, beta=4.0)
        x = np.linspace(0.1, 0.9, 9)
        # B(3, 4) = Gamma(3) * Gamma(4) / Gamma(7) = 2! * 3! / 6! = 12/720 = 1/60
        log_beta = math.log(1.0 / 60.0)
        ref = (3 - 1) * np.log(x) + (4 - 1) * np.log1p(-x) - log_beta
        np.testing.assert_allclose(d.log_prob(x), ref, rtol=1e-10)

    def test_log_prob_rejects_boundary(self):
        d = ts.distributions.Beta(2.0, 3.0)
        with pytest.raises(ValueError, match="open interval"):
            d.log_prob(np.array([0.0]))
        with pytest.raises(ValueError, match="open interval"):
            d.log_prob(np.array([1.0]))

    def test_negative_concentrations_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            ts.distributions.Beta(0.0, 1.0)
        with pytest.raises(ValueError, match="positive"):
            ts.distributions.Beta(1.0, -1.0)


# ─────────────────────────────────────────────────────────────────────────────
# kl_divergence
# ─────────────────────────────────────────────────────────────────────────────


class TestKLDivergence:
    def test_kl_normal_to_self_is_zero(self):
        d = ts.distributions.Normal(0.0, 1.0)
        assert abs(float(ts.distributions.kl_divergence(d, d))) < 1e-12

    def test_kl_normal_normal_closed_form(self):
        p = ts.distributions.Normal(0.0, 1.0)
        q = ts.distributions.Normal(2.0, 1.5)
        # Reference: log(s1/s0) + (s0^2 + (mu0-mu1)^2)/(2 s1^2) - 0.5
        ref = (
            np.log(1.5 / 1.0) + (1.0 + 4.0) / (2 * 2.25) - 0.5
        )
        np.testing.assert_allclose(
            ts.distributions.kl_divergence(p, q), ref, rtol=1e-12
        )

    def test_kl_beta_to_self_is_zero(self):
        d = ts.distributions.Beta(2.5, 3.5)
        np.testing.assert_allclose(
            ts.distributions.kl_divergence(d, d), 0.0, atol=1e-12
        )

    def test_kl_cross_type_uses_monte_carlo(self):
        """Normal vs Beta has no closed form; the MC fallback should
        return a finite, non-NaN array."""
        p = ts.distributions.Normal(0.5, 0.1)
        q = ts.distributions.Beta(2.0, 2.0)
        # Constrain p to roughly the Beta support so log_prob doesn't
        # explode — KL is approximate either way.
        kl = ts.distributions.kl_divergence(
            p, q, monte_carlo_samples=4096, seed=17,
        )
        assert np.isfinite(kl).all()


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion-LLM-style usage smoke
# ─────────────────────────────────────────────────────────────────────────────


def test_diffusion_llm_style_kl_smoke():
    """The exact pattern at examples/advanced/Diffusion_LLM/
    tessera_diffusion_llm.py:520-525:
        kl = ts.distributions.kl_divergence(
            Normal(pred_noise, ts.exp(0.5 * pred_var)),
            Normal(noise, ts.sqrt(true_var)),
        )
    """
    np.random.seed(0)
    pred_noise = np.random.randn(8).astype(np.float32)
    pred_var = np.random.randn(8).astype(np.float32) * 0.1
    noise = np.random.randn(8).astype(np.float32)
    true_var = np.abs(np.random.randn(8).astype(np.float32)) + 0.1
    p = ts.distributions.Normal(pred_noise, np.exp(0.5 * pred_var))
    q = ts.distributions.Normal(noise, np.sqrt(true_var))
    kl = ts.distributions.kl_divergence(p, q)
    assert kl.shape == (8,)
    assert np.isfinite(kl).all()
