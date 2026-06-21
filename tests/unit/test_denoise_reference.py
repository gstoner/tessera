"""Conformance fixture: classification-as-denoising reference model.

A small, seed-reproducible numpy oracle exercising band-dispatch + EDM
preconditioning + an Euler ODE sampler end-to-end (DiffusionBlocks recipe,
arXiv:2506.14202).  The Bayes-optimal denoiser over a fixed class codebook is
closed-form, so these are exact/oracle checks — the kind of metamorphic
invariant the Evaluator's metamorphic oracle would grade a compiled denoiser
against.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler import denoise_reference as dr


def _model():
    return dr.make_classifier(num_classes=8, dim=16, seed=20260621, num_blocks=4)


# ── posterior / denoiser sanity ─────────────────────────────────────────────


def test_posterior_is_a_distribution():
    m = _model()
    z = m.embeddings + 0.1
    post = m.posterior(z, sigma=0.3)
    assert post.shape == (8, 8)
    np.testing.assert_allclose(post.sum(axis=-1), 1.0, atol=1e-12)
    assert np.all(post >= 0.0)


def test_classify_recovers_clean_labels():
    """Classification-as-denoising: clean class embeddings classify to their
    own index."""
    m = _model()
    preds = m.classify(m.embeddings, sigma=0.05)
    np.testing.assert_array_equal(preds, np.arange(m.num_classes))


def test_denoise_recovers_clean_embedding_at_low_noise():
    m = _model()
    rng = np.random.default_rng(7)
    y = m.embeddings
    sigma = 0.05
    z = y + sigma * rng.standard_normal(y.shape)
    d = m.denoise(z, sigma)
    # well-separated codebook ⇒ posterior collapses onto the true class
    np.testing.assert_allclose(d, y, atol=1e-2)


def test_denoise_blends_toward_prior_mean_at_high_noise():
    """At very high σ the posterior is near-uniform, so the denoiser returns
    ~the codebook mean (here ≈0 for an orthonormal codebook)."""
    m = _model()
    d = m.denoise(m.embeddings[:1], sigma=1e4)
    np.testing.assert_allclose(d, m.embeddings.mean(axis=0, keepdims=True), atol=1e-3)


# ── Euler ODE sampler + band dispatch ───────────────────────────────────────


def test_sampler_recovers_planted_class():
    """From a noised sample at SNR≈1 (σ_start ≈ σ_data), the probability-flow
    ODE contracts back to the planted class.  (From full σ_max noise the plant
    is information-theoretically swamped — recovery is only defined once the
    signal is resolvable, which is the regime tested here.)"""
    from tessera.compiler.diffusion_schedule import karras_sigmas

    m = _model()
    rng = np.random.default_rng(123)
    targets = np.arange(m.num_classes)
    sigma_start = m.sigma_data
    z0 = m.embeddings + sigma_start * rng.standard_normal(m.embeddings.shape)
    sigmas = karras_sigmas(
        32, sigma_min=m.schedule.sigma_min, sigma_max=sigma_start
    )
    trace = dr.euler_denoise_sample(m, z0, sigmas=sigmas)
    recovered = np.mean(trace.pred_class == targets)
    assert recovered >= 0.75, f"only recovered {recovered:.2f} of planted classes"


def test_band_dispatch_trace_is_valid_and_monotone():
    m = _model()
    rng = np.random.default_rng(1)
    z0 = m.schedule.sigma_max * rng.standard_normal((4, m.dim))
    trace = dr.euler_denoise_sample(m, z0, num_steps=24)
    path = trace.block_path
    # every step routed to a real block
    assert path.min() >= 0 and path.max() <= m.schedule.num_blocks - 1
    # σ descends ⇒ band index (which increases with σ) is non-increasing
    assert np.all(np.diff(path) <= 0)
    # the run starts in the highest-noise band and ends in the lowest
    assert path[0] == m.schedule.num_blocks - 1
    assert path[-1] == 0


def test_sigma_max_routes_to_last_block_sigma_min_to_first():
    m = _model()
    assert int(m.block_for_sigma(m.schedule.sigma_max)) == m.schedule.num_blocks - 1
    assert int(m.block_for_sigma(m.schedule.sigma_min)) == 0


# ── metamorphic self-consistency (oracle invariant) ─────────────────────────


@pytest.mark.parametrize("sigma_a, sigma_b", [(0.05, 0.1), (0.1, 0.05), (0.1, 0.1)])
def test_denoise_renoise_self_consistency(sigma_a, sigma_b):
    """denoise→renoise→denoise recovers the same clean estimate where classes
    are resolvable (σ well below the inter-class separation): the score field
    is self-consistent."""
    m = dr.make_classifier(num_classes=6, dim=16, seed=11, num_blocks=4)
    gap = dr.renoise_consistency_gap(m, m.embeddings, sigma_a, sigma_b, seed=3)
    assert gap < 1e-3, f"renoise consistency gap too large: {gap}"


def test_determinism_same_seed_same_result():
    m1 = dr.make_classifier(seed=42)
    m2 = dr.make_classifier(seed=42)
    np.testing.assert_array_equal(m1.embeddings, m2.embeddings)
    z = np.ones((2, m1.dim))
    np.testing.assert_array_equal(m1.denoise(z, 0.5), m2.denoise(z, 0.5))
