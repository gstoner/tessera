"""Unit tests for tessera.compiler.diffusion_schedule.

Covers the two DiffusionBlocks-derived primitives:
  - equi-probability (CDF) noise-band partitioning + γ overlap
  - EDM preconditioning (c_skip/c_out/c_in/c_noise) + σ-weighted loss

These are pure-numpy reference checks (no hardware, no SciPy).  The equal-mass
property is the load-bearing claim from the paper's ablation, so it gets the
most attention.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera.compiler import diffusion_schedule as ds


# ── inverse normal CDF ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "p, expected",
    [
        (0.5, 0.0),
        (0.975, 1.959963984540054),       # classic 95% two-sided z
        (0.025, -1.959963984540054),
        (0.9, 1.2815515655446004),
        (0.1, -1.2815515655446004),
        (0.999, 3.090232306167813),
        (0.001, -3.090232306167813),
    ],
)
def test_norm_ppf_matches_known_quantiles(p, expected):
    assert ds.norm_ppf(p) == pytest.approx(expected, abs=1e-10)


def test_norm_cdf_ppf_round_trip():
    for x in np.linspace(-4.0, 4.0, 17):
        assert ds.norm_ppf(ds.norm_cdf(float(x))) == pytest.approx(x, abs=1e-9)


def test_norm_ppf_rejects_out_of_range():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            ds.norm_ppf(bad)


# ── equi-probability boundaries ─────────────────────────────────────────────


def test_boundaries_are_monotonic_and_pin_endpoints():
    bnd = ds.equiprob_sigma_boundaries(6)
    assert bnd.shape == (7,)
    assert np.all(np.diff(bnd) > 0.0)
    assert bnd[0] == pytest.approx(ds.DEFAULT_SIGMA_MIN)
    assert bnd[-1] == pytest.approx(ds.DEFAULT_SIGMA_MAX)


@pytest.mark.parametrize("num_blocks", [1, 2, 3, 4, 6, 8])
def test_each_band_carries_equal_probability_mass(num_blocks):
    """The defining property: every band has mass 1/B of the truncated
    log-normal noise distribution."""
    p_mean, p_std = ds.DEFAULT_P_MEAN, ds.DEFAULT_P_STD
    bnd = ds.equiprob_sigma_boundaries(num_blocks, p_mean=p_mean, p_std=p_std)

    def cdf(sigma: float) -> float:
        return ds.norm_cdf((math.log(sigma) - p_mean) / p_std)

    total = cdf(bnd[-1]) - cdf(bnd[0])
    masses = np.array(
        [(cdf(bnd[i + 1]) - cdf(bnd[i])) / total for i in range(num_blocks)]
    )
    np.testing.assert_allclose(masses, 1.0 / num_blocks, atol=1e-9)


def test_equiprob_concentrates_bands_near_the_median():
    """Equal-mass partitioning makes intermediate-noise bands narrower than a
    uniform-in-σ split would (the whole point versus uniform partitioning)."""
    num_blocks = 8
    bnd = ds.equiprob_sigma_boundaries(num_blocks)
    widths = np.diff(bnd)
    # The median band (around exp(P_mean) ≈ 0.30) must be far narrower than the
    # top band, which absorbs the long right tail out to σ_max.
    assert widths[0] < widths[-1]
    assert widths[num_blocks // 2 - 1] < widths[-1] / 10.0


# ── band schedule + γ overlap ───────────────────────────────────────────────


def test_band_schedule_basic_shape():
    sched = ds.equiprob_band_schedule(4)
    assert sched.num_blocks == 4
    assert len(sched.bands) == 4
    assert [b.index for b in sched.bands] == [0, 1, 2, 3]
    # core bands tile contiguously
    for i in range(3):
        assert sched.bands[i].hi == pytest.approx(sched.bands[i + 1].lo)
    np.testing.assert_allclose([b.prob_mass for b in sched.bands], 0.25)


def test_gamma_zero_means_no_overlap():
    sched = ds.equiprob_band_schedule(4, gamma=0.0)
    for b in sched.bands:
        assert b.lo_train == pytest.approx(b.lo)
        assert b.hi_train == pytest.approx(b.hi)


def test_gamma_widens_training_bands_and_creates_overlap():
    sched = ds.equiprob_band_schedule(4, gamma=0.1)
    for b in sched.bands:
        assert b.lo_train <= b.lo + 1e-12
        assert b.hi_train >= b.hi - 1e-12
    # adjacent training bands now overlap: band i's top reaches past band i+1's
    # core start.
    interior = sched.bands[1]
    assert interior.hi_train > sched.bands[2].lo
    # training ranges stay clamped to the global σ window
    assert sched.bands[0].lo_train >= ds.DEFAULT_SIGMA_MIN - 1e-12
    assert sched.bands[-1].hi_train <= ds.DEFAULT_SIGMA_MAX + 1e-9


def test_gamma_out_of_range_rejected():
    for bad in (-0.1, 0.5, 1.0):
        with pytest.raises(ValueError):
            ds.equiprob_band_schedule(4, gamma=bad)


# ── dispatch ────────────────────────────────────────────────────────────────


def test_block_for_sigma_dispatch_is_consistent_with_boundaries():
    sched = ds.equiprob_band_schedule(5)
    bnd = sched.boundaries()
    # midpoint of each core band dispatches back to that band
    for i in range(sched.num_blocks):
        mid = 0.5 * (bnd[i] + bnd[i + 1])
        assert int(sched.block_for_sigma(mid)) == i


def test_block_for_sigma_clamps_out_of_range():
    sched = ds.equiprob_band_schedule(4)
    assert int(sched.block_for_sigma(1e-9)) == 0
    assert int(sched.block_for_sigma(1e9)) == sched.num_blocks - 1


def test_block_for_sigma_is_vectorized():
    sched = ds.equiprob_band_schedule(4)
    sig = sched.boundaries()[:-1] + 1e-6
    idx = sched.block_for_sigma(sig)
    assert idx.shape == sig.shape
    np.testing.assert_array_equal(idx, np.arange(4))


def test_reversed_for_depth_is_high_noise_first():
    sched = ds.equiprob_band_schedule(4)
    rev = sched.reversed_for_depth()
    assert rev[0].index == 3
    assert rev[0].lo > rev[-1].lo


# ── Karras inference-time σ schedule ────────────────────────────────────────


def test_karras_sigmas_descend_and_pin_endpoints():
    sig = ds.karras_sigmas(20)
    assert sig.shape == (20,)
    assert np.all(np.diff(sig) < 0.0)  # strictly descending
    assert sig[0] == pytest.approx(ds.DEFAULT_SIGMA_MAX)
    assert sig[-1] == pytest.approx(ds.DEFAULT_SIGMA_MIN)


def test_karras_rho7_front_loads_high_noise_steps():
    """ρ=7 spends more steps at low noise: the low-σ end is more finely
    resolved than the high-σ end."""
    sig = ds.karras_sigmas(11)
    gaps = -np.diff(sig)
    assert gaps[0] > gaps[-1]


def test_karras_single_step_is_sigma_max():
    np.testing.assert_allclose(ds.karras_sigmas(1), [ds.DEFAULT_SIGMA_MAX])


# ── EDM preconditioning ─────────────────────────────────────────────────────


@pytest.mark.parametrize("sigma", [0.01, 0.1, 0.5, 1.0, 10.0, 80.0])
def test_edm_precondition_closed_form(sigma):
    sd = 0.5
    sc = ds.edm_precondition(sigma, sigma_data=sd)
    denom = sigma * sigma + sd * sd
    assert float(sc.c_skip) == pytest.approx(sd * sd / denom)
    assert float(sc.c_out) == pytest.approx(sigma * sd / math.sqrt(denom))
    assert float(sc.c_in) == pytest.approx(1.0 / math.sqrt(denom))
    assert float(sc.c_noise) == pytest.approx(0.25 * math.log(sigma))


def test_edm_loss_weight_is_reciprocal_of_c_out_squared():
    sigma = np.array([0.02, 0.1, 0.5, 2.0, 20.0])
    sc = ds.edm_precondition(sigma)
    w = ds.edm_loss_weight(sigma)
    np.testing.assert_allclose(w, 1.0 / (np.asarray(sc.c_out) ** 2), rtol=1e-12)


def test_edm_denoiser_identity_recovers_clean_signal():
    """With the ideal network F* such that the preconditioned denoiser equals
    the clean target, D(y+σε, σ) = y for every σ — the EDM self-consistency
    that makes block hand-offs exact.  Here we verify the algebra: if
    F*(c_in·z, c_noise) = (y - c_skip·z)/c_out, then D == y identically."""
    rng = np.random.default_rng(20260621)
    y = rng.standard_normal((4, 8))
    for sigma in (0.05, 0.7, 5.0):
        eps = rng.standard_normal(y.shape)
        z = y + sigma * eps
        sc = ds.edm_precondition(sigma)
        f_star = (y - np.asarray(sc.c_skip) * z) / np.asarray(sc.c_out)
        d = np.asarray(sc.c_skip) * z + np.asarray(sc.c_out) * f_star
        np.testing.assert_allclose(d, y, atol=1e-10)


def test_edm_precondition_arrays_broadcast():
    sig = np.linspace(0.01, 10.0, 5)
    sc = ds.edm_precondition(sig)
    assert np.asarray(sc.c_skip).shape == sig.shape
    # c_skip → 1 as σ → 0 ; → 0 as σ → ∞
    assert ds.edm_precondition(1e-6).c_skip == pytest.approx(1.0, abs=1e-6)
    assert ds.edm_precondition(1e6).c_skip == pytest.approx(0.0, abs=1e-6)


def test_edm_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        ds.edm_precondition(0.0)
    with pytest.raises(ValueError):
        ds.edm_loss_weight(-1.0)


# ── registry wiring ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "edm_precondition",
        "edm_loss_weight",
        "equiprob_band_partition",
        "karras_sigma_schedule",
    ],
)
def test_primitives_registered_in_coverage(name):
    from tessera.compiler.primitive_coverage import coverage_for

    entry = coverage_for(name)
    assert entry.name == name
    assert entry.status == "partial"
    assert entry.contract_status["tests"] == "complete"
    assert entry.contract_status["math_semantics"] == "complete"


def test_edm_primitives_carry_numeric_policy():
    from tessera.compiler.primitive_coverage import coverage_for

    for name in ("edm_precondition", "edm_loss_weight"):
        entry = coverage_for(name)
        policy = entry.metadata.get("numeric_policy")
        assert policy is not None, f"{name} should pin a numeric_policy"
        assert policy["storage"] == "fp32"
        assert policy["accum"] == "fp32"


def test_band_partition_is_non_differentiable_config():
    from tessera.compiler.primitive_coverage import coverage_for

    entry = coverage_for("equiprob_band_partition")
    assert entry.contract_status["vjp"] == "non_differentiable"
    assert entry.contract_status["jvp"] == "non_differentiable"
