"""
tessera.compiler.diffusion_schedule — principled noise-schedule band
partitioning + EDM preconditioning for block-diffusion / denoising models.

This module lifts two self-contained, backend-portable primitives out of the
DiffusionBlocks recipe (Shing/Koyama/Akiba, *DiffusionBlocks: Blockwise
Training for Generative Models*, arXiv:2506.14202, ICLR 2026) into a reusable
Tessera scheduling utility:

1. **Equi-probability (CDF) band partitioning.**  Given a continuous noise
   schedule ``log σ ~ N(P_mean, P_std²)`` and a block count ``B``, partition the
   noise axis ``[σ_min, σ_max]`` into ``B`` contiguous bands that each carry the
   **same cumulative probability mass** (``1/B``) under the training noise
   distribution — rather than splitting σ uniformly.  Equal-mass bands are
   narrow where probability concentrates (intermediate noise, the hardest place
   to denoise) and wide at the extremes, balancing per-block difficulty.  The
   paper's ablation shows this is load-bearing: equi-probability gives
   FID 45.5 versus uniform partitioning's 68.1 on CIFAR-10.  A small overlap
   coefficient ``γ`` widens adjacent bands so block hand-offs are smooth.

2. **EDM preconditioning + σ-weighted loss** (Karras et al. 2022,
   *Elucidating the Design Space of Diffusion-Based Generative Models*).  The
   ``c_skip/c_out/c_in/c_noise`` scalings and the loss weight
   ``w(σ) = (σ² + σ_data²) / (σ · σ_data)²`` that turn a raw network into a
   numerically well-conditioned denoiser ``D_θ``.

The module is pure ``numpy`` + stdlib (``math``): no SciPy, no Torch, no JAX
(Decision #23 — standalone runtime).  The inverse normal CDF is implemented
directly (Acklam's rational approximation + one Halley refinement via
``math.erfc``) so the equal-mass partition has no external dependency.

Band ↔ noise convention
-----------------------
Bands are returned ordered by **increasing σ**: ``band 0`` owns the lowest noise
interval ``[σ_min, …]`` and ``band B-1`` owns the highest ``[…, σ_max]``.  A
block-diffusion network that maps high noise → low noise across its depth (the
reverse-diffusion direction) consumes the bands in reverse; use
:meth:`BandSchedule.reversed_for_depth` to get the depth-ordered view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

__all__ = [
    "DEFAULT_P_MEAN",
    "DEFAULT_P_STD",
    "DEFAULT_SIGMA_MIN",
    "DEFAULT_SIGMA_MAX",
    "DEFAULT_SIGMA_DATA",
    "SigmaBand",
    "BandSchedule",
    "EDMScalings",
    "norm_cdf",
    "norm_ppf",
    "equiprob_sigma_boundaries",
    "equiprob_band_schedule",
    "karras_sigmas",
    "edm_precondition",
    "edm_loss_weight",
]

# EDM log-normal training-noise defaults (Karras et al. 2022, Table 1).
DEFAULT_P_MEAN: float = -1.2
DEFAULT_P_STD: float = 1.2
DEFAULT_SIGMA_MIN: float = 0.002
DEFAULT_SIGMA_MAX: float = 80.0
DEFAULT_SIGMA_DATA: float = 0.5

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Standard normal CDF / inverse-CDF (standalone — no SciPy)
# ---------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    """Standard normal CDF ``Φ(x)`` via ``math.erfc`` (full fp64 accuracy)."""
    return 0.5 * math.erfc(-x / _SQRT2)


# Acklam's rational approximation coefficients for the inverse normal CDF.
_ACKLAM_A = (
    -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
    1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
    6.680131188771972e01, -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
    -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
    3.754408661907416e00,
)
_ACKLAM_PLOW = 0.02425
_ACKLAM_PHIGH = 1.0 - _ACKLAM_PLOW


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF ``Φ⁻¹(p)`` for ``p ∈ (0, 1)``.

    Acklam's rational approximation (relative error ≲ 1.15e-9) refined by a
    single Halley step using ``math.erfc`` for near-machine accuracy.  Raises
    ``ValueError`` outside the open interval ``(0, 1)`` — the equal-mass
    partition never evaluates the (±∞) endpoints.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"norm_ppf requires p in (0, 1), got {p}")

    a, b, c, d = _ACKLAM_A, _ACKLAM_B, _ACKLAM_C, _ACKLAM_D
    if p < _ACKLAM_PLOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= _ACKLAM_PHIGH:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    # One Halley refinement: e = Φ(x) - p ; correct against the true CDF.
    e = norm_cdf(x) - p
    u = e * _SQRT2PI * math.exp(0.5 * x * x)
    x = x - u / (1.0 + 0.5 * x * u)
    return x


# ---------------------------------------------------------------------------
# Equi-probability band partition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SigmaBand:
    """One contiguous noise band owned by a single block.

    Attributes
    ----------
    index : int
        Band index, ordered by increasing σ (0 = lowest noise).
    lo, hi : float
        Core (non-overlapping) band boundaries, ``lo < hi``.  Dispatch
        (:meth:`BandSchedule.block_for_sigma`) uses the core boundaries so the
        block owning any σ is unambiguous.
    lo_train, hi_train : float
        γ-expanded training boundaries (``lo_train ≤ lo``, ``hi_train ≥ hi``).
        Equal to the core boundaries when ``gamma == 0``.  These are the σ-range
        a block is actually trained over, so adjacent blocks overlap slightly.
    prob_mass : float
        Cumulative probability mass of the *core* band under the (truncated)
        training noise distribution.  Equal across bands by construction.
    """

    index: int
    lo: float
    hi: float
    lo_train: float
    hi_train: float
    prob_mass: float

    @property
    def width(self) -> float:
        """Core band width in σ (``hi - lo``)."""
        return self.hi - self.lo


@dataclass(frozen=True)
class BandSchedule:
    """An equal-probability-mass partition of ``[σ_min, σ_max]`` into bands.

    Construct via :func:`equiprob_band_schedule`.  Carries the distribution
    parameters so it is fully self-describing and reproducible.
    """

    bands: tuple[SigmaBand, ...]
    p_mean: float
    p_std: float
    sigma_min: float
    sigma_max: float
    gamma: float

    @property
    def num_blocks(self) -> int:
        return len(self.bands)

    def boundaries(self) -> np.ndarray:
        """The ``B + 1`` core boundary σ values, ascending."""
        return np.array(
            [self.bands[0].lo] + [b.hi for b in self.bands], dtype=np.float64
        )

    def block_for_sigma(self, sigma: float | np.ndarray) -> np.ndarray:
        """Map σ value(s) to the index of the (core) band that owns them.

        σ below ``σ_min`` clamps to band 0; σ at/above ``σ_max`` clamps to the
        last band.  Vectorized over array input.
        """
        bnd = self.boundaries()
        s = np.asarray(sigma, dtype=np.float64)
        idx = np.searchsorted(bnd, s, side="right") - 1
        idx = np.clip(idx, 0, self.num_blocks - 1)
        return idx.astype(np.int64)

    def reversed_for_depth(self) -> tuple[SigmaBand, ...]:
        """Bands in reverse-diffusion (network-depth) order: high σ first.

        A block-diffusion network whose early layers handle the noisiest input
        consumes the bands in this order.
        """
        return tuple(reversed(self.bands))

    def __repr__(self) -> str:
        return (
            f"BandSchedule(num_blocks={self.num_blocks}, "
            f"sigma=[{self.sigma_min:g}, {self.sigma_max:g}], "
            f"P_mean={self.p_mean:g}, P_std={self.p_std:g}, gamma={self.gamma:g})"
        )


def equiprob_sigma_boundaries(
    num_blocks: int,
    *,
    p_mean: float = DEFAULT_P_MEAN,
    p_std: float = DEFAULT_P_STD,
    sigma_min: float = DEFAULT_SIGMA_MIN,
    sigma_max: float = DEFAULT_SIGMA_MAX,
) -> np.ndarray:
    """Return the ``B + 1`` ascending σ boundaries of an equal-mass partition.

    Boundaries are placed at evenly spaced cumulative-probability quantiles of
    the log-normal training-noise distribution ``log σ ~ N(P_mean, P_std²)``,
    truncated to ``[σ_min, σ_max]``::

        q_b = Φ(σ_min) + (b / B) · (Φ(σ_max) − Φ(σ_min))
        σ_b = exp(P_mean + P_std · Φ⁻¹(q_b)),   b = 0 … B

    so every band ``[σ_b, σ_{b+1}]`` carries mass ``1/B`` of the truncated
    distribution.  ``σ_0 == σ_min`` and ``σ_B == σ_max`` exactly.

    Raises ``ValueError`` for non-positive ``num_blocks`` or a degenerate σ
    range.
    """
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
    if not (0.0 < sigma_min < sigma_max):
        raise ValueError(
            f"require 0 < sigma_min < sigma_max, got "
            f"sigma_min={sigma_min}, sigma_max={sigma_max}"
        )
    if p_std <= 0.0:
        raise ValueError(f"p_std must be > 0, got {p_std}")

    cdf_min = norm_cdf((math.log(sigma_min) - p_mean) / p_std)
    cdf_max = norm_cdf((math.log(sigma_max) - p_mean) / p_std)

    out = np.empty(num_blocks + 1, dtype=np.float64)
    for b in range(num_blocks + 1):
        q = cdf_min + (cdf_max - cdf_min) * (b / num_blocks)
        # Clamp away from the open-interval endpoints for norm_ppf; the b=0 and
        # b=B boundaries are pinned to the exact σ range below regardless.
        q = min(max(q, 1e-12), 1.0 - 1e-12)
        out[b] = math.exp(p_mean + p_std * norm_ppf(q))
    # Pin the extremes to the requested range (eliminate ppf round-off there).
    out[0] = sigma_min
    out[num_blocks] = sigma_max
    return out


def equiprob_band_schedule(
    num_blocks: int,
    *,
    gamma: float = 0.0,
    p_mean: float = DEFAULT_P_MEAN,
    p_std: float = DEFAULT_P_STD,
    sigma_min: float = DEFAULT_SIGMA_MIN,
    sigma_max: float = DEFAULT_SIGMA_MAX,
) -> BandSchedule:
    """Build an equal-probability-mass :class:`BandSchedule` of ``num_blocks``.

    ``gamma`` (``0 ≤ γ < 0.5``, typically ``0.1``) is the block-overlap
    coefficient.  With ``α = (hi / lo) ** γ`` each band's *training* range is
    widened to ``[lo / α, hi · α]`` (clamped to ``[σ_min, σ_max]``) so adjacent
    blocks see overlapping noise levels and hand off smoothly.  The paper finds
    ``γ ≈ 0.1`` optimal (FID 41.4) versus ``γ = 0`` (45.5) and ``γ = 0.2``
    (56.7 — too much overlap creates conflicting per-block objectives).  Core
    (dispatch) boundaries are unaffected by ``γ``.
    """
    if not (0.0 <= gamma < 0.5):
        raise ValueError(f"gamma must be in [0, 0.5), got {gamma}")

    bnd = equiprob_sigma_boundaries(
        num_blocks,
        p_mean=p_mean,
        p_std=p_std,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    each_mass = 1.0 / num_blocks
    bands: list[SigmaBand] = []
    for i in range(num_blocks):
        lo = float(bnd[i])
        hi = float(bnd[i + 1])
        if gamma > 0.0:
            alpha = (hi / lo) ** gamma
            lo_train = max(lo / alpha, sigma_min)
            hi_train = min(hi * alpha, sigma_max)
        else:
            lo_train = lo
            hi_train = hi
        bands.append(
            SigmaBand(
                index=i,
                lo=lo,
                hi=hi,
                lo_train=lo_train,
                hi_train=hi_train,
                prob_mass=each_mass,
            )
        )
    return BandSchedule(
        bands=tuple(bands),
        p_mean=p_mean,
        p_std=p_std,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        gamma=gamma,
    )


def karras_sigmas(
    num_steps: int,
    *,
    rho: float = 7.0,
    sigma_min: float = DEFAULT_SIGMA_MIN,
    sigma_max: float = DEFAULT_SIGMA_MAX,
) -> np.ndarray:
    """EDM inference-time σ discretization (Karras et al. 2022, eq. 5).

    Returns ``num_steps`` descending σ values from ``σ_max`` down to ``σ_min``::

        σ_i = (σ_max^(1/ρ) + i/(N−1) · (σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ

    ``ρ = 7`` warps the spacing to spend more steps at low noise (where the
    denoiser must be precise).  This is the time-discretization the Euler ODE
    sampler integrates over, dual to :func:`equiprob_band_schedule` (which is
    the *training*-time partition).  ``σ[0] == σ_max`` and ``σ[-1] == σ_min``.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if not (0.0 < sigma_min < sigma_max):
        raise ValueError("require 0 < sigma_min < sigma_max")
    if rho <= 0.0:
        raise ValueError(f"rho must be > 0, got {rho}")
    if num_steps == 1:
        return np.array([sigma_max], dtype=np.float64)
    inv_rho = 1.0 / rho
    ramp = np.linspace(0.0, 1.0, num_steps, dtype=np.float64)
    min_inv = sigma_min ** inv_rho
    max_inv = sigma_max ** inv_rho
    return (max_inv + ramp * (min_inv - max_inv)) ** rho


# ---------------------------------------------------------------------------
# EDM preconditioning + loss weighting
# ---------------------------------------------------------------------------

class EDMScalings(NamedTuple):
    """The four EDM preconditioning scalars for a given σ (Karras 2022).

    The denoiser is parameterized as
    ``D_θ(z, σ) = c_skip·z + c_out·F_θ(c_in·z, c_noise)`` so that the raw
    network ``F_θ`` always sees unit-variance inputs and predicts a
    unit-variance target.  Fields are scalars or numpy arrays matching the
    σ input shape.
    """

    c_skip: np.ndarray
    c_out: np.ndarray
    c_in: np.ndarray
    c_noise: np.ndarray


def edm_precondition(
    sigma: float | np.ndarray,
    *,
    sigma_data: float = DEFAULT_SIGMA_DATA,
) -> EDMScalings:
    """Compute the EDM ``(c_skip, c_out, c_in, c_noise)`` scalings for ``σ``.

    ::

        c_skip  = σ_data² / (σ² + σ_data²)
        c_out   = σ · σ_data / sqrt(σ² + σ_data²)
        c_in    = 1 / sqrt(σ² + σ_data²)
        c_noise = ln(σ) / 4

    Computed in fp64; the caller casts to the storage dtype.  ``σ`` may be a
    scalar or an array (the scalings broadcast elementwise).  Raises
    ``ValueError`` for non-positive ``σ`` (``c_noise`` needs ``ln σ``) or
    ``sigma_data``.
    """
    if sigma_data <= 0.0:
        raise ValueError(f"sigma_data must be > 0, got {sigma_data}")
    s = np.asarray(sigma, dtype=np.float64)
    if np.any(s <= 0.0):
        raise ValueError("edm_precondition requires sigma > 0 (c_noise uses ln σ)")

    sd2 = sigma_data * sigma_data
    denom = s * s + sd2
    root = np.sqrt(denom)
    c_skip = sd2 / denom
    c_out = s * sigma_data / root
    c_in = 1.0 / root
    c_noise = 0.25 * np.log(s)
    return EDMScalings(c_skip=c_skip, c_out=c_out, c_in=c_in, c_noise=c_noise)


def edm_loss_weight(
    sigma: float | np.ndarray,
    *,
    sigma_data: float = DEFAULT_SIGMA_DATA,
) -> np.ndarray:
    """EDM denoising-loss weight ``w(σ) = (σ² + σ_data²) / (σ · σ_data)²``.

    This is the reciprocal of ``c_out²``: weighting the per-σ reconstruction
    loss by ``w(σ)`` makes the effective training target unit-variance at every
    noise level.  Returns fp64; broadcasts over array ``σ``.
    """
    if sigma_data <= 0.0:
        raise ValueError(f"sigma_data must be > 0, got {sigma_data}")
    s = np.asarray(sigma, dtype=np.float64)
    if np.any(s <= 0.0):
        raise ValueError("edm_loss_weight requires sigma > 0")
    sd2 = sigma_data * sigma_data
    return (s * s + sd2) / (s * sigma_data) ** 2
