"""Probability distribution utilities (deferred-items plan, Item 1).

Lightweight Python-side surface for the distributions referenced by
``examples/advanced/Diffusion_LLM/`` (the only Tessera example that
imports ``tessera.distributions``). Closed-form ``log_prob`` /
``sample`` / ``entropy`` / ``kl_divergence`` for ``Normal`` and
``Beta``; cross-type KL falls back to a Monte-Carlo estimator.

All sampling routes through ``numpy.random.default_rng(seed)`` so a
seeded call is bit-reproducible.

Design notes:
  * Distributions hold their parameter tensors as plain ``numpy``
    arrays (the diffusion-LLM training loop already operates on
    numpy values; nothing in `Diffusion_LLM` needs autograd through
    the distribution's parameters yet).
  * ``log_prob`` is differentiable through the value `x`; if your
    code wants gradients w.r.t. the distribution parameters, register
    custom VJPs via ``tessera.autodiff.custom_rule`` once that demand
    is concrete. (No example currently needs it — keeping the surface
    minimal.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

ShapeLike = Union[int, Tuple[int, ...], None]


def _as_array(x):
    if hasattr(x, "_data"):
        x = x._data
    return np.asarray(x, dtype=np.float64)


def _as_tuple(shape: ShapeLike) -> Tuple[int, ...]:
    if shape is None:
        return ()
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


# ─────────────────────────────────────────────────────────────────────────────
# Distribution base
# ─────────────────────────────────────────────────────────────────────────────


class Distribution:
    """Base class. Subclasses implement ``sample`` and ``log_prob``."""

    def sample(self, shape: ShapeLike = (), *, seed: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError

    def log_prob(self, x) -> np.ndarray:
        raise NotImplementedError

    def entropy(self) -> np.ndarray:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Normal — diagonal Gaussian
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Normal(Distribution):
    """Diagonal Normal (Gaussian).

    ``loc`` and ``scale`` broadcast against each other; the resulting
    batch shape is the broadcast shape. ``scale`` must be positive.
    """

    loc: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        self.loc = _as_array(self.loc)
        self.scale = _as_array(self.scale)
        if np.any(self.scale <= 0):
            raise ValueError("Normal.scale must be strictly positive")
        # Validate broadcast compatibility eagerly.
        np.broadcast_shapes(self.loc.shape, self.scale.shape)

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return np.broadcast_shapes(self.loc.shape, self.scale.shape)

    def sample(self, shape: ShapeLike = (), *, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        out_shape = _as_tuple(shape) + self.batch_shape
        return (self.loc + self.scale * rng.standard_normal(out_shape)).astype(np.float32)

    def log_prob(self, x) -> np.ndarray:
        x = _as_array(x)
        var = self.scale ** 2
        return -0.5 * (((x - self.loc) ** 2) / var + np.log(2 * np.pi * var))

    def entropy(self) -> np.ndarray:
        # H = 0.5 * log(2 * pi * e * sigma^2) per-dim
        return 0.5 * np.log(2 * np.pi * np.e * self.scale ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Beta — concentration-parameterized
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Beta(Distribution):
    """Beta(alpha, beta) distribution on the open interval (0, 1).

    Both concentrations must be strictly positive.
    """

    alpha: np.ndarray
    beta: np.ndarray

    def __post_init__(self) -> None:
        self.alpha = _as_array(self.alpha)
        self.beta = _as_array(self.beta)
        if np.any(self.alpha <= 0) or np.any(self.beta <= 0):
            raise ValueError("Beta.alpha and Beta.beta must be strictly positive")
        np.broadcast_shapes(self.alpha.shape, self.beta.shape)

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return np.broadcast_shapes(self.alpha.shape, self.beta.shape)

    def sample(self, shape: ShapeLike = (), *, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        out_shape = _as_tuple(shape) + self.batch_shape
        # numpy Beta sampler wants broadcastable arrays of the target shape.
        a = np.broadcast_to(self.alpha, out_shape)
        b = np.broadcast_to(self.beta, out_shape)
        return rng.beta(a, b).astype(np.float32)

    def log_prob(self, x) -> np.ndarray:
        x = _as_array(x)
        if np.any((x <= 0) | (x >= 1)):
            raise ValueError("Beta.log_prob requires x in the open interval (0, 1)")
        # log_prob = (a-1) log x + (b-1) log(1-x) - log B(a, b)
        # log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a+b)
        log_beta = (
            _gammaln(self.alpha)
            + _gammaln(self.beta)
            - _gammaln(self.alpha + self.beta)
        )
        return (
            (self.alpha - 1) * np.log(x)
            + (self.beta - 1) * np.log1p(-x)
            - log_beta
        )

    def entropy(self) -> np.ndarray:
        # H = log B(a, b) - (a-1) digamma(a) - (b-1) digamma(b)
        #     + (a + b - 2) digamma(a + b)
        log_beta = (
            _gammaln(self.alpha)
            + _gammaln(self.beta)
            - _gammaln(self.alpha + self.beta)
        )
        return (
            log_beta
            - (self.alpha - 1) * _digamma(self.alpha)
            - (self.beta - 1) * _digamma(self.beta)
            + (self.alpha + self.beta - 2) * _digamma(self.alpha + self.beta)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Special functions — broadcasting-friendly lgamma / digamma.
#
# Prefer scipy when installed (it's an optional dev dep). Fall back to
# pure-numpy implementations otherwise:
#   * lgamma: vectorized `math.lgamma`.
#   * digamma: recurrence + asymptotic series (Abramowitz & Stegun 6.3.18 +
#     6.3.5). fp64 accurate to ~1e-12 for x > 0.
# ─────────────────────────────────────────────────────────────────────────────


def _gammaln(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import gammaln
        return gammaln(x)
    except ImportError:
        # Vectorize math.lgamma — accurate to fp64 for positive arguments.
        # `frompyfunc` returns object dtype; force fp64 via np.asarray
        # (which handles scalars too — `astype` doesn't on bare floats).
        out = np.frompyfunc(math.lgamma, 1, 1)(np.asarray(x))
        return np.asarray(out, dtype=np.float64)


def _digamma(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import digamma
        return digamma(x)
    except ImportError:
        # Pure-numpy: shift x up using recurrence digamma(x) = digamma(x+1) - 1/x
        # until x > 6, then use the asymptotic series. Accurate to ~1e-12.
        x = np.asarray(x, dtype=np.float64).copy()
        result = np.zeros_like(x)
        # Shift small values up to the asymptotic regime, accumulating the
        # recurrence terms.
        while np.any(x < 6.0):
            mask = x < 6.0
            result = np.where(mask, result - 1.0 / x, result)
            x = np.where(mask, x + 1.0, x)
        # Asymptotic series:
        #   digamma(x) ≈ log(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6)
        inv_x = 1.0 / x
        inv_x2 = inv_x * inv_x
        result = (
            result
            + np.log(x)
            - 0.5 * inv_x
            - inv_x2 * (
                1.0 / 12.0
                - inv_x2 * (
                    1.0 / 120.0
                    - inv_x2 * (1.0 / 252.0)
                )
            )
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# KL divergence — closed-form for matching types; Monte-Carlo otherwise.
# ─────────────────────────────────────────────────────────────────────────────


def kl_divergence(
    p: Distribution,
    q: Distribution,
    *,
    monte_carlo_samples: int = 1024,
    seed: Optional[int] = None,
) -> np.ndarray:
    """KL(p || q).

    Closed-form for matching distribution types (Normal/Normal,
    Beta/Beta). Cross-type KL falls back to a Monte-Carlo estimator
    using ``monte_carlo_samples`` draws from ``p``.
    """
    if isinstance(p, Normal) and isinstance(q, Normal):
        # KL(N(mu0, s0) || N(mu1, s1)) =
        #   log(s1/s0) + (s0^2 + (mu0-mu1)^2) / (2 * s1^2) - 0.5
        return (
            np.log(q.scale / p.scale)
            + (p.scale ** 2 + (p.loc - q.loc) ** 2) / (2 * q.scale ** 2)
            - 0.5
        )
    if isinstance(p, Beta) and isinstance(q, Beta):
        # KL(Beta(a0, b0) || Beta(a1, b1)) =
        #   log B(a1, b1) - log B(a0, b0)
        #   + (a0 - a1)(digamma(a0) - digamma(a0 + b0))
        #   + (b0 - b1)(digamma(b0) - digamma(a0 + b0))
        log_b_p = (
            _gammaln(p.alpha) + _gammaln(p.beta) - _gammaln(p.alpha + p.beta)
        )
        log_b_q = (
            _gammaln(q.alpha) + _gammaln(q.beta) - _gammaln(q.alpha + q.beta)
        )
        dig_ab = _digamma(p.alpha + p.beta)
        return (
            log_b_q
            - log_b_p
            + (p.alpha - q.alpha) * (_digamma(p.alpha) - dig_ab)
            + (p.beta - q.beta) * (_digamma(p.beta) - dig_ab)
        )

    # Monte-Carlo fallback.
    samples = p.sample((monte_carlo_samples,), seed=seed)
    log_p = p.log_prob(samples)
    log_q = q.log_prob(samples)
    return np.mean(log_p - log_q, axis=0)


__all__ = [
    "Distribution",
    "Normal",
    "Beta",
    "kl_divergence",
]
