"""``tessera.hyperbolic`` — hyperbolic geometry primitives (Ch. 6).

Two-line wrappers over :func:`tessera.complex.cross_ratio` give us
the hyperbolic distance on the Poincaré disk and on the upper
half-plane.  Needham develops both models in parallel — they're
isometric via the Cayley transform ``H⁺ → 𝔻: z ↦ (z − i)/(z + i)``.

The hyperbolic structure on each model:

  * **Poincaré disk** ``𝔻 = {z : |z| < 1}`` with metric
    ``ds = 2 |dz| / (1 − |z|²)``.
  * **Upper half-plane** ``H⁺ = {z : Im(z) > 0}`` with metric
    ``ds = |dz| / Im(z)``.

Distance closed forms (used by the functions below):

  * Disk: ``d(z, w) = 2·artanh(|z − w| / |1 − z̄·w|)``.
    Equivalently: ``cosh d = 1 + 2·|z−w|² / ((1−|z|²)(1−|w|²))``.
  * Half-plane: ``d(z, w) = arccosh(1 + |z−w|² / (2·Im(z)·Im(w)))``.

Geodesics:

  * Disk: circular arcs orthogonal to the unit circle (or
    diameters through the origin).
  * Half-plane: vertical lines, or semicircles centered on the
    real axis.

Isometry: every Möbius transformation that maps 𝔻 to itself is a
hyperbolic isometry (and vice versa for H⁺).  The distance is
:func:`tessera.complex.cross_ratio`-derivable — see
:func:`poincare_distance` for the explicit formula.
"""

from __future__ import annotations

import math
from typing import Any

from .complex import _to_complex


# ─────────────────────────────────────────────────────────────────────────────
# Poincaré disk
# ─────────────────────────────────────────────────────────────────────────────

def poincare_distance(z: Any, w: Any) -> float:
    """Hyperbolic distance on the Poincaré disk.

    Both ``z`` and ``w`` must satisfy ``|z| < 1`` strictly.  The
    formula::

        d(z, w) = 2 · artanh(|z − w| / |1 − z̄·w|)

    is the standard closed form and matches what you get by
    integrating the disk metric along the unique geodesic between
    the two points.
    """
    z_c = _to_complex(z)
    w_c = _to_complex(w)
    if abs(z_c) >= 1.0 or abs(w_c) >= 1.0:
        raise ValueError(
            f"poincare_distance requires |z|, |w| < 1; got "
            f"|z|={abs(z_c)}, |w|={abs(w_c)}"
        )
    num = abs(z_c - w_c)
    den = abs(1.0 - z_c.conjugate() * w_c)
    ratio = num / den
    # Numerically clamp to avoid math.atanh(1.0) infinities on edge.
    ratio = min(ratio, 1.0 - 1e-15)
    return 2.0 * math.atanh(ratio)


def poincare_isometry_image(
    coefs: tuple[Any, Any, Any, Any], z: Any,
) -> complex:
    """Apply a Möbius isometry of the disk ``(a, b, c, d)`` to ``z``.

    For ``(a, b, c, d)`` defining a disk automorphism — i.e.,
    ``|a|² − |c|² = 1`` and ``b = c̄``, ``d = ā`` (the Blaschke
    form) — the resulting Möbius map preserves the Poincaré
    distance.  This helper applies the map; the isometry test
    is in :mod:`tests/unit/test_hyperbolic.py`.
    """
    a, b, c, d = (complex(coef) for coef in coefs)
    z_c = _to_complex(z)
    return (a * z_c + b) / (c * z_c + d)


# ─────────────────────────────────────────────────────────────────────────────
# Upper half-plane
# ─────────────────────────────────────────────────────────────────────────────

def upper_half_plane_distance(z: Any, w: Any) -> float:
    """Hyperbolic distance on the upper half-plane.

    Both ``z`` and ``w`` must satisfy ``Im(z) > 0`` strictly.

    Closed form::

        d(z, w) = arccosh(1 + |z − w|² / (2 · Im(z) · Im(w)))
    """
    z_c = _to_complex(z)
    w_c = _to_complex(w)
    if z_c.imag <= 0 or w_c.imag <= 0:
        raise ValueError(
            f"upper_half_plane_distance requires Im(z), Im(w) > 0; "
            f"got Im(z)={z_c.imag}, Im(w)={w_c.imag}"
        )
    diff_sq = abs(z_c - w_c) ** 2
    arg = 1.0 + diff_sq / (2.0 * z_c.imag * w_c.imag)
    # arccosh(x) = log(x + sqrt(x² − 1)) for x ≥ 1.
    if arg < 1.0:
        # Numerical noise can push slightly below 1 when z = w.
        arg = 1.0
    return math.acosh(arg)


# ─────────────────────────────────────────────────────────────────────────────
# Cayley transform — the canonical isometry H⁺ → 𝔻.
# ─────────────────────────────────────────────────────────────────────────────

def cayley_to_disk(z: Any) -> complex:
    """Cayley transform ``H⁺ → 𝔻``: ``c(z) = (z − i)/(z + i)``.

    Carries the upper half-plane to the open unit disk and is
    an isometry between the two hyperbolic models.
    """
    z_c = _to_complex(z)
    return (z_c - 1j) / (z_c + 1j)


def cayley_from_disk(w: Any) -> complex:
    """Inverse Cayley transform ``𝔻 → H⁺``: ``c⁻¹(w) = i(1 + w)/(1 − w)``."""
    w_c = _to_complex(w)
    return 1j * (1.0 + w_c) / (1.0 - w_c)


__all__ = [
    "poincare_distance",
    "poincare_isometry_image",
    "upper_half_plane_distance",
    "cayley_to_disk",
    "cayley_from_disk",
]
