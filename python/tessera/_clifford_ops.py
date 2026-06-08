"""Canonical ``tessera.ops.clifford_*`` flat-coefficient shim over the GA lane.

GA primitives live in the ``tessera.ga.*`` Multivector lane (and already
GPU-dispatch through ``tessera.ga.ops._try_apple_gpu_*`` to the ``cl30`` Metal
kernels). This module exposes them on the *canonical* ``tessera.ops`` surface as
thin wrappers over flat coefficient arrays (Cl(3,0): last-axis length 8), so:

  1. they're reachable from the standard ``tessera.ops`` / ``@jit`` surface
     (unifying the two lanes), and
  2. they flow through the autodiff tape chokepoint, which makes VJP/JVP
     registration meaningful (see ``autodiff/vjp.py`` / ``jvp.py``).

The wrappers inherit the GA lane's GPU dispatch for free — a Cl(3,0) f32 input
runs the fused MSL kernel; everything else falls back to the numpy reference.
Scope: **Cl(3,0)** (the signature with shipped kernels). Inputs are ``(..., 8)``
float arrays (batched leading dims supported).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Lazy GA imports — keep this module import-light (it is pulled by the ops
# namespace builder very early).
_CL = None


def _cl():
    global _CL
    if _CL is None:
        from tessera.ga.signature import Cl
        _CL = Cl(3, 0, 0)
    return _CL


def _mv(coeffs: Any):
    from tessera.ga.multivector import Multivector
    return Multivector(np.asarray(coeffs), _cl())


def _coeffs(mv) -> np.ndarray:
    return np.asarray(mv.coefficients)


# ── bilinear products ─────────────────────────────────────────────────────── #
def clifford_geometric_product(a: Any, b: Any) -> np.ndarray:
    from tessera.ga import ops as G
    return _coeffs(G.geometric_product(_mv(a), _mv(b)))


def clifford_wedge(a: Any, b: Any) -> np.ndarray:
    from tessera.ga import ops as G
    return _coeffs(G.wedge(_mv(a), _mv(b)))


def clifford_left_contraction(a: Any, b: Any) -> np.ndarray:
    from tessera.ga import ops as G
    return _coeffs(G.left_contraction(_mv(a), _mv(b)))


def clifford_inner(a: Any, b: Any) -> np.ndarray:
    from tessera.ga import ops as G
    return np.asarray(G.inner(_mv(a), _mv(b)))


def clifford_rotor_sandwich(rotor: Any, x: Any) -> np.ndarray:
    """Rotor conjugation ``R x R†`` — the GA rotation of ``x`` by rotor ``R``.
    Routes Cl(3,0) f32 to the cl30 ``rotor_sandwich`` MSL kernel via the GA lane.
    """
    from tessera.ga import ops as G
    return _coeffs(G.rotor_sandwich(_mv(rotor), _mv(x)))


# ── linear involutions / projections ──────────────────────────────────────── #
def clifford_reverse(a: Any) -> np.ndarray:
    from tessera.ga import ops as G
    return _coeffs(G.reverse(_mv(a)))


def clifford_grade_involution(a: Any) -> np.ndarray:
    from tessera.ga import ops as G
    return _coeffs(G.grade_involution(_mv(a)))


def clifford_conjugate(a: Any) -> np.ndarray:
    from tessera.ga import ops as G
    return _coeffs(G.conjugate(_mv(a)))


def clifford_grade_projection(a: Any, grade: int | None = None, *, k: int | None = None) -> np.ndarray:
    # `grade` may be passed positionally OR by keyword (`grade=`/`k=`). The
    # autodiff tape only captures array-like positional args + kwargs, so the
    # VJP/JVP read the grade from kwargs; calls inside a tape must therefore use
    # the keyword form (``clifford_grade_projection(a, grade=2)``).
    from tessera.ga import ops as G
    g = grade if grade is not None else k
    if g is None:
        raise TypeError("clifford_grade_projection requires a grade (positional or grade=/k=)")
    return _coeffs(G.grade_projection(_mv(a), int(g)))


# ── transcendental (exp / log) ─────────────────────────────────────────────── #
def clifford_exp(a: Any) -> np.ndarray:
    """Multivector exponential ``exp(a)`` (Cl(3,0); closed-form for pure
    bivectors, power series otherwise). Routes f32 to the cl30 ``exp`` kernel."""
    from tessera.ga import ops as G
    return _coeffs(G.exp_mv(_mv(a)))


def clifford_log(a: Any) -> np.ndarray:
    """Multivector logarithm ``log(a)`` (Cl(3,0) closed-form rotor log
    ``(θ/2)·B̂`` for scalar+bivector inputs). Routes f32 to the cl30 ``log``
    kernel."""
    from tessera.ga import ops as G
    return _coeffs(G.log_mv(_mv(a)))


# ── norms (scalar-valued) ──────────────────────────────────────────────────── #
def clifford_norm(a: Any) -> np.ndarray:
    from tessera.ga import ops as G
    return np.asarray(G.norm(_mv(a)))


def clifford_norm_squared(a: Any) -> np.ndarray:
    from tessera.ga import ops as G
    return np.asarray(G.norm_squared(_mv(a)))


# Names registered into the tessera.ops namespace (see __init__).
CLIFFORD_OPS = {
    "clifford_geometric_product": clifford_geometric_product,
    "clifford_wedge": clifford_wedge,
    "clifford_left_contraction": clifford_left_contraction,
    "clifford_inner": clifford_inner,
    "clifford_rotor_sandwich": clifford_rotor_sandwich,
    "clifford_reverse": clifford_reverse,
    "clifford_grade_involution": clifford_grade_involution,
    "clifford_conjugate": clifford_conjugate,
    "clifford_grade_projection": clifford_grade_projection,
    "clifford_exp": clifford_exp,
    "clifford_log": clifford_log,
    "clifford_norm": clifford_norm,
    "clifford_norm_squared": clifford_norm_squared,
}
