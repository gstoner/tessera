"""Coalition-lattice transforms (G1): zeta/Möbius butterflies + marginals.

Reference semantics seeded from the numerically verified harness
(`research/game_theory/verify_game_theory_plan.py`, 27 checks). Each transform
is the Yates radix-2 butterfly — ``n`` stages, stride ``2^i`` at stage ``i`` —
differing only in the per-stage 2×2 kernel (GAME_THEORY_PLAN.md §4.6.2):

    subset zeta      [[1,0],[1,1]]      (ζv)(T) = Σ_{S⊆T} v(S)
    subset Möbius    [[1,0],[-1,1]]     its inverse
    superset zeta    [[1,1],[0,1]]      the transpose of subset zeta
    superset Möbius  [[1,-1],[0,1]]     the transpose of subset Möbius

The transpose pairs are declared via ``def_transpose`` (through the
``transpose_rule=`` kwarg), which for a ``linear=True`` primitive wires the VJP
and derives the JVP — giving the registry's ``transpose_rule`` axis a genuine
consumer (Decision #29) and making oracle 2 (⟨ζv, w⟩ = ⟨v, ζᵀw⟩) load-bearing.

Numerics: forward results are float64 by mandate (plan §6 — the fp32 wall for
nonnegative games is at ``n ≈ 16–20`` and it is a *storage* wall, not an
accumulator wall).
"""

from __future__ import annotations

import numpy as np

from ..custom import custom_primitive


def lattice_players(x: np.ndarray) -> int:
    """Derive ``n`` from the trailing lattice axis; fail closed otherwise.

    ``2^n == shape[-1]`` is verified, never assumed (G0 contract): a lattice
    tensor whose trailing extent is not a power of two is an error, not a
    truncation.
    """
    size = int(np.asarray(x).shape[-1])
    if size <= 0 or (size & (size - 1)) != 0:
        raise ValueError(
            f"coalition-lattice axis must have power-of-two extent 2^n; got "
            f"{size} — n is derived from the shape and never defaulted")
    return size.bit_length() - 1


def _butterfly(x: np.ndarray, *, half: int, sign: float) -> np.ndarray:
    """Shared Yates recurrence. ``half`` selects which half accumulates
    (1 = subset transforms, 0 = superset transforms); ``sign`` is the kernel's
    off-diagonal entry (+1 zeta, −1 Möbius)."""
    out = np.array(x, dtype=np.float64, copy=True)
    n = lattice_players(out)
    other = 1 - half
    for i in range(n):
        g = out.reshape(-1, out.shape[-1] >> (i + 1), 2, 1 << i)
        g[:, :, half, :] += sign * g[:, :, other, :]
    return out


def _subset_zeta_impl(v: np.ndarray) -> np.ndarray:
    return _butterfly(v, half=1, sign=+1.0)


def _subset_mobius_impl(v: np.ndarray) -> np.ndarray:
    return _butterfly(v, half=1, sign=-1.0)


def _superset_zeta_impl(v: np.ndarray) -> np.ndarray:
    return _butterfly(v, half=0, sign=+1.0)


def _superset_mobius_impl(v: np.ndarray) -> np.ndarray:
    return _butterfly(v, half=0, sign=-1.0)


def _coalition_marginal_impl(v: np.ndarray) -> np.ndarray:
    """``∂_i v(S) = v(S) − v(S Δ {i})`` for every player: [..., 2^n] →
    [..., n, 2^n]. The bit-flip is a symmetric permutation, so the map is
    linear and self-adjoint-per-player."""
    v = np.asarray(v, dtype=np.float64)
    n = lattice_players(v)
    size = v.shape[-1]
    idx = np.arange(size)
    out = np.empty(v.shape[:-1] + (n, size), dtype=np.float64)
    for i in range(n):
        out[..., i, :] = v - v[..., idx ^ (1 << i)]
    return out


def _coalition_marginal_transpose(dout: np.ndarray, v: np.ndarray,
                                  **_kw: object) -> np.ndarray:
    dout = np.asarray(dout, dtype=np.float64)
    n = dout.shape[-2]
    size = dout.shape[-1]
    idx = np.arange(size)
    grad = np.zeros(dout.shape[:-2] + (size,), dtype=np.float64)
    for i in range(n):
        di = dout[..., i, :]
        grad += di - di[..., idx ^ (1 << i)]
    return grad


# Transpose pairs (oracle 2 made load-bearing): ζ_sub ↔ ζ_sup, μ_sub ↔ μ_sup.
subset_zeta = custom_primitive(
    "game_subset_zeta", linear=True,
    transpose_rule=lambda dout, v, **_kw: _superset_zeta_impl(dout),
)(_subset_zeta_impl)

subset_mobius = custom_primitive(
    "game_subset_mobius", linear=True,
    transpose_rule=lambda dout, v, **_kw: _superset_mobius_impl(dout),
)(_subset_mobius_impl)

superset_zeta = custom_primitive(
    "game_superset_zeta", linear=True,
    transpose_rule=lambda dout, v, **_kw: _subset_zeta_impl(dout),
)(_superset_zeta_impl)

superset_mobius = custom_primitive(
    "game_superset_mobius", linear=True,
    transpose_rule=lambda dout, v, **_kw: _subset_mobius_impl(dout),
)(_superset_mobius_impl)

coalition_marginal = custom_primitive(
    "game_coalition_marginal", linear=True,
    transpose_rule=_coalition_marginal_transpose,
)(_coalition_marginal_impl)
