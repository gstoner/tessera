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

Numerics: fp64 ACCUMULATION by mandate with the input's storage dtype
preserved (Decision #15a — storage lives on the tensor; §6's fp32 wall for
nonnegative games past ``n ≈ 16`` is a *materialization* hazard owned by the
fused lowering / H1's narrow-materialization diagnostic, not by a silent
storage promotion in the reference tier).
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


def _storage_dtype(x: np.ndarray) -> np.dtype:
    """Result STORAGE dtype: the input's floating dtype, or float64 for
    non-floating inputs. Decision #15a — storage lives on the tensor and the
    lattice family's fp64 mandate is an ACCUMULATION contract
    (`numeric_policy.accum = fp64`), not a storage rewrite. The §6 hazard
    (fp32-materialized zeta is noise for nonnegative games past n ≈ 16) is
    handled where the plan puts it: fused consumers / a diagnostic on narrow
    materialization at lowering (H1), never a silent storage promotion here."""
    dt = np.asarray(x).dtype
    # Preserve every floating storage dtype INCLUDING the ml_dtypes reduced
    # precisions (bf16 et al.), whose numpy kind is not 'f' — upcasting out of
    # reduced precision is exactly the bug the reduced-precision drift gate
    # exists to catch. Only integer/bool inputs promote (to the fp64 oracle).
    if dt.kind in ("i", "u", "b"):
        return np.dtype(np.float64)
    return dt


def _butterfly(x: np.ndarray, *, half: int, sign: float) -> np.ndarray:
    """Shared Yates recurrence. ``half`` selects which half accumulates
    (1 = subset transforms, 0 = superset transforms); ``sign`` is the kernel's
    off-diagonal entry (+1 zeta, −1 Möbius). Accumulates in float64 (§6),
    returns in the input's storage dtype (#15a)."""
    out = np.array(x, dtype=np.float64, copy=True)
    n = lattice_players(out)
    other = 1 - half
    for i in range(n):
        g = out.reshape(-1, out.shape[-1] >> (i + 1), 2, 1 << i)
        g[:, :, half, :] += sign * g[:, :, other, :]
    return out.astype(_storage_dtype(x), copy=False)


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
    orig = v
    v = np.asarray(v, dtype=np.float64)
    n = lattice_players(v)
    size = v.shape[-1]
    idx = np.arange(size)
    out = np.empty(v.shape[:-1] + (n, size), dtype=np.float64)
    for i in range(n):
        out[..., i, :] = v - v[..., idx ^ (1 << i)]
    return out.astype(_storage_dtype(orig), copy=False)


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
