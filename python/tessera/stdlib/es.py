"""``tessera.stdlib.es`` — Evolution-Strategies reference tier (EGGROLL, Track ES).

Background / why this module exists
-----------------------------------
Evolution Strategies is a zeroth-order optimizer: perturb the weights, score a
scalar *fitness* per population member, move the mean toward high-fitness
directions. EGGROLL (*Evolution Strategies at the Hyperscale*, arXiv:2511.16652)
makes it viable at scale with a **low-rank** perturbation ``E = (1/√r) A Bᵀ``,
turning the population forward into batched-LoRA:

    u(M + σE)ᵀ  =  u·Mᵀ  +  (σ/√r)·(u·B)·Aᵀ          (Thm 1, exact)

and the update into a materialization-free GEMM:

    ΔM = (1/N) Σ_i f_i · sign_i · A_i B_iᵀ / √r         (Thm 2, exact)

This is the **reference tier** (Decision #28 tier-1): numpy semantics + host-free
oracles, no IR and no hardware — the analogue of ``delta_rule``'s recurrent form.
The fused Graph-IR op ``tessera.es_low_rank_correction`` and its emitters are the
W2 follow-up; this module is the math the op must reproduce, and the oracle that
proves it (``tests/unit/test_es_reference.py``). Full plan + proofs:
``docs/audit/compiler/EGGROLL_SUPPORT_PLAN.md``.

Three contracts the design audit forced out, honored here:

* **Member-keyed RNG (G2).** Factors are reconstructed from
  ``key.fold_in(epoch).fold_in(pair)`` — the population member (pair), never the
  device rank, is folded in. Every worker reconstructing member ``m`` gets
  bit-identical ``(A, B)``. This is the *opposite* of Decision #18's per-rank
  device stream; the two RNG axes are orthogonal and must not be confused.
* **``σ`` is a ``σ√d``-regime semantic (G4).** The linearization that makes the
  update consistent needs ``σ√(mn) = o(1)``; ``sigma`` is required (no default).
* **fp32/s32 accumulation (I6 / Decision #32).** ``es_update`` sums over a large
  population; bf16 accumulation loses the signal to cancellation. The reference
  accumulates in float64; the production contract is fp32 (int lane: s32).

Score linearity is a *requirement*, not a choice (verified): a nonlinear score
destroys the rank-r structure that carries the gradient and forfeits the
factorization, so the Gaussian score ``Ŝ(E) = -E`` is the only one supported.

Layout convention (matches ``stdlib`` linear layers):
    weight ``M`` : [out, in]      activation ``x`` : [P, ..., in]
    factors      A : [out, r]     B : [in, r]      E = A Bᵀ / √r : [out, in]
    member ``i`` -> pair = i // 2, sign = +1 if i even else -1  (antithetic).
"""

from __future__ import annotations

from typing import Sequence

import ml_dtypes
import numpy as np

from .. import rng as _rng
from ..dtype import canonicalize_dtype

__all__ = [
    "reconstruct_factors",
    "low_rank_perturbation",
    "population_forward",
    "es_update",
    "fitness_shaping",
    "centered_rank",
    "antithetic_sign",
]

def _as_key(key) -> "_rng.RNGKey":
    """Accept an ``RNGKey`` or a bare integer seed."""
    if isinstance(key, _rng.RNGKey):
        return key
    return _rng.RNGKey.from_seed(int(key))


def reconstruct_factors(key, member: int, out_dim: int, in_dim: int, rank: int,
                        *, epoch: int = 0, antithetic: bool = True):
    """Member-keyed low-rank factors ``(A[out,r], B[in,r], sign)`` (G2).

    One draw of shape ``(in+out, r)`` is sliced ``B = draw[:in]``, ``A =
    draw[in:]`` (matches the EGGROLL reference). The RNG key folds in ``epoch``
    then the antithetic ``pair = member // 2`` — **never the device rank** — so
    the factors are identical on every worker that reconstructs this member.
    ``sign`` is ``+1`` for even members and ``-1`` for odd (antithetic pairs
    share a direction with opposite sign); with ``antithetic=False`` every member
    is its own direction with ``sign = +1``.
    """
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    pair = (member // 2) if antithetic else member
    sign = 1.0 if (not antithetic or member % 2 == 0) else -1.0
    child = _as_key(key).fold_in(int(epoch)).fold_in(int(pair))
    draw = _rng.normal(child, (in_dim + out_dim, rank), dtype="fp32")
    return draw[in_dim:], draw[:in_dim], sign          # A, B, sign


def low_rank_perturbation(key, member: int, out_dim: int, in_dim: int, rank: int,
                          *, sigma: float, epoch: int = 0, antithetic: bool = True,
                          dtype: str = "fp32") -> np.ndarray:
    """The (materialized) perturbation ``sign · σ · A Bᵀ / √r`` for one member.

    Reference/ground-truth only — the whole point of EGGROLL is that this ``[out,
    in]`` matrix is *never* formed in the fast path (``population_forward`` and
    ``es_update`` keep the factored form). ``sigma`` is required (G4).
    """
    if sigma is None:
        raise ValueError("sigma is semantic and must be provided (G4)")
    A, B, sign = reconstruct_factors(key, member, out_dim, in_dim, rank,
                                     epoch=epoch, antithetic=antithetic)
    E = sign * float(sigma) * (A @ B.T) / np.sqrt(rank)
    return E.astype(_np_storage(dtype))


def population_forward(x: np.ndarray, weight: np.ndarray, key,
                       members: Sequence[int], *, sigma: float, rank: int,
                       epoch: int = 0, antithetic: bool = True) -> np.ndarray:
    """Batched-LoRA population forward ``x·Mᵀ + (σ/√r)·sign·(x·B)·Aᵀ`` (Thm 1).

    ``x`` is ``[P, in]`` (one activation row per population member); ``weight`` is
    ``[out, in]``; ``members`` is the length-``P`` list of member ids this call
    evaluates. The base ``x·Mᵀ`` is a single shared GEMM (high arithmetic
    intensity); only the low-rank correction is per-member. Returns ``[P, out]``.
    """
    if sigma is None:
        raise ValueError("sigma is semantic and must be provided (G4)")
    x = np.asarray(x, dtype=np.float64)
    M = np.asarray(weight, dtype=np.float64)
    out_dim, in_dim = M.shape
    if x.shape[0] != len(members):
        raise ValueError(
            f"population axis mismatch: x has {x.shape[0]} rows, "
            f"{len(members)} members")
    base = x @ M.T
    corr = np.zeros_like(base)
    scale = float(sigma) / np.sqrt(rank)
    for p, mem in enumerate(members):
        A, B, sign = reconstruct_factors(key, mem, out_dim, in_dim, rank,
                                         epoch=epoch, antithetic=antithetic)
        corr[p] = sign * scale * ((x[p] @ B) @ A.T)
    return base + corr


def es_update(key, members: Sequence[int], fitness: np.ndarray,
              out_dim: int, in_dim: int, *, sigma: float, rank: int,
              epoch: int = 0, antithetic: bool = True,
              accum: str = "fp32") -> np.ndarray:
    """Materialization-free ES update ``(1/N) Σ_i f_i·sign_i·A_iB_iᵀ / √r`` (Thm 2).

    Computed as the factored GEMM ``Ā B̄ᵀ`` (contraction dim ``N·r``) — no
    per-member ``E_i`` is ever formed. ``fitness`` is the length-``P`` shaped
    fitness (see :func:`fitness_shaping`). ``accum`` is the accumulation-width
    contract (I6 / Decision #32): the reference implements ``fp32`` (float64
    here); ``s32`` (the EGG integer lane) is reserved and raises
    ``NotImplementedError`` until that track lands; bf16 is rejected because the
    sum over the population cancels. Returns the ``[out, in]`` update (rank
    ``min(⌈N/2⌉·r, out, in)`` under antithetic pairing).
    """
    if sigma is None:
        raise ValueError("sigma is semantic and must be provided (G4)")
    if accum in ("s32", "int32"):
        raise NotImplementedError(
            "integer (s32) accumulation is the EGG integer lane (W-later); the "
            "reference tier implements the fp32 lane only (I6 / Decision #32)")
    if accum not in ("fp32", "f32"):
        raise ValueError(f"accum must be 'fp32' or 's32' (I6), got {accum!r}")
    P = len(members)
    f = np.asarray(fitness, dtype=np.float64).reshape(P)
    A = np.empty((P, out_dim, rank)); B = np.empty((P, in_dim, rank)); s = np.empty(P)
    for p, mem in enumerate(members):
        A[p], B[p], s[p] = reconstruct_factors(key, mem, out_dim, in_dim, rank,
                                               epoch=epoch, antithetic=antithetic)
    w = (f * s) / np.sqrt(rank) / P                    # per-member scalar
    # factored contraction over (member, rank) — the Ā B̄ᵀ form (Thm 2)
    return np.einsum('nir,njr->ij', w[:, None, None] * A, B)


# ------------------------------------------------------------- fitness shaping
def centered_rank(scores: np.ndarray) -> np.ndarray:
    """Centered-rank transform to ``[-0.5, 0.5]`` (Salimans et al. 2017).

    Rank-based shaping is scale-free and robust to fitness outliers; ranks are
    normalized to ``[0, 1]`` then shifted to zero mean.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    order = np.empty(len(s), dtype=np.int64)
    order[np.argsort(s, kind="stable")] = np.arange(len(s))
    if len(s) <= 1:
        return np.zeros_like(s)
    return order / (len(s) - 1) - 0.5


def antithetic_sign(s_plus: np.ndarray, s_minus: np.ndarray) -> np.ndarray:
    """Ternary antithetic shaping ``sign(s⁺ − s⁻) ∈ {-1, 0, +1}`` per pair.

    The 1-bit-per-pair signal used by the integer / moment-free path — throws
    away magnitude for stability and cheap (base-3 packable) communication.
    """
    return np.sign(np.asarray(s_plus, dtype=np.float64)
                   - np.asarray(s_minus, dtype=np.float64))


def fitness_shaping(scores: np.ndarray, *, method: str = "zscore",
                    group_axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Shape raw fitnesses into an update weighting.

    ``method``:
      * ``"zscore"``  — group-relative standardization; delegates to
        ``rl.normalize_group_advantages`` so ES and GRPO share one implementation
        (operator improvement O6). ``group_axis`` selects the group dimension.
      * ``"centered_rank"`` — :func:`centered_rank` (flat, scale-free).

    (Antithetic ternary shaping is :func:`antithetic_sign`, which takes paired
    scores and so has a distinct signature.)
    """
    if method == "centered_rank":
        return centered_rank(scores)
    if method == "zscore":
        from ..rl import normalize_group_advantages   # lazy: avoid import cycle
        return normalize_group_advantages(scores, group_axis=group_axis, eps=eps)
    raise ValueError(f"unknown fitness-shaping method {method!r} "
                     "(expected 'zscore' or 'centered_rank')")


_STORAGE_NP = {"fp16": np.float16, "bf16": ml_dtypes.bfloat16,
               "fp32": np.float32, "fp64": np.float64}


def _np_storage(dtype: str):
    """Canonical storage dtype → numpy dtype; rejects unsupported spellings.

    Normalizes through the canonical registry (``f16``→``fp16`` etc.) and maps to
    *real* numpy storage — ``fp16`` keeps ``np.float16`` rounding, ``bf16`` uses
    ``ml_dtypes.bfloat16`` — instead of silently widening to float32 (Decision
    #21a: no silent fallthrough)."""
    canon = canonicalize_dtype(str(dtype))
    if canon not in _STORAGE_NP:
        raise ValueError(
            f"storage dtype {dtype!r} (→ {canon}) is unsupported by the ES "
            f"reference tier; expected one of {sorted(_STORAGE_NP)}")
    return _STORAGE_NP[canon]
