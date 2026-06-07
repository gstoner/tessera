"""Mamba-2 selective-SSD: chunked-parallel form with an injectable batched matmul.

The sequential ``selective_ssm`` reference (``tessera.ops.selective_ssm``) is an
inherently serial scan over the sequence axis. This module provides the
*chunked-parallel* reformulation that turns the heavy work into batched matmuls
— the standard Mamba-2 / SSD algorithm — so the contractions can run on a GPU
(Apple Metal `bmm`) while only a short per-chunk state recurrence stays on host.

Numerically **bit-exact** vs the sequential reference (validated to ~1e-15) for
the scalar-state case (``A`` shape ``(D,)``, the common Mamba-2 / per-head config).
The general per-state-dim ``A`` shape ``(D, N)`` does not reduce to a clean matmul
form and is left to the sequential reference (the caller falls back).

Derivation (scalar A, per channel ``d``; decay ``da_t = delta_t · A`` ≤ 0):

    Dcum_t = cumsum_{k≤t} da_k                  (inclusive, per chunk)
    y_state_t = exp(Dcum_t) · (C_t · h0)        state carried in
    M_{t,s}   = C_t · B_s                        gram   [bmm]
    seg_{t,s} = exp(Dcum_t − Dcum_s)·[s≤t]       bounded decay ∈ (0,1]
    y_local_t = Σ_{s≤t} seg_{t,s}·M_{t,s}·(δ_s x_s)
    y = y_state + y_local
    h_new = exp(Dcum_L)·h0 + Σ_s exp(Dcum_L−Dcum_s)·(δ_s x_s)·B_s   [bmm]

``seg`` is formed from the *pairwise* decay difference (≤ 0 → exp ∈ (0,1]), so it
is numerically stable — no ``exp(+large)`` overflow that the naive cumprod form
would hit.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def _np_bmm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Default batched matmul: (B, M, K) @ (B, K, N) -> (B, M, N)."""
    return np.matmul(a, b)


def supports_parallel(A: np.ndarray) -> bool:
    """Whether the chunked-parallel matmul form applies (scalar-state ``A``)."""
    A = np.asarray(A)
    return A.ndim == 1


def selective_ssm_parallel(
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    delta: np.ndarray,
    *,
    gate: Optional[np.ndarray] = None,
    state: Optional[np.ndarray] = None,
    chunk_size: int = 64,
    matmul3d: Callable[[np.ndarray, np.ndarray], np.ndarray] = _np_bmm,
) -> np.ndarray:
    """Chunked-parallel selective-SSD. ``A`` must be 1-D ``(D,)`` (scalar state).

    ``matmul3d`` is the batched-matmul backend — pass an Apple GPU `bmm` to run
    the three contractions (state projection, gram, state update) on Metal; it
    defaults to ``np.matmul`` (CPU). Returns ``y`` of shape ``(B, S, D)``.
    """
    x = np.asarray(x, dtype=np.float64)
    A1d = np.asarray(A, dtype=np.float64)
    Bt = np.asarray(B, dtype=np.float64)
    Ct = np.asarray(C, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if A1d.ndim != 1:
        raise ValueError(
            "selective_ssm_parallel requires scalar-state A of shape (D,); "
            f"got {A1d.shape}. Use the sequential reference for (D, N) A.")

    Bsz, S, D = x.shape
    N = Bt.shape[2]
    h0 = (np.zeros((Bsz, D, N)) if state is None
          else np.asarray(state, dtype=np.float64).copy())
    y = np.zeros((Bsz, S, D))
    da = delta * A1d[None, None, :]                         # (B, S, D) ≤ 0
    chunk = max(1, int(chunk_size))

    for c0 in range(0, S, chunk):
        c1 = min(S, c0 + chunk)
        L = c1 - c0
        Dcum = np.cumsum(da[:, c0:c1, :], axis=1)           # (B, L, D)
        xl = x[:, c0:c1, :]
        dl = delta[:, c0:c1, :]
        Bl = Bt[:, c0:c1, :]
        Cl = Ct[:, c0:c1, :]
        gbar = dl * xl                                      # (B, L, D)

        # state projection  sproj[b,l,d] = Σ_n Cl[b,l,n] h0[b,d,n]  = Cl @ h0ᵀ
        sproj = matmul3d(Cl, np.swapaxes(h0, -1, -2))       # (B, L, D)  [bmm]
        y_state = np.exp(Dcum) * sproj

        # gram  M[b,t,s] = Σ_n Cl[b,t,n] Bl[b,s,n]  = Cl @ Blᵀ
        M = matmul3d(Cl, np.swapaxes(Bl, -1, -2))           # (B, L, L)  [bmm]

        # bounded pairwise decay  seg[b,t,s,d] = exp(Dcum_t − Dcum_s)·[s≤t]
        diff = Dcum[:, :, None, :] - Dcum[:, None, :, :]    # (B, L, L, D)
        tri = np.tril(np.ones((L, L)))[None, :, :, None]
        seg = np.exp(diff) * tri
        y_local = np.einsum("btsd,bts,bsd->btd", seg, M, gbar)

        y[:, c0:c1, :] = y_state + y_local

        # carry state to next chunk  h_new = exp(Dcum_L)·h0 + Σ_s decay·gbar·B_s
        DL = Dcum[:, -1, :]                                 # (B, D)
        decay_s = np.exp(DL[:, None, :] - Dcum)             # (B, L, D)
        upd = matmul3d(np.swapaxes(decay_s * gbar, -1, -2), Bl)  # (B, D, N) [bmm]
        h0 = np.exp(DL)[:, :, None] * h0 + upd

    if gate is not None:
        y = y * np.asarray(gate, dtype=np.float64)
    return y.astype(np.float32)         # apple_gpu runtime convention
