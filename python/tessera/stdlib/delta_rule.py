"""``tessera.stdlib.delta_rule`` — the *true* gated delta rule (Track L, L1+L2).

Background / why this module exists
-----------------------------------
Tessera already ships ``tessera.ops.gated_deltanet`` (and the ``kimi_delta`` /
``modified_delta`` siblings), but their reference recurrence
(``tessera/__init__.py::_delta_attention_impl``) is

    Ŝ_t = α_t · Ŝ_{t-1} + β_t · k_t v_tᵀ          # additive accumulation

which is **gated linear attention**, *not* the delta rule.  The DeltaNet update
(Yang et al., "Gated Delta Networks", arXiv:2412.06464) carries an **erase**
term that removes the value currently bound to key ``k_t`` before writing the
new one:

    Ŝ_t = α_t · Ŝ_{t-1} + β_t · k_t · (v_t − α_t · v̂_t)ᵀ ,   v̂_t = Ŝ_{t-1}ᵀ k_t

The ``(v_t − α_t v̂_t)`` correction is the generalized-Householder
``(I − β_t k_t k_tᵀ)`` erase in the paper's layout.  Dropping it recovers the
existing linear-attention reference exactly — which is the reduction oracle in
``tests/unit/test_stdlib_delta_rule.py``.

This module provides that genuine rule in two algebraically-equivalent forms:

* ``gated_delta_rule_recurrent`` — the obviously-correct O(S) sequential
  recurrence (the decode form; carries a constant-size ``[d_k, d_v]`` state).
* ``gated_delta_rule_chunked`` — the chunk-parallel **UT-transform** form (the
  prefill form): everything is GEMM except one within-chunk unit-lower-triangular
  solve ``(I + A)⁻¹`` done by explicit forward substitution
  (``_forward_substitution`` — the "triangular-solve tile primitive" a real
  kernel would specialize).  The headline oracle is **chunk ≡ recurrent**.

Layout convention (matches ``_delta_attention_impl``):
    Q, K : [B, H, S, d_k]   V : [B, H, S, d_v]   state Ŝ : [B, H, d_k, d_v]
    read  O_t = q_tᵀ Ŝ_t        (state read *after* the t-th update)
    β, decay(α) : optional [B, H, S]   gate : optional, broadcastable to O

State accumulates in float64 here (the reference tier); the production contract
is fp32 accumulation regardless of bf16 storage — the erase + rank-update are
numerically sensitive.  The fused MSL kernel is the L1.1/L2.1 follow-up; this is
the reference + oracle that proves the math, provable host-free.
"""

from __future__ import annotations

import numpy as np


def _arr(x) -> np.ndarray:
    if hasattr(x, "_data"):
        x = x._data
    return np.asarray(x, dtype=np.float64)


def _per_token(x, B: int, H: int, S: int) -> np.ndarray | None:
    """Normalize an optional β / decay argument to a [B, H, S] float64 array."""
    if x is None:
        return None
    a = _arr(x)
    return np.broadcast_to(a, (B, H, S)).astype(np.float64, copy=False)


def _apply_gate(O: np.ndarray, gate, out_dtype) -> np.ndarray:
    if gate is not None:
        g = 1.0 / (1.0 + np.exp(-_arr(gate)))
        O = O * np.broadcast_to(g, O.shape)
    return O.astype(out_dtype, copy=False)


def _state_dtype(state_dtype: str, out_dtype):
    return np.float32 if state_dtype in ("fp32", "bf16") else out_dtype


# ─────────────────────────────────────────────────────────────────────────────
# L1 — the genuine recurrence (decode form, constant-size state)
# ─────────────────────────────────────────────────────────────────────────────
def gated_delta_rule_recurrent(Q, K, V, *, beta=None, decay=None, gate=None,
                               state=None, causal: bool = True,
                               return_state: bool = False,
                               state_dtype: str = "fp32",
                               erase: bool = True, backend: str = "numpy"):
    """True gated delta rule via the sequential recurrence.

    ``erase=True`` is the DeltaNet rule; ``erase=False`` degenerates to the
    existing gated-linear-attention reference (``tessera.ops.gated_deltanet``),
    which is the reduction oracle.  ``backend="apple_gpu"`` (L1.1) runs the
    genuine recurrence on Metal (``tessera_apple_gpu_gated_delta_rule_f32``),
    falling back to numpy on a Metal miss or out-of-envelope shape — the DESIL
    Metal≡numpy oracle.  The GPU path covers ``state=None`` /
    ``return_state=False`` (decode-from-zero); other cases use numpy.
    """
    if not causal:
        raise ValueError("gated_delta_rule_recurrent is a causal recurrence")
    Q, K, V = _arr(Q), _arr(K), _arr(V)
    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        raise ValueError("delta rule expects rank-4 (B, H, S, D) tensors")
    out_dtype = np.result_type(np.asarray(Q), np.asarray(K), np.asarray(V))
    B, H, S, d_k = Q.shape
    d_v = V.shape[-1]
    beta_a = _per_token(beta, B, H, S)
    decay_a = _per_token(decay, B, H, S)

    if backend == "apple_gpu" and state is None and not return_state:
        try:
            from tessera import _apple_gpu_backend as _agb
            ones = np.ones((B, H, S), dtype=np.float32)
            O = _agb.gpu_gated_delta_rule(
                Q.astype(np.float32), K.astype(np.float32), V.astype(np.float32),
                ones if beta_a is None else beta_a.astype(np.float32),
                ones if decay_a is None else decay_a.astype(np.float32),
                erase=erase).astype(np.float64)
            return _apply_gate(O, gate, out_dtype)
        except Exception:  # noqa: BLE001 — any Metal/load miss → numpy oracle
            pass

    if state is None:
        Sst = np.zeros((B, H, d_k, d_v), dtype=np.float64)
    else:
        Sst = _arr(state).copy()

    O = np.zeros((B, H, S, d_v), dtype=np.float64)
    for t in range(S):
        a = decay_a[:, :, t][:, :, None, None] if decay_a is not None else 1.0
        b = beta_a[:, :, t][:, :, None, None] if beta_a is not None else 1.0
        k_t = K[:, :, t, :]                                   # [B,H,d_k]
        v_t = V[:, :, t, :]                                   # [B,H,d_v]
        # v̂_t = Ŝ_{t-1}ᵀ k_t  — the value currently bound to key k_t.
        v_hat = np.einsum("bhd,bhde->bhe", k_t, Sst)          # [B,H,d_v]
        a_s = decay_a[:, :, t][:, :, None] if decay_a is not None else 1.0
        target = v_t - a_s * v_hat if erase else v_t          # erase correction
        Sst = a * Sst + b * np.einsum("bhd,bhe->bhde", k_t, target)
        O[:, :, t, :] = np.einsum("bhd,bhde->bhe", Q[:, :, t, :], Sst)

    O = _apply_gate(O, gate, out_dtype)
    Sst = Sst.astype(_state_dtype(state_dtype, out_dtype), copy=False)
    return (O, Sst) if return_state else O


# ─────────────────────────────────────────────────────────────────────────────
# L2 — the chunk-parallel UT-transform (prefill form)
# ─────────────────────────────────────────────────────────────────────────────
def _forward_substitution(A_strict: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Solve (I + A) U = W for U, where A is strictly lower-triangular (so
    (I + A) is unit lower-triangular).  Explicit forward substitution — the
    "triangular-solve tile primitive"; the only non-GEMM step of the chunk form.

    A_strict : [..., C, C]   W : [..., C, d]   ->  U : [..., C, d]
    """
    C = A_strict.shape[-1]
    U = np.array(W, dtype=np.float64, copy=True)
    for t in range(C):
        if t > 0:
            # U[t] = W[t] - Σ_{j<t} A[t, j] · U[j]
            corr = np.einsum("...j,...jd->...d", A_strict[..., t, :t], U[..., :t, :])
            U[..., t, :] = W[..., t, :] - corr
    return U


def gated_delta_rule_chunked(Q, K, V, *, beta=None, decay=None, gate=None,
                             state=None, chunk_size: int = 64,
                             causal: bool = True, return_state: bool = False,
                             state_dtype: str = "fp32", erase: bool = True,
                             backend: str = "numpy"):
    """True gated delta rule via the chunk-parallel UT-transform.

    Per chunk (carried state Ŝ₀, cumulative within-chunk decay γ_t = Π_{i≤t} α_i):

        Ã[t,j] = β_t (γ_t/γ_j)(k_tᵀ k_j)   for j<t        (strictly lower)
        W̃[t]  = β_t (v_t − γ_t · Ŝ₀ᵀ k_t)                  (delta target)
        Ũ      = (I + Ã)⁻¹ W̃                               (forward substitution)
        Ŝ_C    = γ_C Ŝ₀ + Kᵀ · diag(γ_C/γ_t) · Ũ           (state carry)
        O_t    = γ_t q_tᵀ Ŝ₀ + Σ_{j≤t}(γ_t/γ_j)(q_tᵀ k_j) ũ_j

    Algebraically identical to ``gated_delta_rule_recurrent`` (the chunk≡recurrent
    oracle); every step is GEMM except the (I + Ã)⁻¹ solve.  ``backend="apple_gpu"``
    (L2.1) runs the chunk form on Metal (one threadgroup per (b,h), the within-chunk
    UT solve on-device), falling back to numpy on a Metal miss / out-of-envelope
    shape (``state=None``, ``return_state=False``, ``chunk_size ≤ 32``, d ≤ 16).
    """
    if not causal:
        raise ValueError("gated_delta_rule_chunked is a causal recurrence")
    Q, K, V = _arr(Q), _arr(K), _arr(V)
    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        raise ValueError("delta rule expects rank-4 (B, H, S, D) tensors")
    out_dtype = np.result_type(np.asarray(Q), np.asarray(K), np.asarray(V))
    B, H, S, d_k = Q.shape
    d_v = V.shape[-1]
    beta_a = _per_token(beta, B, H, S)
    decay_a = _per_token(decay, B, H, S)

    if (backend == "apple_gpu" and state is None and not return_state
            and chunk_size <= 32 and d_k <= 16 and d_v <= 16):
        try:
            from tessera import _apple_gpu_backend as _agb
            ones = np.ones((B, H, S), dtype=np.float32)
            O = _agb.gpu_gated_delta_rule_chunked(
                Q.astype(np.float32), K.astype(np.float32), V.astype(np.float32),
                ones if beta_a is None else beta_a.astype(np.float32),
                ones if decay_a is None else decay_a.astype(np.float32),
                chunk=chunk_size, erase=erase).astype(np.float64)
            return _apply_gate(O, gate, out_dtype)
        except Exception:  # noqa: BLE001 — any Metal/load miss → numpy oracle
            pass

    if beta_a is None:
        beta_a = np.ones((B, H, S), dtype=np.float64)

    Sst = (np.zeros((B, H, d_k, d_v), dtype=np.float64)
           if state is None else _arr(state).copy())
    O = np.zeros((B, H, S, d_v), dtype=np.float64)

    for c0 in range(0, S, chunk_size):
        c1 = min(c0 + chunk_size, S)
        C = c1 - c0
        Qc = Q[:, :, c0:c1, :]                                # [B,H,C,d_k]
        Kc = K[:, :, c0:c1, :]
        Vc = V[:, :, c0:c1, :]
        bc = beta_a[:, :, c0:c1]                              # [B,H,C]

        # γ_t = Π_{i≤t} α_i within the chunk (γ for the ungated case is all-ones).
        if decay_a is not None:
            gamma = np.cumprod(decay_a[:, :, c0:c1], axis=2)  # [B,H,C]
        else:
            gamma = np.ones((B, H, C), dtype=np.float64)
        gamma_C = gamma[:, :, -1]                             # [B,H]

        KKt = np.einsum("bhtd,bhjd->bhtj", Kc, Kc)            # [B,H,C,C]
        ratio = gamma[:, :, :, None] / gamma[:, :, None, :]   # γ_t/γ_j
        tril_strict = np.tril(np.ones((C, C)), k=-1)
        A_strict = (bc[:, :, :, None] * ratio * KKt) * tril_strict if erase \
            else np.zeros((B, H, C, C))

        # Target W̃ = β (V − γ · Ŝ₀ᵀ K).
        KS0 = np.einsum("bhtd,bhde->bhte", Kc, Sst)           # k_tᵀ Ŝ₀  -> [B,H,C,d_v]
        if erase:
            W = bc[:, :, :, None] * (Vc - gamma[:, :, :, None] * KS0)
        else:
            W = bc[:, :, :, None] * Vc
        U = _forward_substitution(A_strict, W)                # [B,H,C,d_v]

        # Output: O = diag(γ) Q Ŝ₀ + (tril(QKᵀ,0) ⊙ Γ) U.
        QS0 = np.einsum("bhtd,bhde->bhte", Qc, Sst)           # [B,H,C,d_v]
        QKt = np.einsum("bhtd,bhjd->bhtj", Qc, Kc)            # [B,H,C,C]
        tril_incl = np.tril(np.ones((C, C)), k=0)
        Mintra = QKt * ratio * tril_incl                      # γ_t/γ_j on j≤t
        O[:, :, c0:c1, :] = (gamma[:, :, :, None] * QS0
                             + np.einsum("bhtj,bhje->bhte", Mintra, U))

        # State carry: Ŝ_C = γ_C Ŝ₀ + Kᵀ diag(γ_C/γ_t) U.
        scale = (gamma_C[:, :, None] / gamma)                 # [B,H,C]
        Sst = (gamma_C[:, :, None, None] * Sst
               + np.einsum("bhtd,bhte->bhde", Kc, scale[:, :, :, None] * U))

    O = _apply_gate(O, gate, out_dtype)
    Sst = Sst.astype(_state_dtype(state_dtype, out_dtype), copy=False)
    return (O, Sst) if return_state else O
