"""Track-R (ReplaySSM) Phase 6 — Gated DeltaNet decode-state handle.

The SSM analogue of :class:`SSMStateHandle` for the **gated delta rule**
(DeltaNet — Yang et al., arXiv:2412.06464), the other linear-recurrent family
ReplaySSM accelerates.  Unlike Mamba-2's diagonal scalar-A decay, the delta
rule's transition is matrix-valued (a gated generalized-Householder):

    Ŝ_t = α_t (I − β_t k_t k_tᵀ) Ŝ_{t-1} + β_t k_t v_tᵀ
        = α_t Ŝ_{t-1} + β_t k_t (v_t − α_t·v̂_t)ᵀ,   v̂_t = Ŝ_{t-1}ᵀ k_t
    O_t = q_tᵀ Ŝ_t

Because the transition is non-commutative, there is no closed-form decay-sum
reconstruction (the Mamba-2 inner-product-first trick does not apply).  The
replay win is still real and identical in spirit: keep a checkpoint state ``S0``
plus a small ring buffer of recent ``(k, v, β, α)`` and **replay the short
recurrence** from ``S0`` to produce outputs — never *writing* the new ``(d_k,
d_v)`` state to memory every token (that happens only at flush).  Speculative
rollback is a cursor rewind, exactly as for Mamba-2.

This is the *reference* handle (float64 accumulation, matching
``stdlib.delta_rule``); the fused on-device decode kernel is a follow-up.  The
reconstruction is exact vs ``stdlib.delta_rule.gated_delta_rule_recurrent``.

Layout (matches ``stdlib.delta_rule``):
    q, k : [B, H, d_k]   v : [B, H, d_v]   state Ŝ : [B, H, d_k, d_v]
    per-token β, α (decay) : [B, H]   read O_t = q_tᵀ Ŝ_t (state *after* update)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import numpy as np


def _to_array(value: Any) -> np.ndarray:
    if hasattr(value, "_data"):
        value = value._data
    return np.asarray(value)


@dataclass
class DeltaNetStateHandle:
    """Opaque decode-state handle for the gated delta rule (DeltaNet).

    Parameters
    ----------
    batch, num_heads, key_dim, value_dim : int
        ``B``, ``H``, ``d_k``, ``d_v``.
    capacity : int
        Ring-buffer length ``L`` — max live replay tokens between flushes.
    dtype : str
        Canonical storage dtype for the ABI (default ``"fp32"``); internal
        accumulation is float64 (reference tier).
    spec_window : int
        Speculative window ``T`` for the flush rule (``count + 2T + n > L``).
    erase : bool
        ``True`` → the genuine delta rule (with the erase correction);
        ``False`` → gated linear attention (the reduction oracle).
    """

    batch: int
    num_heads: int
    key_dim: int
    value_dim: int
    capacity: int = 64
    dtype: str = "fp32"
    spec_window: int = 0
    erase: bool = True
    #: Optional block decode kernel (Track-R Phase 6).  Signature
    #: ``(Q(B,H,T,dk), K(B,H,T,dk), V(B,H,T,dv), beta(B,H,T), decay(B,H,T),
    #: S0(B,H,dk,dv), B,H,T,dk,dv,erase) -> (O(B,H,T,dv), Sout(B,H,dk,dv))``.
    block_fn: Optional[Callable[..., Any]] = field(default=None, repr=False)
    #: Provenance tag: "reference" (host numpy) or "apple_gpu" (block kernel).
    backend: str = "reference"

    _s0: np.ndarray = field(init=False, repr=False)
    _k: np.ndarray = field(init=False, repr=False)
    _v: np.ndarray = field(init=False, repr=False)
    _beta: np.ndarray = field(init=False, repr=False)
    _decay: np.ndarray = field(init=False, repr=False)
    _count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        from ..dtype import canonicalize_dtype

        object.__setattr__(
            self, "dtype", canonicalize_dtype(self.dtype, allow_planned_gated=True)
        )
        B, H, dk, dv, L = (self.batch, self.num_heads, self.key_dim,
                           self.value_dim, self.capacity)
        if min(B, H, dk, dv) <= 0:
            raise ValueError(
                f"batch/num_heads/key_dim/value_dim must be positive, "
                f"got ({B}, {H}, {dk}, {dv})"
            )
        if L <= 0:
            raise ValueError(f"capacity must be positive, got {L}")
        if self.spec_window < 0:
            raise ValueError(f"spec_window must be >= 0, got {self.spec_window}")
        self._s0 = np.zeros((B, H, dk, dv), dtype=np.float64)
        self._k = np.zeros((L, B, H, dk), dtype=np.float64)
        self._v = np.zeros((L, B, H, dv), dtype=np.float64)
        self._beta = np.ones((L, B, H), dtype=np.float64)
        self._decay = np.ones((L, B, H), dtype=np.float64)
        self._count = 0

    # ── Read-only views ─────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_full(self) -> bool:
        return self._count >= self.capacity

    @property
    def checkpoint_state(self) -> np.ndarray:
        """The folded checkpoint state ``S0`` (B, H, d_k, d_v)."""
        return self._s0.copy()

    # ── Flush policy / route selection (shared contract) ────────────────

    def should_flush(self, n_new: int = 1) -> bool:
        from ..compiler.ssm_replay import should_flush as _should_flush

        return _should_flush(self._count, self.capacity, self.spec_window, n_new)

    def route_for(self, n_new: int = 1) -> str:
        from ..compiler.ssm_replay import select_route

        return select_route(self._count, self.capacity, self.spec_window, n_new)

    # ── Recurrence replay (the shared kernel of every route) ────────────

    def materialize_state(self) -> np.ndarray:
        """Replay the gated delta-rule recurrence from ``S0`` over the live
        buffer → the *live* state ``Ŝ`` (B, H, d_k, d_v), without mutating the
        handle.  O(count) rank-1 updates — the replay cost."""
        S = self._s0.copy()
        for i in range(self._count):
            k = self._k[i]                                 # (B, H, d_k)
            v = self._v[i]                                 # (B, H, d_v)
            a = self._decay[i][..., None, None]            # (B, H, 1, 1)
            b = self._beta[i][..., None, None]
            if self.erase:
                v_hat = np.einsum("bhd,bhde->bhe", k, S)   # Ŝᵀ k
                a_s = self._decay[i][..., None]            # (B, H, 1)
                target = v - a_s * v_hat
            else:
                target = v
            S = a * S + b * np.einsum("bhd,bhe->bhde", k, target)
        return S

    # ── Append + read (the decode step) ─────────────────────────────────

    def append(
        self,
        k_t: np.ndarray,
        v_t: np.ndarray,
        *,
        beta_t: np.ndarray | float | None = None,
        decay_t: np.ndarray | float | None = None,
        auto_flush: bool = True,
    ) -> "DeltaNetStateHandle":
        """Append one token's replay inputs ``(k, v, β, α)`` to the ring buffer.

        ``k_t`` is ``(B, H, d_k)``; ``v_t`` is ``(B, H, d_v)``; ``β``/``α`` are
        ``(B, H)`` (or scalars, default 1).  Overflow flushes first when
        ``auto_flush`` is set, else raises.
        """
        B, H, dk, dv = self.batch, self.num_heads, self.key_dim, self.value_dim
        k = _to_array(k_t).astype(np.float64).reshape(B, H, dk)
        v = _to_array(v_t).astype(np.float64).reshape(B, H, dv)
        beta = (np.broadcast_to(np.float64(1.0) if beta_t is None
                                else _to_array(beta_t).astype(np.float64), (B, H)))
        decay = (np.broadcast_to(np.float64(1.0) if decay_t is None
                                 else _to_array(decay_t).astype(np.float64), (B, H)))
        if self._count >= self.capacity:
            if auto_flush:
                self.flush()
            else:
                raise ValueError(
                    f"DeltaNetStateHandle ring buffer full (capacity="
                    f"{self.capacity}); flush() or pass auto_flush=True"
                )
        i = self._count
        self._k[i] = k
        self._v[i] = v
        self._beta[i] = beta
        self._decay[i] = decay
        self._count = i + 1
        return self

    def read_output(
        self, q_t: np.ndarray, *, gate_t: np.ndarray | None = None
    ) -> np.ndarray:
        """Replay-then-read: ``O_t = q_tᵀ Ŝ_t`` (B, H, d_v) for the most recently
        appended token.  Replays the recurrence from ``S0`` to obtain ``Ŝ_t``
        (transient — not persisted, the output-only route).  Optional sigmoid
        ``gate_t`` ``(B, H, d_v)`` is applied as in ``gated_delta_rule``."""
        q = _to_array(q_t).astype(np.float64).reshape(
            self.batch, self.num_heads, self.key_dim
        )
        S = self.materialize_state()
        o = np.einsum("bhd,bhde->bhe", q, S)               # (B, H, d_v)
        if gate_t is not None:
            g = _to_array(gate_t).astype(np.float64)
            o = o * (1.0 / (1.0 + np.exp(-np.broadcast_to(g, o.shape))))
        return o

    def step(
        self,
        q_t: np.ndarray,
        k_t: np.ndarray,
        v_t: np.ndarray,
        *,
        beta_t: np.ndarray | float | None = None,
        decay_t: np.ndarray | float | None = None,
        gate_t: np.ndarray | None = None,
    ) -> np.ndarray:
        """One decode token: append ``(k, v, β, α)`` then read ``O_t = q·Ŝ_t``.

        Flushes first when the flush rule fires.  Returns ``O_t`` (B, H, d_v).
        """
        if self.should_flush(1):
            self.flush()
        self.append(k_t, v_t, beta_t=beta_t, decay_t=decay_t, auto_flush=False)
        return self.read_output(q_t, gate_t=gate_t)

    # ── Block decode (dispatch-overhead fix) ────────────────────────────

    def decode_block(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        *,
        beta: np.ndarray | None = None,
        decay: np.ndarray | None = None,
    ) -> np.ndarray:
        """Replay a block of ``T`` tokens from the current state in one shot —
        the prefill / speculative-verification / benchmark path.

        ``Q``/``K`` are ``(B, H, T, d_k)``; ``V`` is ``(B, H, T, d_v)``;
        ``beta``/``decay`` are ``(B, H, T)`` (default 1).  Returns the per-token
        outputs ``O`` ``(B, H, T, d_v)``.  Pure — does not mutate the handle.
        """
        B, H, dk, dv = self.batch, self.num_heads, self.key_dim, self.value_dim
        Q = _to_array(Q).astype(np.float64)
        K = _to_array(K).astype(np.float64)
        V = _to_array(V).astype(np.float64)
        T = Q.shape[2]
        beta_a = (np.ones((B, H, T)) if beta is None
                  else np.broadcast_to(_to_array(beta).astype(np.float64), (B, H, T)))
        decay_a = (np.ones((B, H, T)) if decay is None
                   else np.broadcast_to(_to_array(decay).astype(np.float64), (B, H, T)))
        S0 = self.materialize_state()                      # current live state
        if self.block_fn is not None:
            res = self.block_fn(Q, K, V, beta_a, decay_a, S0,
                                B, H, T, dk, dv, int(self.erase))
            if res is not None:
                O, _Sout = res
                return np.asarray(O, dtype=np.float64).reshape(B, H, T, dv)
        # numpy reference block replay.
        S = S0.copy()
        O = np.zeros((B, H, T, dv))
        for t in range(T):
            k = K[:, :, t, :]
            v = V[:, :, t, :]
            a = decay_a[:, :, t][..., None, None]
            b = beta_a[:, :, t][..., None, None]
            if self.erase:
                v_hat = np.einsum("bhd,bhde->bhe", k, S)
                a_s = decay_a[:, :, t][..., None]
                target = v - a_s * v_hat
            else:
                target = v
            S = a * S + b * np.einsum("bhd,bhe->bhde", k, target)
            O[:, :, t, :] = np.einsum("bhd,bhde->bhe", Q[:, :, t, :], S)
        return O

    # ── Flush (state-and-output) ────────────────────────────────────────

    def flush(self) -> "DeltaNetStateHandle":
        """Fold the live buffer into a new checkpoint ``S0`` and clear it."""
        if self._count == 0:
            return self
        self._s0 = self.materialize_state()
        self._k[: self._count] = 0
        self._v[: self._count] = 0
        self._beta[: self._count] = 1.0
        self._decay[: self._count] = 1.0
        self._count = 0
        return self

    # ── Speculative rollback (cursor move) ──────────────────────────────

    def rollback(self, n: int) -> "DeltaNetStateHandle":
        """Drop the last ``n`` appended replay tokens — a cursor rewind."""
        n = min(int(n), self._count)
        if n <= 0:
            return self
        keep = self._count - n
        self._k[keep : self._count] = 0
        self._v[keep : self._count] = 0
        self._beta[keep : self._count] = 1.0
        self._decay[keep : self._count] = 1.0
        self._count = keep
        return self

    def reset(self) -> "DeltaNetStateHandle":
        self._s0[...] = 0
        self._k[...] = 0
        self._v[...] = 0
        self._beta[...] = 1.0
        self._decay[...] = 1.0
        self._count = 0
        return self

    def clone(self) -> "DeltaNetStateHandle":
        return copy.deepcopy(self)

    # ── S12 checkpoint round-trip ───────────────────────────────────────

    def checkpoint(self) -> dict[str, Any]:
        return {
            "batch": int(self.batch),
            "num_heads": int(self.num_heads),
            "key_dim": int(self.key_dim),
            "value_dim": int(self.value_dim),
            "capacity": int(self.capacity),
            "dtype": self.dtype,
            "spec_window": int(self.spec_window),
            "erase": bool(self.erase),
            "checkpoint_state": np.array(self._s0, copy=True),
            "k": np.array(self._k[: self._count], copy=True),
            "v": np.array(self._v[: self._count], copy=True),
            "beta": np.array(self._beta[: self._count], copy=True),
            "decay": np.array(self._decay[: self._count], copy=True),
            "count": int(self._count),
        }

    @classmethod
    def restore(cls, state: Mapping[str, Any]) -> "DeltaNetStateHandle":
        handle = cls(
            batch=int(state["batch"]),
            num_heads=int(state["num_heads"]),
            key_dim=int(state["key_dim"]),
            value_dim=int(state["value_dim"]),
            capacity=int(state["capacity"]),
            dtype=str(state.get("dtype", "fp32")),
            spec_window=int(state.get("spec_window", 0)),
            erase=bool(state.get("erase", True)),
        )
        handle._s0 = np.asarray(state["checkpoint_state"], dtype=np.float64).copy()
        count = int(state.get("count", 0))
        if count:
            handle._k[:count] = np.asarray(state["k"], dtype=np.float64)
            handle._v[:count] = np.asarray(state["v"], dtype=np.float64)
            handle._beta[:count] = np.asarray(state["beta"], dtype=np.float64)
            handle._decay[:count] = np.asarray(state["decay"], dtype=np.float64)
        handle._count = count
        return handle


__all__ = ["DeltaNetStateHandle"]
