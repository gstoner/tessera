"""Track-R (ReplaySSM) Phase 1 — SSM decode-state handle with replay inputs.

``SSMStateHandle`` is the production decode-state ABI for selective state-space
models (Mamba-2 / SSD).  It is the SSM analogue of ``KVCacheHandle`` /
``MemoryStateHandle``: opaque to ops, functional read, in-place-with-COW
mutation, ``clone()`` / ``checkpoint()`` / ``restore()`` round-trip.

The idea (ReplaySSM, Dao AI Lab 2026 — see
``docs/audit/roadmap/REPLAYSSM_PLAN.md``): instead of writing the full
recurrent state ``S`` of shape ``(B, D, N)`` to memory every decode token (the
*summary route*), keep a **checkpoint state** ``S0`` plus a small **ring buffer**
of recent per-token inputs ``(delta, x, b)`` and reconstruct outputs on demand
from the buffer (the *history route*).  Two routes fall out of the identity:

* **output-only** (most steps): compute ``y_t`` directly from ``S0`` + buffer,
  never materializing the new state — no state write;
* **state-and-output** (*flush*): materialize the state from ``S0`` + buffer,
  fold it into a new ``S0``, and clear the buffer — done only when the buffer
  is near-full (``count + 2*spec_window + n_new > capacity``).

Speculative rollback is then a **cursor move**: rejected draft tokens are
removed by rewinding the buffer count (``rollback(n)``), with no per-position
state snapshot.

This is the *reference* handle — correctness + the ABI shape.  It accumulates
in float64 (the reference tier, matching ``_mamba_ssd`` and
``stdlib.delta_rule``); the production contract is fp32 accumulation regardless
of storage dtype.  The fused MSL decode kernel that realizes the state-traffic
halving is the (hardware-gated) Phase 5 follow-up.

The reconstruction is exact w.r.t. ``tessera.ops.selective_ssm``.  For
checkpoint state ``S0`` and buffer tokens ``i = 0..m`` (append order), with
per-token input outer product ``u_i[d,n] = (delta_i[d] * x_i[d]) * b_i[n]`` and
log-decay ``da_i[d,n] = delta_i[d] * a[d,n]``::

    Dcum_m = sum_{k<=m} da_k                         # inclusive cumulative decay
    h_m    = exp(Dcum_m) * S0 + sum_{i<=m} exp(Dcum_m - Dcum_i) * u_i
    y_m[d] = sum_n c_m[n] * h_m[d,n]                 # optional gate: y_m *= g_m

``exp(Dcum_m - Dcum_i)`` is a *bounded* pairwise decay (the cumulative sum of
``delta*a`` is <= 0, so the exponent is <= 0 → value in ``(0, 1]``), so the
reconstruction is numerically stable — no ``exp(+large)`` overflow.
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
class SSMStateHandle:
    """Opaque decode-state handle for a selective SSM (Mamba-2 / SSD).

    Parameters
    ----------
    batch : int
        Batch size ``B``.
    num_channels : int
        Channel dim ``D`` (``x``/``delta`` trailing dim).
    state_dim : int
        State dim ``N`` (``b``/``c`` trailing dim).
    a : np.ndarray
        State-matrix diagonal — rank-1 ``(D,)`` (scalar-state, the common
        per-head Mamba-2 config) or rank-2 ``(D, N)``.  Typically negative.
    capacity : int
        Ring-buffer length ``L`` — max live replay tokens between flushes.
    dtype : str
        Canonical storage dtype for the declared ABI (default ``"fp32"``).
        Internal accumulation is float64 (reference tier).
    spec_window : int
        Speculative window ``T`` reserved by the flush rule
        (``count + 2*T + n_new > L`` triggers a flush).  Default 0 (pure AR).

    Attributes (read-only)
    ----------------------
    checkpoint_state : np.ndarray   shape (B, D, N) — folded state ``S0``
    count : int                     live replay tokens since last flush
    capacity : int                  ring length ``L``
    """

    batch: int
    num_channels: int
    state_dim: int
    a: np.ndarray
    capacity: int = 64
    dtype: str = "fp32"
    spec_window: int = 0
    #: Optional batched-matmul backend ``(…,M,K) @ (…,K,N) -> (…,M,N)`` for the
    #: scalar-A reconstruction's contractions (projection / gram / state
    #: update).  Default numpy; the Apple GPU bmm lane routes them to Metal
    #: (Track-R Phase 4).  General ``(D,N)`` A ignores this (not bmm-separable).
    matmul3d: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = field(
        default=None, repr=False
    )
    #: Provenance tag: "reference" (host numpy), "apple_gpu" (contractions on
    #: the Metal bmm lane), or "apple_gpu_fused" (single-dispatch decode
    #: kernel).  Set by the runtime factory, not load-bearing.
    backend: str = "reference"
    #: Optional fused output-only decode kernel (Track-R Phase 5).  Signature
    #: ``(delta(m,B,D), x(m,B,D), b(m,B,N), s0(B,D,N), c(B,N), a(D,), B,D,N,m)
    #: -> y(B,D)`` (or ``None`` to fall back).  Used by ``read_output`` for the
    #: scalar-A path when wired; the Metal kernel keeps ``s0`` resident.
    decode_fn: Optional[Callable[..., Any]] = field(default=None, repr=False)
    #: Optional block decode kernel (Track-R dispatch-overhead fix).  Signature
    #: ``(delta(T,B,D), x(T,B,D), b(T,B,N), c(T,B,N), s0(B,D,N), a(D,), B,T,D,N)
    #: -> out(T,B,D)``.  Computes a whole block in one dispatch; used by
    #: :meth:`decode_block`.
    block_fn: Optional[Callable[..., Any]] = field(default=None, repr=False)

    _a2d: np.ndarray = field(init=False, repr=False)
    _s0: np.ndarray = field(init=False, repr=False)
    _delta: np.ndarray = field(init=False, repr=False)
    _x: np.ndarray = field(init=False, repr=False)
    _b: np.ndarray = field(init=False, repr=False)
    _count: int = field(init=False, default=0)
    _scalar_a: bool = field(init=False, default=True)
    _a1d: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    #: Provenance of the most recent output-only replay read.  Numerical host
    #: fallback is never eligible for native Apple proof.
    last_decode_execution: str = field(init=False, default="reference_cpu")
    #: Provenance of the most recent block decode. A status-returning backend
    #: must confirm its dispatch; fallback-capable numerical output is not
    #: sufficient to earn this marker.
    last_block_execution: str = field(init=False, default="reference_cpu")

    def __post_init__(self) -> None:
        from ..dtype import canonicalize_dtype

        object.__setattr__(
            self, "dtype", canonicalize_dtype(self.dtype, allow_planned_gated=True)
        )
        B, D, N, L = self.batch, self.num_channels, self.state_dim, self.capacity
        if min(B, D, N) <= 0:
            raise ValueError(
                f"batch/num_channels/state_dim must be positive, "
                f"got ({B}, {D}, {N})"
            )
        if L <= 0:
            raise ValueError(f"capacity must be positive, got {L}")
        if self.spec_window < 0:
            raise ValueError(f"spec_window must be >= 0, got {self.spec_window}")

        a = np.asarray(self.a, dtype=np.float64)
        if a.ndim == 1:
            if a.shape[0] != D:
                raise ValueError(f"a 1-d shape {a.shape} != (D={D},)")
            a2d = np.broadcast_to(a[:, None], (D, N)).copy()
        elif a.ndim == 2:
            if a.shape != (D, N):
                raise ValueError(f"a 2-d shape {a.shape} != (D={D}, N={N})")
            a2d = a.copy()
        else:
            raise ValueError(f"a must be (D,) or (D, N); got {a.shape}")
        self._a2d = a2d                                    # (D, N), float64
        # Scalar-state A (the common per-head Mamba-2 config) makes the decay
        # independent of the state dim N, so the reconstruction factorizes into
        # batched matmuls (projection / gram / state update) — the bmm lane.
        self._scalar_a = a.ndim == 1
        self._a1d = a.copy() if a.ndim == 1 else None      # (D,), float64

        # Checkpoint state S0 (B, D, N) and ring buffers of small per-token
        # inputs — this is the whole point: 2D + N scalars per token, not D*N.
        self._s0 = np.zeros((B, D, N), dtype=np.float64)
        self._delta = np.zeros((L, B, D), dtype=np.float64)
        self._x = np.zeros((L, B, D), dtype=np.float64)
        self._b = np.zeros((L, B, N), dtype=np.float64)
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
        """The folded checkpoint state ``S0`` (B, D, N) — not the live state."""
        return self._s0.copy()

    # ── Flush policy / route selection ──────────────────────────────────

    def should_flush(self, n_new: int = 1) -> bool:
        """ReplaySSM flush rule: ``count + 2*spec_window + n_new > capacity``.

        Reserves room for a speculative window of draft tokens so a spec burst
        never truncates mid-stream.  Delegates to
        ``tessera.compiler.ssm_replay.should_flush`` — the single source of
        truth shared with the compiler-side route selection.
        """
        from ..compiler.ssm_replay import should_flush as _should_flush

        return _should_flush(self._count, self.capacity, self.spec_window, n_new)

    def route_for(self, n_new: int = 1) -> str:
        """``"state_and_output"`` if appending ``n_new`` forces a flush, else
        ``"output_only"``.  The ``--replayssm-route`` decision as a
        compiler-visible policy; delegates to
        ``tessera.compiler.ssm_replay.select_route``."""
        from ..compiler.ssm_replay import select_route

        return select_route(self._count, self.capacity, self.spec_window, n_new)

    # ── Cumulative decay (the shared kernel of every route) ─────────────

    def _mm(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Batched matmul backend — the Apple GPU bmm lane when wired, else
        numpy.  Result cast back to float64 (reference accumulation)."""
        fn = self.matmul3d if self.matmul3d is not None else np.matmul
        return np.asarray(fn(p, q), dtype=np.float64)

    def _try_fused_decode(self, c: np.ndarray) -> Optional[np.ndarray]:
        """Run the fused single-dispatch decode kernel (Phase 5) over the live
        buffer; return ``y`` (B, D) or ``None`` when no kernel is wired / it
        declines.  Scalar-A only; callers guard ``count > 0``."""
        if self.decode_fn is None:
            return None
        assert self._a1d is not None
        m = self._count
        res = self.decode_fn(
            self._delta[:m], self._x[:m], self._b[:m], self._s0, c,
            self._a1d, self.batch, self.num_channels, self.state_dim, m,
        )
        if res is None:
            return None
        if isinstance(res, tuple):
            res, native = res
            self.last_decode_execution = "native_gpu" if native else "reference_cpu"
        else:
            # Preserve legacy/custom callback compatibility without promoting it.
            self.last_decode_execution = "reference_cpu"
        return np.asarray(res, dtype=np.float64).reshape(
            self.batch, self.num_channels
        )

    def _dcum(self) -> np.ndarray:
        """Inclusive cumulative log-decay ``Dcum`` over live tokens.

        Returns ``(count, B, D, N)`` float64; ``Dcum[m] = sum_{k<=m} delta_k*a``.
        Empty buffer → shape ``(0, B, D, N)``.
        """
        m = self._count
        if m == 0:
            return np.zeros((0, self.batch, self.num_channels, self.state_dim))
        # da[k] = delta_k[:, :, None] * a2d  → (m, B, D, N)
        da = self._delta[:m, :, :, None] * self._a2d[None, None, :, :]
        return np.cumsum(da, axis=0)

    def _dcum_bd(self) -> np.ndarray:
        """Scalar-A cumulative log-decay ``(count, B, D)`` — decay is
        independent of the state dim N, so it collapses an axis vs ``_dcum``."""
        m = self._count
        if m == 0:
            return np.zeros((0, self.batch, self.num_channels))
        assert self._a1d is not None                       # scalar-A invariant
        da = self._delta[:m] * self._a1d[None, None, :]    # (m, B, D)
        return np.cumsum(da, axis=0)

    def materialize_state(self) -> np.ndarray:
        """Compute the *live* state ``h`` (B, D, N) = checkpoint + replayed
        buffer, without mutating the handle (the state-and-output path's state).

        Scalar-A routes the state update through the batched-matmul backend
        (``matmul3d`` — the Metal bmm lane when wired); general ``(D,N)`` A uses
        the dense einsum reference path.
        """
        m = self._count
        if m == 0:
            return self._s0.copy()
        if self._scalar_a:
            dcum = self._dcum_bd()                         # (m, B, D)
            dcum_t = dcum[-1]                              # (B, D)
            g = self._delta[:m] * self._x[:m]             # (m, B, D)
            decay = np.exp(dcum_t[None] - dcum)            # (m, B, D)
            w = np.moveaxis(decay * g, 0, -1)              # (B, D, m)
            bbuf = np.moveaxis(self._b[:m], 0, 1)          # (B, m, N)
            upd = self._mm(w, bbuf)                        # (B, D, N)  [bmm]
            return np.exp(dcum_t)[:, :, None] * self._s0 + upd
        dcum = self._dcum()                                # (m, B, D, N)
        dcum_t = dcum[-1]                                  # (B, D, N)
        h = np.exp(dcum_t) * self._s0
        g = (self._delta[:m] * self._x[:m])                # (m, B, D)
        u = g[:, :, :, None] * self._b[:m, :, None, :]     # (m, B, D, N)
        decay = np.exp(dcum_t[None] - dcum)                # (m, B, D, N) in (0,1]
        h = h + np.einsum("mbdn,mbdn->bdn", decay, u)
        return h

    # ── Append + read (the decode step) ─────────────────────────────────

    def append(
        self,
        delta_t: np.ndarray,
        x_t: np.ndarray,
        b_t: np.ndarray,
        *,
        auto_flush: bool = True,
    ) -> "SSMStateHandle":
        """Append one token's replay inputs to the ring buffer.

        ``delta_t``/``x_t`` are ``(B, D)``; ``b_t`` is ``(B, N)``.  When the
        buffer would overflow and ``auto_flush`` is set, :meth:`flush` runs
        first; otherwise an overflow raises.  Mutates in place (COW via
        :meth:`clone`).
        """
        d = _to_array(delta_t).astype(np.float64).reshape(self.batch, self.num_channels)
        x = _to_array(x_t).astype(np.float64).reshape(self.batch, self.num_channels)
        b = _to_array(b_t).astype(np.float64).reshape(self.batch, self.state_dim)
        if self._count >= self.capacity:
            if auto_flush:
                self.flush()
            else:
                raise ValueError(
                    f"SSMStateHandle ring buffer full (capacity={self.capacity}); "
                    f"flush() or pass auto_flush=True"
                )
        i = self._count
        self._delta[i] = d
        self._x[i] = x
        self._b[i] = b
        self._count = i + 1
        return self

    def read_output(
        self, c_t: np.ndarray, *, gate_t: np.ndarray | None = None
    ) -> np.ndarray:
        """Output-only read: ``y_t`` (B, D) for the most-recently-appended token.

        Reconstructs ``y_t = sum_n c_t[n] * h_t[d,n]`` from ``S0`` + buffer
        without materializing ``h_t`` — the ReplaySSM output-only path.
        ``c_t`` is ``(B, N)``; optional ``gate_t`` ``(B, D)`` is applied as
        ``y_t *= gate_t`` (matching ``selective_ssm``'s output gate).
        """
        c = _to_array(c_t).astype(np.float64).reshape(self.batch, self.state_dim)
        self.last_decode_execution = "reference_cpu"
        m = self._count
        if m == 0:
            # No replay tokens: y = c · S0  → bmm (B,1,N)@(B,N,D).
            y = self._mm(c[:, None, :], np.swapaxes(self._s0, -1, -2))[:, 0, :]
        elif self._scalar_a:
            # Fused single-dispatch decode kernel (Track-R Phase 5), if wired —
            # keeps S0 resident, reads only the small replay inputs.
            fused = self._try_fused_decode(c)
            if fused is not None:
                y = fused
            else:
                # Output-only via the bmm lane: projection + gram never
                # materialize h_t (the ReplaySSM inner-product-first trick).
                dcum = self._dcum_bd()                     # (m, B, D)
                dcum_t = dcum[-1]                          # (B, D)
                # checkpoint: sproj = c · S0ᵀ ; y_check = exp(Dcum_t) * sproj
                sproj = self._mm(c[:, None, :], np.swapaxes(self._s0, -1, -2))[:, 0, :]
                yk = np.exp(dcum_t) * sproj                # (B, D)
                # buffer: gram[b,i] = c · b_i  → bmm (B,1,N)@(B,N,m)
                bbuf = np.moveaxis(self._b[:m], 0, 1)      # (B, m, N)
                gram = self._mm(c[:, None, :], np.swapaxes(bbuf, -1, -2))[:, 0, :]
                g = self._delta[:m] * self._x[:m]          # (m, B, D)
                decay = np.exp(dcum_t[None] - dcum)        # (m, B, D)
                y = yk + np.einsum("mbd,bm->bd", decay * g, gram)
        else:
            dcum = self._dcum()                            # (m, B, D, N)
            dcum_t = dcum[-1]                              # (B, D, N)
            # checkpoint contribution: sum_n c[n] exp(Dcum_t)[d,n] S0[d,n]
            y = np.einsum("bdn,bdn,bn->bd", np.exp(dcum_t), self._s0, c)
            # buffer contribution: never materializes h_t
            g = self._delta[:m] * self._x[:m]              # (m, B, D)
            decay = np.exp(dcum_t[None] - dcum)            # (m, B, D, N)
            # weight_i[b,d] = sum_n decay_i[b,d,n] * b_i[b,n] * c[b,n]
            w = np.einsum("mbdn,mbn,bn->mbd", decay, self._b[:m], c)
            y = y + np.einsum("mbd,mbd->bd", w, g)
        if gate_t is not None:
            y = y * _to_array(gate_t).astype(np.float64).reshape(
                self.batch, self.num_channels
            )
        return y

    def step(
        self,
        delta_t: np.ndarray,
        x_t: np.ndarray,
        b_t: np.ndarray,
        c_t: np.ndarray,
        *,
        gate_t: np.ndarray | None = None,
    ) -> np.ndarray:
        """One decode token: append ``(delta, x, b)`` then output-only read.

        Flushes first when the flush rule fires (route ``state_and_output``),
        so the buffer never overflows mid-step.  Returns ``y_t`` (B, D).
        """
        if self.should_flush(1):
            self.flush()
        self.append(delta_t, x_t, b_t, auto_flush=False)
        return self.read_output(c_t, gate_t=gate_t)

    # ── Block decode (dispatch-overhead fix) ────────────────────────────

    def decode_block(
        self,
        deltas: np.ndarray,
        xs: np.ndarray,
        bs: np.ndarray,
        cs: np.ndarray,
    ) -> np.ndarray:
        """Compute outputs for a block of ``T`` tokens in one shot, from the
        current live state — the prefill / speculative-verification /
        benchmark path that avoids per-token command-buffer dispatch.

        ``deltas``/``xs`` are ``(T, B, D)``; ``bs``/``cs`` are ``(T, B, N)``.
        Returns ``(T, B, D)``.  Pure — does not mutate the handle (commit
        accepted tokens via :meth:`step`/:meth:`append`).  Scalar-A only.
        """
        self.last_block_execution = "reference_cpu"
        if not self._scalar_a:
            raise ValueError("decode_block requires scalar-state A")
        deltas = _to_array(deltas).astype(np.float64)
        xs = _to_array(xs).astype(np.float64)
        bs = _to_array(bs).astype(np.float64)
        cs = _to_array(cs).astype(np.float64)
        T, B, D, N = deltas.shape[0], self.batch, self.num_channels, self.state_dim
        s0 = self.materialize_state()                      # current live state
        if self.block_fn is not None:
            assert self._a1d is not None
            res = self.block_fn(deltas, xs, bs, cs, s0, self._a1d, B, T, D, N)
            if res is not None:
                value = res
                if isinstance(res, tuple) and len(res) == 2:
                    value, provenance = res
                    self.last_block_execution = (
                        "native_gpu" if provenance is True else
                        "reference_cpu" if provenance is False else str(provenance))
                return np.asarray(value, dtype=np.float64).reshape(T, B, D)
        # numpy reference block (one sequential pass from S0).
        h = s0.copy()
        out = np.zeros((T, B, D))
        for t in range(T):
            abar = np.exp(deltas[t][:, :, None] * self._a2d[None])
            h = abar * h + (deltas[t] * xs[t])[:, :, None] * bs[t][:, None, :]
            out[t] = np.einsum("bdn,bn->bd", h, cs[t])
        return out

    # ── Flush (state-and-output) ────────────────────────────────────────

    def flush(self) -> "SSMStateHandle":
        """Fold the live buffer into a new checkpoint ``S0`` and clear it.

        Algebraically exact: ``S0 <- materialize_state()``; ``count <- 0``.
        The expensive full-state write happens here only — amortized across
        the buffer window.
        """
        if self._count == 0:
            return self
        self._s0 = self.materialize_state()
        self._delta[: self._count] = 0
        self._x[: self._count] = 0
        self._b[: self._count] = 0
        self._count = 0
        return self

    # ── Speculative rollback (cursor move) ──────────────────────────────

    def rollback(self, n: int) -> "SSMStateHandle":
        """Drop the last ``n`` appended replay tokens — a cursor rewind.

        This is how a rejected speculative draft is undone: no per-position
        state snapshot, just ``count -= n``.  ``n`` is clamped to ``count``.
        """
        n = min(int(n), self._count)
        if n <= 0:
            return self
        keep = self._count - n
        self._delta[keep : self._count] = 0
        self._x[keep : self._count] = 0
        self._b[keep : self._count] = 0
        self._count = keep
        return self

    def reset(self) -> "SSMStateHandle":
        """Reset to a zero checkpoint and empty buffer."""
        self._s0[...] = 0
        self._delta[...] = 0
        self._x[...] = 0
        self._b[...] = 0
        self._count = 0
        return self

    def clone(self) -> "SSMStateHandle":
        """Deep copy — for the functional ``tape()`` / speculative-fork path."""
        return copy.deepcopy(self)

    # ── S12 checkpoint round-trip ───────────────────────────────────────

    def checkpoint(self) -> dict[str, Any]:
        """Serialize to a plain dict for ``tessera.checkpoint.save_state``.

        Maps onto ``STATE_COLLECTION_SPECS["recurrent_state"]`` (mutable,
        non-persistent across runs but checkpointable within one).
        """
        return {
            "batch": int(self.batch),
            "num_channels": int(self.num_channels),
            "state_dim": int(self.state_dim),
            "capacity": int(self.capacity),
            "dtype": self.dtype,
            "spec_window": int(self.spec_window),
            "a": np.array(self._a2d, copy=True),
            "checkpoint_state": np.array(self._s0, copy=True),
            "delta": np.array(self._delta[: self._count], copy=True),
            "x": np.array(self._x[: self._count], copy=True),
            "b": np.array(self._b[: self._count], copy=True),
            "count": int(self._count),
        }

    @classmethod
    def restore(cls, state: Mapping[str, Any]) -> "SSMStateHandle":
        """Inverse of :meth:`checkpoint`."""
        handle = cls(
            batch=int(state["batch"]),
            num_channels=int(state["num_channels"]),
            state_dim=int(state["state_dim"]),
            a=np.asarray(state["a"]),
            capacity=int(state["capacity"]),
            dtype=str(state.get("dtype", "fp32")),
            spec_window=int(state.get("spec_window", 0)),
        )
        handle._s0 = np.asarray(state["checkpoint_state"], dtype=np.float64).copy()
        count = int(state.get("count", 0))
        if count:
            handle._delta[:count] = np.asarray(state["delta"], dtype=np.float64)
            handle._x[:count] = np.asarray(state["x"], dtype=np.float64)
            handle._b[:count] = np.asarray(state["b"], dtype=np.float64)
        handle._count = count
        return handle


__all__ = ["SSMStateHandle"]
