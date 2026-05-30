"""Resident decode loop — GPU-resident activations across decode steps (R3).

The payoff of the GPU-resident architecture (R0–R2): a per-token decode step runs
entirely on-device in **one command buffer**, the model weights stay **resident
across steps** (uploaded once, never re-sent), and **only the sampled token id
reads back** to the host each step — logits and activations never round-trip.

`ResidentMLADecoder` drives a single-sequence (B=1) attention decode step:

    qn      = rmsnorm(x, gamma)            # per-head query norm
    scores  = qn @ Kᵀ                      # [H, Skv]
    attn    = softmax(scores)
    ctx     = attn @ V                      # [H, D]  -> flatten [H*D]
    logits  = ctx_flat @ W_logit           # [vocab]
    token   = gumbel-sample(logits)        # the only value read back

Built on `runtime.AppleGPUEncodeSession` (one command buffer / step) +
`runtime.DeviceTensor` (resident weights + activations). Falls back to numpy when
the Metal encode session is unavailable, so the loop is portable.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .. import runtime as R
from .. import rng as _rng


def _softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def _rmsnorm(x, gamma, eps):
    d = x.astype(np.float64)
    n = d / np.sqrt((d * d).mean(-1, keepdims=True) + eps)
    return n * gamma.astype(np.float64)


class ResidentMLADecoder:
    """Single-sequence resident attention decoder.

    Parameters
    ----------
    num_heads, head_dim, vocab
        Geometry. The query is ``[num_heads, head_dim]`` per step; the logit
        projection maps the flattened context ``[num_heads*head_dim]`` to
        ``vocab``.
    rmsnorm_gamma
        ``[head_dim]`` RMSNorm weight (applied per head).
    w_logit
        ``[num_heads*head_dim, vocab]`` output projection.
    eps
        RMSNorm epsilon.
    """

    def __init__(self, *, num_heads: int, head_dim: int, vocab: int,
                 rmsnorm_gamma: Any, w_logit: Any, eps: float = 1e-6) -> None:
        self.H = int(num_heads)
        self.D = int(head_dim)
        self.V = int(vocab)
        self.eps = float(eps)
        self._gamma_np = np.ascontiguousarray(rmsnorm_gamma, np.float32)
        self._wlogit_np = np.ascontiguousarray(w_logit, np.float32)
        if self._gamma_np.shape != (self.D,):
            raise ValueError(f"rmsnorm_gamma must be [head_dim]={(self.D,)}")
        if self._wlogit_np.shape != (self.H * self.D, self.V):
            raise ValueError(
                f"w_logit must be [num_heads*head_dim, vocab]="
                f"{(self.H * self.D, self.V)}; got {self._wlogit_np.shape}")

        # Resident weights — uploaded ONCE, reused every step (the cross-step win).
        self._dt = R.DeviceTensor
        self._gamma = self._dt.from_numpy(self._gamma_np)
        # logit proj as a batch-1 bmm operand [1, H*D, V] (broadcast over batch)
        self._wlogit = self._dt.from_numpy(self._wlogit_np[None])
        self._resident = self._gamma is not None and self._wlogit is not None
        self._step_count = 0
        self._weight_uploads = 1 if self._resident else 0

    @property
    def resident(self) -> bool:
        """True when running the on-device resident path (vs numpy fallback)."""
        return self._resident

    @property
    def weight_uploads(self) -> int:
        """Number of weight uploads — should stay 1 across the whole loop."""
        return self._weight_uploads

    # ------------------------------------------------------------------
    def step(self, x: Any, k_t: Any, v: Any, *, key: Optional[Any] = None,
             temperature: float = 1.0, greedy: bool = False) -> int:
        """Decode one token. ``x`` is the query ``[H, D]``; ``k_t`` is ``[H, D,
        Skv]`` (per-head Kᵀ over the current cache window); ``v`` is
        ``[H, Skv, D]``. Returns the sampled token id. Only that id crosses back
        to the host — logits and activations stay resident."""
        x = np.ascontiguousarray(x, np.float32)
        k_t = np.ascontiguousarray(k_t, np.float32)
        v = np.ascontiguousarray(v, np.float32)
        H, D = self.H, self.D
        Skv = int(k_t.shape[-1])
        if x.shape != (H, D) or k_t.shape != (H, D, Skv) or v.shape != (H, Skv, D):
            raise ValueError("shape mismatch in decode step inputs")

        # Gumbel noise from the canonical Philox stream (deterministic, #18-safe).
        if greedy or temperature == 0.0:
            noise = np.zeros((1, self.V), np.float32)
            inv_temp = 1.0
        else:
            k = key if key is not None else _rng.RNGKey.from_seed(self._step_count)
            noise = R._gumbel_noise_from_key((1, self.V), k, np)
            inv_temp = 1.0 / float(temperature)

        tok = (self._step_resident(x, k_t, v, noise, inv_temp) if self._resident
               else None)
        if tok is None:
            tok = self._step_numpy(x, k_t, v, noise, inv_temp)
        self._step_count += 1
        return int(tok)

    def _step_resident(self, x, k_t, v, noise, inv_temp) -> "int | None":
        dt = self._dt
        H, D = self.H, self.D
        Skv = int(k_t.shape[-1])
        dx = dt.from_numpy(x)
        dKt = dt.from_numpy(k_t)
        dV = dt.from_numpy(v)
        dnoise = dt.from_numpy(noise)
        ids = None
        sess = R.AppleGPUEncodeSession()
        if not sess.available or dx is None or dKt is None or dV is None \
                or dnoise is None or self._gamma is None or self._wlogit is None:
            sess.commit()
            for t in (dx, dKt, dV, dnoise):
                if t is not None:
                    t.free()
            return None
        with sess:
            qn = sess.rmsnorm(dx, self._gamma, self.eps)              # [H, D]
            scores = sess.bmm(qn.reshape_view(H, 1, D), dKt) if qn is not None else None
            attn = sess.softmax(scores.reshape_view(H, Skv)) if scores is not None else None
            ctx = sess.bmm(attn.reshape_view(H, 1, Skv), dV) if attn is not None else None
            ctx_flat = ctx.reshape_view(1, 1, H * D) if ctx is not None else None
            logits = sess.bmm(ctx_flat, self._wlogit) if ctx_flat is not None else None
            ids = (sess.gumbel(logits.reshape_view(1, self.V), dnoise, inv_temp)
                   if logits is not None else None)
        tok = int(ids.numpy()[0]) if ids is not None else None
        for t in (dx, dKt, dV, dnoise):
            if t is not None:
                t.free()
        return tok

    def _step_numpy(self, x, k_t, v, noise, inv_temp) -> int:
        H, D = self.H, self.D
        qn = _rmsnorm(x, self._gamma_np, self.eps)               # [H, D]
        scores = np.einsum("hd,hdk->hk", qn, k_t.astype(np.float64))
        attn = _softmax(scores)
        ctx = np.einsum("hk,hkd->hd", attn, v.astype(np.float64))  # [H, D]
        logits = ctx.reshape(1, H * D) @ self._wlogit_np.astype(np.float64)  # [1,V]
        return int(np.argmax(logits * inv_temp + noise.astype(np.float64), axis=-1)[0])

    def free(self) -> None:
        for t in (self._gamma, self._wlogit):
            if t is not None:
                t.free()
        self._gamma = self._wlogit = None
        self._resident = False

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (f"ResidentMLADecoder(H={self.H}, D={self.D}, vocab={self.V}, "
                f"resident={self._resident}, steps={self._step_count})")
