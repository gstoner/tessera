"""``tessera.stdlib.hybrid`` — hybrid linear/attention/SSM schedules (Track L L3+L4.1).

Frontier hybrids alternate a constant-state mixer with periodic full-attention
"anchor" layers:
  * Qwen3.6  — Gated DeltaNet (linear) ×3 : Gated Attention ×1
  * Nemotron — Mamba-2 SSM (linear) : sparse attention anchors
  * Mellum2  — sliding-window : full

The defining *systems* contract is a **dual cache**: linear/SSM layers carry a
fixed-size recurrent state (delta Ŝ[d_k,d_v] / SSM h[D,N]); full-attention layers
carry a growing KV cache.  Step-by-step decode equals a full recompute only if
all of them are threaded correctly.

This module makes the schedule first-class (`HybridSchedule`) and the linear-slot
mixer pluggable (`linear_mixer = "delta" | "ssm"`).  The stack is built from
per-mixer **span functions** that handle a token-span of any length carrying
their own cache, so `hybrid_forward` (one span) and `hybrid_decode` (streamed
spans) run identical per-layer code — the oracle (`tests/unit/test_stdlib_hybrid.py`)
is **streaming dual-cache decode ≡ full recompute**.  Delta layers L2-normalize
keys (the L1.1 conditioning finding); the SSM step reproduces
`tessera.ops.selective_ssm` (the L4 op) and returns the carried state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import delta_rule as _dr

LINEAR = "linear"
FULL = "full"
DELTA = "delta"
SSM = "ssm"


# ─────────────────────────────────────────────────────────────────────────────
# The schedule — first-class layer_types
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HybridSchedule:
    """Per-layer mixer assignment.  ``full`` layer iff ``(i + full_offset) %
    period == 0``.  Qwen3.6 = period 4, full_offset 1 → ``[lin,lin,lin,full]·N``."""
    num_layers: int
    period: int = 4
    full_offset: int = 1

    def __post_init__(self):
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.period < 1:
            raise ValueError("period must be >= 1")

    def layer_types(self) -> list[str]:
        return [FULL if ((i + self.full_offset) % self.period == 0) else LINEAR
                for i in range(self.num_layers)]

    def is_full(self, i: int) -> bool:
        return (i + self.full_offset) % self.period == 0

    def full_indices(self) -> list[int]:
        return [i for i in range(self.num_layers) if self.is_full(i)]

    def linear_indices(self) -> list[int]:
        return [i for i in range(self.num_layers) if not self.is_full(i)]

    def counts(self) -> dict[str, int]:
        t = self.layer_types()
        return {LINEAR: t.count(LINEAR), FULL: t.count(FULL)}


def qwen3_6_schedule(num_layers: int = 40) -> HybridSchedule:
    """Qwen3.6-35B-A3B: ``[Gated DeltaNet ×3, Gated Attention] × (N/4)``."""
    return HybridSchedule(num_layers=num_layers, period=4, full_offset=1)


def nemotron_schedule(num_layers: int, attn_period: int = 8) -> HybridSchedule:
    """Nemotron-3-style: predominantly linear (Mamba), a sparse attention anchor
    every ``attn_period`` layers."""
    return HybridSchedule(num_layers=num_layers, period=attn_period, full_offset=1)


# ─────────────────────────────────────────────────────────────────────────────
# Reference hybrid stack (the dual-cache contract)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class HybridConfig:
    d_model: int
    num_heads: int
    head_dim: int
    schedule: HybridSchedule
    ffn_mult: int = 2
    linear_mixer: str = DELTA   # "delta" (Qwen3.6) | "ssm" (Nemotron/Mamba)
    ssm_state: int = 8          # SSM state dim N (scalar-state config)

    @property
    def inner(self) -> int:
        return self.num_heads * self.head_dim

    def mixer_for(self, i: int) -> str:
        """One of ``full`` / ``delta`` / ``ssm`` for layer ``i``."""
        if self.schedule.is_full(i):
            return FULL
        return self.linear_mixer


def synth_weights(cfg: HybridConfig, rng) -> list[dict]:
    """Small random per-layer weights (reference tier), keyed by mixer type."""
    Dm, inner, N = cfg.d_model, cfg.inner, cfg.ssm_state
    ff = cfg.ffn_mult * Dm
    s = 1.0 / np.sqrt(Dm)
    layers = []
    for i in range(cfg.schedule.num_layers):
        w = dict(
            n1=rng.standard_normal(Dm) * 0.1 + 1.0,
            n2=rng.standard_normal(Dm) * 0.1 + 1.0,
            wo=rng.standard_normal((inner, Dm)) / np.sqrt(inner),
            wg=rng.standard_normal((Dm, ff)) * s,
            wu=rng.standard_normal((Dm, ff)) * s,
            wd=rng.standard_normal((ff, Dm)) / np.sqrt(ff),
        )
        mixer = cfg.mixer_for(i)
        if mixer == FULL:
            w.update(wq=rng.standard_normal((Dm, inner)) * s,
                     wk=rng.standard_normal((Dm, inner)) * s,
                     wv=rng.standard_normal((Dm, inner)) * s)
        elif mixer == DELTA:
            w.update(wq=rng.standard_normal((Dm, inner)) * s,
                     wk=rng.standard_normal((Dm, inner)) * s,
                     wv=rng.standard_normal((Dm, inner)) * s,
                     wbeta=rng.standard_normal((Dm, cfg.num_heads)) * s,
                     wdecay=rng.standard_normal((Dm, cfg.num_heads)) * s)
        else:  # SSM
            w.update(w_x=rng.standard_normal((Dm, inner)) * s,
                     w_b=rng.standard_normal((Dm, N)) * s,
                     w_c=rng.standard_normal((Dm, N)) * s,
                     w_dt=rng.standard_normal((Dm, inner)) * s,
                     dt_bias=rng.standard_normal(inner) * 0.1 - 1.0,
                     a_log=rng.standard_normal(inner) * 0.1)   # A = -exp(a_log) < 0
        layers.append(w)
    return layers


def _rmsnorm(x, g, eps=1e-5):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * g


def _swiglu(x, w):
    g = x @ w["wg"]
    u = x @ w["wu"]
    return (g / (1.0 + np.exp(-g)) * u) @ w["wd"]


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _heads(x, W, H, Dh):
    B, S, _ = x.shape
    return np.transpose((x @ W).reshape(B, S, H, Dh), (0, 2, 1, 3))


def _merge(o, Wo):
    B, H, S, Dh = o.shape
    return np.transpose(o, (0, 2, 1, 3)).reshape(B, S, H * Dh) @ Wo


def _norm_last(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def _per_head_scalar(x, W):
    return np.transpose(1.0 / (1.0 + np.exp(-(x @ W))), (0, 2, 1))   # [B,H,S]


def _causal_attention(Q, K, V):
    Dh = Q.shape[-1]
    scores = np.einsum("bhsd,bhtd->bhst", Q, K) / np.sqrt(Dh)
    Sq, Sk = Q.shape[2], K.shape[2]
    mask = np.tril(np.ones((Sq, Sk), bool), k=Sk - Sq)   # aligns q-offset to cached keys
    scores = np.where(mask, scores, -1e30)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("bhst,bhtd->bhsd", p, V)


def _ssm_scan(x, A, B, C, dt, h0=None):
    """Mamba-2 SSD recurrence, returns (y, h_final).  Exactly reproduces
    ``tessera.ops.selective_ssm`` (scalar-state A), plus the carried state.

    x, dt : [B, S, D]   A : [D]   B, C : [B, S, N]   h : [B, D, N]
    """
    Bsz, Sl, D = x.shape
    N = B.shape[-1]
    A2d = A[None, :, None]                                 # [1, D, 1] (scalar-state)
    h = np.zeros((Bsz, D, N)) if h0 is None else np.array(h0, copy=True)
    y = np.zeros((Bsz, Sl, D))
    for t in range(Sl):
        A_bar = np.exp(dt[:, t, :, None] * A2d)            # [B, D, 1] → bcast N
        B_bar = dt[:, t, :, None] * B[:, t, None, :]       # [B, D, N]
        h = A_bar * h + B_bar * x[:, t, :, None]
        y[:, t, :] = np.einsum("bdn,bn->bd", h, C[:, t, :])
    return y, h


# ── per-mixer span functions (handle a token-span of any length + cache) ─────
def _delta_span(x, w, cfg, cache):
    H, Dh = cfg.num_heads, cfg.head_dim
    h = _rmsnorm(x, w["n1"])
    Q = _heads(h, w["wq"], H, Dh)
    K = _norm_last(_heads(h, w["wk"], H, Dh))   # L2-normalized keys (L1.1 finding)
    V = _heads(h, w["wv"], H, Dh)
    beta = _per_head_scalar(h, w["wbeta"])
    decay = _per_head_scalar(h, w["wdecay"])
    o, st = _dr.gated_delta_rule_recurrent(
        Q, K, V, beta=beta, decay=decay, state=cache.get("S"),
        return_state=True, state_dtype="fp64")
    cache["S"] = st
    x = x + _merge(np.asarray(o), w["wo"])
    return x + _swiglu(_rmsnorm(x, w["n2"]), w)


def _ssm_span(x, w, cfg, cache):
    H, Dh = cfg.num_heads, cfg.head_dim
    h = _rmsnorm(x, w["n1"])
    x_ssm = h @ w["w_x"]
    Bp = h @ w["w_b"]
    Cp = h @ w["w_c"]
    dt = _softplus(h @ w["w_dt"] + w["dt_bias"])
    A = -np.exp(w["a_log"])
    y, hnew = _ssm_scan(x_ssm, A, Bp, Cp, dt, h0=cache.get("H"))
    cache["H"] = hnew
    x = x + y @ w["wo"]                          # y [B,S,inner] -> [B,S,Dm]
    return x + _swiglu(_rmsnorm(x, w["n2"]), w)


def _full_span(x, w, cfg, cache):
    H, Dh = cfg.num_heads, cfg.head_dim
    h = _rmsnorm(x, w["n1"])
    q = _heads(h, w["wq"], H, Dh)
    k = _heads(h, w["wk"], H, Dh)
    v = _heads(h, w["wv"], H, Dh)
    cache["K"] = k if cache.get("K") is None else np.concatenate([cache["K"], k], axis=2)
    cache["V"] = v if cache.get("V") is None else np.concatenate([cache["V"], v], axis=2)
    o = _causal_attention(q, cache["K"], cache["V"])
    x = x + _merge(o, w["wo"])
    return x + _swiglu(_rmsnorm(x, w["n2"]), w)


_SPAN = {DELTA: _delta_span, SSM: _ssm_span, FULL: _full_span}


def _run_layers(x_span, weights, cfg, caches):
    for i, w in enumerate(weights):
        x_span = _SPAN[cfg.mixer_for(i)](x_span, w, cfg, caches[i])
    return x_span


def hybrid_forward(x, weights, cfg: HybridConfig):
    """Full (parallel) forward over the whole sequence; returns hidden [B,S,Dm]."""
    return _run_layers(x, weights, cfg, [{} for _ in weights])


def hybrid_decode(x, weights, cfg: HybridConfig, prefill: int = 1):
    """Stream the sequence token-by-token (after a `prefill` chunk) carrying the
    dual cache — recurrent Ŝ (delta) / h (SSM) for linear layers, KV for full
    layers.  Equals `hybrid_forward` for the same input (the oracle)."""
    B, S, Dm = x.shape
    caches: list[dict] = [{} for _ in weights]
    out = np.zeros_like(x)
    out[:, :prefill] = _run_layers(x[:, :prefill], weights, cfg, caches)
    for t in range(prefill, S):
        out[:, t:t + 1] = _run_layers(x[:, t:t + 1], weights, cfg, caches)
    return out
