"""``tessera.stdlib.hybrid`` — hybrid linear/attention layer schedules (Track L L3).

Frontier hybrids (Qwen3.6, Nemotron-3, Mellum2) alternate a constant-state linear
mixer (gated DeltaNet / Mamba / sliding-window) with periodic full-attention
"anchor" layers.  The defining *systems* contract is a **dual cache**: linear
layers carry a fixed-size recurrent state Ŝ[d_k, d_v]; full-attention layers carry
a growing KV cache.  Step-by-step decode must equal a full recompute only if both
are threaded correctly.

This module makes the schedule a first-class object (`HybridSchedule`, lowered
from a literal `layer_types`) and ships a reference stack whose oracle is exactly
that: **streaming dual-cache decode ≡ full forward** (`tests/unit/test_stdlib_hybrid.py`).
Linear layers use the genuine gated delta rule (`stdlib.delta_rule`) with
L2-normalized keys (the L1.1 conditioning finding); full layers use causal MHA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import delta_rule as _dr

LINEAR = "linear"
FULL = "full"


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

    @property
    def inner(self) -> int:
        return self.num_heads * self.head_dim


def synth_weights(cfg: HybridConfig, rng) -> list[dict]:
    """Small random per-layer weights (reference tier)."""
    Dm, inner = cfg.d_model, cfg.inner
    ff = cfg.ffn_mult * Dm
    s = 1.0 / np.sqrt(Dm)
    layers = []
    for _ in range(cfg.schedule.num_layers):
        layers.append(dict(
            n1=rng.standard_normal(Dm) * 0.1 + 1.0,
            wq=rng.standard_normal((Dm, inner)) * s,
            wk=rng.standard_normal((Dm, inner)) * s,
            wv=rng.standard_normal((Dm, inner)) * s,
            wo=rng.standard_normal((inner, Dm)) / np.sqrt(inner),
            wbeta=rng.standard_normal((Dm, cfg.num_heads)) * s,
            wdecay=rng.standard_normal((Dm, cfg.num_heads)) * s,
            n2=rng.standard_normal(Dm) * 0.1 + 1.0,
            wg=rng.standard_normal((Dm, ff)) * s,
            wu=rng.standard_normal((Dm, ff)) * s,
            wd=rng.standard_normal((ff, Dm)) / np.sqrt(ff),
        ))
    return layers


def _rmsnorm(x, g, eps=1e-5):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * g


def _swiglu(x, w):
    g = x @ w["wg"]
    u = x @ w["wu"]
    return (g / (1.0 + np.exp(-g)) * u) @ w["wd"]


def _heads(x, W, H, Dh):
    # x [B,S,Dm] @ W [Dm, H*Dh] -> [B,H,S,Dh]
    B, S, _ = x.shape
    y = x @ W
    return np.transpose(y.reshape(B, S, H, Dh), (0, 2, 1, 3))


def _merge(o, Wo):
    # o [B,H,S,Dh] -> [B,S,H*Dh] -> [B,S,Dm]
    B, H, S, Dh = o.shape
    return np.transpose(o, (0, 2, 1, 3)).reshape(B, S, H * Dh) @ Wo


def _norm_last(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def _per_head_scalar(x, W):
    # x [B,S,Dm] @ W [Dm,H] -> sigmoid -> [B,H,S]
    y = 1.0 / (1.0 + np.exp(-(x @ W)))
    return np.transpose(y, (0, 2, 1))


def _causal_attention(Q, K, V):
    # Q,K,V [B,H,S,Dh] -> [B,H,S,Dh], causal.
    Dh = Q.shape[-1]
    scores = np.einsum("bhsd,bhtd->bhst", Q, K) / np.sqrt(Dh)
    S = Q.shape[2]
    mask = np.tril(np.ones((S, K.shape[2]), bool), k=K.shape[2] - S)
    scores = np.where(mask, scores, -1e30)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("bhst,bhtd->bhsd", p, V)


def _layer_full_forward(x, w, cfg):
    B, S, _ = x.shape
    H, Dh = cfg.num_heads, cfg.head_dim
    h = _rmsnorm(x, w["n1"])
    Q = _heads(h, w["wq"], H, Dh)
    K = _heads(h, w["wk"], H, Dh)
    V = _heads(h, w["wv"], H, Dh)
    o = _causal_attention(Q, K, V)
    x = x + _merge(o, w["wo"])
    return x + _swiglu(_rmsnorm(x, w["n2"]), w)


def _layer_linear_forward(x, w, cfg):
    H, Dh = cfg.num_heads, cfg.head_dim
    h = _rmsnorm(x, w["n1"])
    Q = _heads(h, w["wq"], H, Dh)
    K = _norm_last(_heads(h, w["wk"], H, Dh))   # L2-normalized keys (L1.1 finding)
    V = _heads(h, w["wv"], H, Dh)
    beta = _per_head_scalar(h, w["wbeta"])
    decay = _per_head_scalar(h, w["wdecay"])
    o = np.asarray(_dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay))
    x = x + _merge(o, w["wo"])
    return x + _swiglu(_rmsnorm(x, w["n2"]), w)


def hybrid_forward(x, weights, cfg: HybridConfig):
    """Full (parallel) forward over the whole sequence; returns hidden [B,S,Dm]."""
    sched = cfg.schedule
    for i, w in enumerate(weights):
        x = (_layer_full_forward if sched.is_full(i) else _layer_linear_forward)(x, w, cfg)
    return x


# ── streaming dual-cache decode ──────────────────────────────────────────────
def _layer_full_step(x_t, w, cfg, cache):
    """One token through a full-attention layer; cache = dict(K=[B,H,t,Dh], V=...)."""
    H, Dh = cfg.num_heads, cfg.head_dim
    h = _rmsnorm(x_t, w["n1"])
    q = _heads(h, w["wq"], H, Dh)
    k = _heads(h, w["wk"], H, Dh)
    v = _heads(h, w["wv"], H, Dh)
    cache["K"] = k if cache.get("K") is None else np.concatenate([cache["K"], k], axis=2)
    cache["V"] = v if cache.get("V") is None else np.concatenate([cache["V"], v], axis=2)
    o = _causal_attention(q, cache["K"], cache["V"])  # q over all cached keys
    x_t = x_t + _merge(o, w["wo"])
    return x_t + _swiglu(_rmsnorm(x_t, w["n2"]), w)


def _layer_linear_step(x_t, w, cfg, cache):
    """One token through a linear layer; cache = dict(S=Ŝ[B,H,Dh,Dh])."""
    H, Dh = cfg.num_heads, cfg.head_dim
    h = _rmsnorm(x_t, w["n1"])
    q = _heads(h, w["wq"], H, Dh)
    k = _norm_last(_heads(h, w["wk"], H, Dh))
    v = _heads(h, w["wv"], H, Dh)
    beta = _per_head_scalar(h, w["wbeta"])
    decay = _per_head_scalar(h, w["wdecay"])
    o, new_state = _dr.gated_delta_rule_recurrent(
        q, k, v, beta=beta, decay=decay, state=cache.get("S"),
        return_state=True, state_dtype="fp64")
    cache["S"] = new_state
    x_t = x_t + _merge(np.asarray(o), w["wo"])
    return x_t + _swiglu(_rmsnorm(x_t, w["n2"]), w)


def hybrid_decode(x, weights, cfg: HybridConfig, prefill: int = 1):
    """Stream the sequence token-by-token (after a `prefill` chunk) carrying the
    dual cache — recurrent Ŝ for linear layers, KV for full layers.  Returns
    hidden [B,S,Dm].  Equals `hybrid_forward` for the same input (the oracle)."""
    sched = cfg.schedule
    B, S, Dm = x.shape
    caches = [{} for _ in weights]
    out = np.zeros_like(x)

    def run_span(x_span, lo, hi):
        for i, w in enumerate(weights):
            if sched.is_full(i):
                if x_span.shape[1] == 1:
                    x_span = _layer_full_step(x_span, w, cfg, caches[i])
                else:
                    # prefill span: process then seed the KV cache.
                    h = _rmsnorm(x_span, w["n1"])
                    H, Dh = cfg.num_heads, cfg.head_dim
                    q = _heads(h, w["wq"], H, Dh); k = _heads(h, w["wk"], H, Dh); v = _heads(h, w["wv"], H, Dh)
                    caches[i]["K"], caches[i]["V"] = k, v
                    o = _causal_attention(q, k, v)
                    x_span = x_span + _merge(o, w["wo"])
                    x_span = x_span + _swiglu(_rmsnorm(x_span, w["n2"]), w)
            else:
                if x_span.shape[1] == 1:
                    x_span = _layer_linear_step(x_span, w, cfg, caches[i])
                else:
                    H, Dh = cfg.num_heads, cfg.head_dim
                    h = _rmsnorm(x_span, w["n1"])
                    q = _heads(h, w["wq"], H, Dh); k = _norm_last(_heads(h, w["wk"], H, Dh)); v = _heads(h, w["wv"], H, Dh)
                    beta = _per_head_scalar(h, w["wbeta"]); decay = _per_head_scalar(h, w["wdecay"])
                    o, st = _dr.gated_delta_rule_recurrent(
                        q, k, v, beta=beta, decay=decay, return_state=True, state_dtype="fp64")
                    caches[i]["S"] = st
                    x_span = x_span + _merge(np.asarray(o), w["wo"])
                    x_span = x_span + _swiglu(_rmsnorm(x_span, w["n2"]), w)
        return x_span

    out[:, :prefill] = run_span(x[:, :prefill], 0, prefill)
    for t in range(prefill, S):
        out[:, t:t + 1] = run_span(x[:, t:t + 1], t, t + 1)
    return out
