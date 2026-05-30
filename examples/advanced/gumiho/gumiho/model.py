"""Gumiho model surface — target model + hybrid draft heads.

All dense math goes through a backend object (``NumpyBackend`` or
``AppleBackend``); host glue (embedding gather, head reshape, score scale,
mask add, concat) is plain numpy so the *same* code path runs on either
backend. Weights are tiny seeded synthetics — this example proves the
architecture + backend execution, not pretrained quality.

Components (Gumiho, ICML'25 — arXiv:2503.10135):
  * ``TargetModel``  — the "big" model: 1 decoder layer (RMSNorm + MHA +
    SwiGLU) + LM head. Exposes hidden states for the draft heads and supports
    an additive attention mask for tree verification.
  * ``SerialHead``   — EAGLE-style 2-layer Transformer that generates the
    first ``serial_tokens`` draft tokens autoregressively from
    ``concat(target hidden, token embedding)``.
  * ``ParallelHeads``— ``parallel_heads`` Medusa-style MLPs (FC→ReLU→FC) that
    predict the remaining positions in parallel from the serial outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import GumihoConfig

NEG_INF = -1e30


# ─────────────────────────────────────────────────────────────────────────────
# Weights
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _Layer:
    ln1: np.ndarray
    wqkv: np.ndarray
    wo: np.ndarray
    ln2: np.ndarray
    w_gate: np.ndarray
    w_up: np.ndarray
    w_down: np.ndarray


@dataclass(frozen=True)
class GumihoWeights:
    embed: np.ndarray
    target_layer: _Layer
    final_norm: np.ndarray
    lm_head: np.ndarray
    serial_fc_in: np.ndarray
    serial_layers: tuple[_Layer, ...]
    serial_norm: np.ndarray
    parallel_fc1: tuple[np.ndarray, ...]
    parallel_fc2: tuple[np.ndarray, ...]


def _layer(rng, d: int, ffn: int, scale: float) -> _Layer:
    return _Layer(
        ln1=rng.standard_normal(d).astype(np.float32) * 0.1 + 1.0,
        wqkv=rng.standard_normal((d, 3 * d)).astype(np.float32) * scale,
        wo=rng.standard_normal((d, d)).astype(np.float32) * scale,
        ln2=rng.standard_normal(d).astype(np.float32) * 0.1 + 1.0,
        w_gate=rng.standard_normal((d, ffn)).astype(np.float32) * scale,
        w_up=rng.standard_normal((d, ffn)).astype(np.float32) * scale,
        w_down=rng.standard_normal((ffn, d)).astype(np.float32) * scale,
    )


def make_weights(cfg: GumihoConfig, seed: int = 0) -> GumihoWeights:
    rng = np.random.default_rng(seed)
    d, ffn, V = cfg.d_model, cfg.ffn_hidden, cfg.vocab
    s = 1.0 / math.sqrt(d)
    return GumihoWeights(
        embed=rng.standard_normal((V, d)).astype(np.float32) * 0.2,
        target_layer=_layer(rng, d, ffn, s),
        final_norm=rng.standard_normal(d).astype(np.float32) * 0.1 + 1.0,
        lm_head=rng.standard_normal((d, V)).astype(np.float32) * s,
        serial_fc_in=rng.standard_normal((2 * d, d)).astype(np.float32) * s,
        serial_layers=tuple(_layer(rng, d, ffn, s) for _ in range(cfg.serial_layers)),
        serial_norm=rng.standard_normal(d).astype(np.float32) * 0.1 + 1.0,
        parallel_fc1=tuple(
            rng.standard_normal((2 * d, cfg.parallel_hidden)).astype(np.float32) * s
            for _ in range(cfg.parallel_heads)
        ),
        parallel_fc2=tuple(
            rng.standard_normal((cfg.parallel_hidden, d)).astype(np.float32) * s
            for _ in range(cfg.parallel_heads)
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared transformer pieces (backend-driven)
# ─────────────────────────────────────────────────────────────────────────────
def _attention(backend, h: np.ndarray, layer: _Layer, cfg: GumihoConfig,
               add_mask: Any | None) -> np.ndarray:
    """Multi-head self-attention with an optional additive mask ``[T, T]``.

    ``add_mask`` is 0 where attention is allowed and ``NEG_INF`` where masked
    (causal for plain decode, tree-ancestor for verification)."""
    T, d = h.shape
    H, dh = cfg.num_heads, cfg.head_dim
    qkv = backend.linear(h, layer.wqkv)              # [T, 3d]  (GPU/CPU)
    q, k, v = np.split(np.asarray(qkv), 3, axis=-1)  # host split
    fold = lambda t: np.ascontiguousarray(           # noqa: E731  [T,d]->[H,T,dh]
        np.asarray(t).reshape(T, H, dh).transpose(1, 0, 2))
    qh, kh, vh = fold(q), fold(k), fold(v)
    scale = 1.0 / math.sqrt(dh)
    scores = backend.matmul(qh, kh.transpose(0, 2, 1)) * scale   # [H,T,T] (GPU bmm)
    if add_mask is not None:
        scores = np.asarray(scores) + add_mask[None, :, :]
    attn = backend.softmax(np.asarray(scores).reshape(H * T, T)).reshape(H, T, T)
    ctx = backend.matmul(attn, vh)                   # [H,T,dh] (GPU bmm)
    ctx = np.ascontiguousarray(np.asarray(ctx).transpose(1, 0, 2).reshape(T, d))
    return backend.linear(ctx, layer.wo)             # [T, d]


def _decoder_layer(backend, h: np.ndarray, layer: _Layer, cfg: GumihoConfig,
                   add_mask: Any | None) -> np.ndarray:
    a = _attention(backend, backend.rmsnorm(h, layer.ln1), layer, cfg, add_mask)
    h = np.asarray(h) + np.asarray(a)
    n2 = backend.rmsnorm(h, layer.ln2)
    gate = backend.linear(n2, layer.w_gate)
    up = backend.linear(n2, layer.w_up)
    down = backend.linear(backend.silu_mul(gate, up), layer.w_down)
    return np.asarray(h) + np.asarray(down)


def _causal_mask(T: int) -> np.ndarray:
    m = np.zeros((T, T), np.float64)
    m[np.triu_indices(T, k=1)] = NEG_INF
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Target model
# ─────────────────────────────────────────────────────────────────────────────
class TargetModel:
    def __init__(self, weights: GumihoWeights, cfg: GumihoConfig) -> None:
        self.w = weights
        self.cfg = cfg

    def embed(self, tokens: np.ndarray) -> np.ndarray:
        return self.w.embed[np.asarray(tokens, np.int64)]

    def forward(self, backend, tokens: np.ndarray, *,
                add_mask: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(hidden [T, d], logits [T, V])``. ``add_mask`` defaults to
        causal; pass a tree-ancestor mask for verification."""
        tokens = np.asarray(tokens, np.int64)
        T = tokens.shape[0]
        if add_mask is None:
            add_mask = _causal_mask(T)
        h = self.embed(tokens)
        h = _decoder_layer(backend, h, self.w.target_layer, self.cfg, add_mask)
        hn = backend.rmsnorm(h, self.w.final_norm)
        logits = backend.linear(hn, self.w.lm_head)
        return np.asarray(h), np.asarray(logits)


# ─────────────────────────────────────────────────────────────────────────────
# Serial head — EAGLE-style autoregressive Transformer
# ─────────────────────────────────────────────────────────────────────────────
class SerialHead:
    def __init__(self, weights: GumihoWeights, cfg: GumihoConfig) -> None:
        self.w = weights
        self.cfg = cfg

    def generate(self, backend, target: TargetModel, last_hidden: np.ndarray,
                 last_token: int) -> tuple[list[int], np.ndarray, list[np.ndarray]]:
        """Autoregressively produce ``serial_tokens`` draft tokens.

        Returns ``(tokens, log_probs[serial_tokens, V], hidden_outputs)``.
        ``hidden_outputs`` are the per-step serial hidden states fed to the
        parallel heads.
        """
        tokens: list[int] = []
        log_probs: list[np.ndarray] = []
        hiddens: list[np.ndarray] = []
        h_t = np.asarray(last_hidden, np.float64).reshape(self.cfg.d_model)
        y_t = int(last_token)
        for _ in range(self.cfg.serial_tokens):
            e = target.embed(np.array([y_t]))[0].astype(np.float64)
            x = np.concatenate([h_t, e]).reshape(1, 2 * self.cfg.d_model)
            h = backend.linear(x, self.w.serial_fc_in)          # [1, d]
            mask = _causal_mask(1)
            for layer in self.w.serial_layers:
                h = _decoder_layer(backend, np.asarray(h), layer, self.cfg, mask)
            hn = backend.rmsnorm(h, self.w.serial_norm)
            logits = np.asarray(backend.linear(hn, self.w.lm_head))[0]   # [V]
            lp = logits - _logsumexp(logits)
            tok = int(np.argmax(logits))
            tokens.append(tok)
            log_probs.append(lp)
            hiddens.append(np.asarray(h, np.float64).reshape(self.cfg.d_model))
            h_t = hiddens[-1]
            y_t = tok
        return tokens, np.stack(log_probs), hiddens


# ─────────────────────────────────────────────────────────────────────────────
# Parallel heads — Medusa-style MLPs
# ─────────────────────────────────────────────────────────────────────────────
class ParallelHeads:
    def __init__(self, weights: GumihoWeights, cfg: GumihoConfig) -> None:
        self.w = weights
        self.cfg = cfg

    def predict(self, backend, serial_hiddens: list[np.ndarray]) -> np.ndarray:
        """Return per-head log-probs ``[parallel_heads, V]``.

        Every head sees the same input: the concatenated serial-step hidden
        outputs. Heads run independently (no autoregression) — that
        independence is what Full Tree Attention later exploits.
        """
        d = self.cfg.d_model
        feat = np.concatenate([np.asarray(serial_hiddens[0], np.float64),
                               np.asarray(serial_hiddens[-1], np.float64)])
        feat = feat.reshape(1, 2 * d)
        out = np.empty((self.cfg.parallel_heads, self.cfg.vocab), np.float64)
        for i in range(self.cfg.parallel_heads):
            h1 = backend.relu(backend.linear(feat, self.w.parallel_fc1[i]))
            h2 = backend.linear(h1, self.w.parallel_fc2[i])     # [1, d]
            logits = np.asarray(backend.linear(h2, self.w.lm_head))[0]
            out[i] = logits - _logsumexp(logits)
        return out


def _logsumexp(z: np.ndarray) -> float:
    z = np.asarray(z, np.float64)
    m = z.max()
    return float(m + np.log(np.exp(z - m).sum()))
