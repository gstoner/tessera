"""Reference causal-decoder target model for DFlash (numpy).

A small, self-contained autoregressive transformer used as the *target* a DFlash
draft accelerates. It is deliberately not tied to any external framework — it is
the kind of model DFlash conditions on, with the two pieces DFlash needs:

  * a multi-layer **hidden-state tap** (concat of a fixed set of layers' outputs)
    that conditions the draft (DFlash #4), and
  * a **stateful KV cache with rollback** so verification can speculate b tokens
    and discard the over-speculated tail (DFlash #3).

``forward`` (stateless, full-sequence) is the greedy-AR ground truth; ``step`` /
``rollback`` are the incremental cached path. The two agree exactly — that
equivalence is the correctness contract (``tests/unit/test_dflash_reference_target.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .dflash import make_rope


@dataclass
class DecoderLMConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    target_layer_ids: Tuple[int, ...] = (0,)
    tie_lm_head: bool = True


@dataclass
class DecoderLayerWeights:
    q_proj: np.ndarray
    k_proj: np.ndarray
    v_proj: np.ndarray
    o_proj: np.ndarray
    input_layernorm: np.ndarray
    post_attention_layernorm: np.ndarray
    mlp_gate: np.ndarray
    mlp_up: np.ndarray
    mlp_down: np.ndarray


def _rms(x, w, eps):
    x = np.asarray(x, dtype=np.float32)
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y * w


def _silu(x):
    return x / (1.0 + np.exp(-x))


class ReferenceDecoderLM:
    """Numpy causal decoder LM with a multi-layer hidden tap and a rollback-able
    KV cache. ``weights`` is a list of :class:`DecoderLayerWeights`; ``embed`` is
    ``(V, D)``; ``final_norm`` is ``(D,)``; ``lm_head`` is ``(D, V)`` or ``None``
    (tied to ``embed.T``)."""

    def __init__(self, cfg: DecoderLMConfig, embed, layers: List[DecoderLayerWeights],
                 final_norm, lm_head=None):
        self.cfg = cfg
        self.embed = np.asarray(embed, dtype=np.float32)
        self.layers = layers
        self.final_norm = np.asarray(final_norm, dtype=np.float32)
        self.lm_head = None if lm_head is None else np.asarray(lm_head, dtype=np.float32)
        self.rope = make_rope(cfg.head_dim, cfg.rope_theta)
        self.scale = cfg.head_dim ** -0.5
        # Stateful cache: per-layer roped-K and V, shape (B, H, T, Dh).
        self._k: List[Optional[np.ndarray]] = [None] * cfg.num_layers
        self._v: List[Optional[np.ndarray]] = [None] * cfg.num_layers
        self._len = 0

    # -- shared layer math --------------------------------------------------
    def _project_heads(self, x, W):
        B, T, _ = x.shape
        H, Dh = self.cfg.num_heads, self.cfg.head_dim
        return (np.asarray(x) @ W).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

    def _attend(self, q, k_all, v_all, q_offset):
        # q: (B,H,Tq,Dh) already roped; k_all/v_all: (B,H,Tk,Dh) (k roped).
        Tq, Tk = q.shape[2], k_all.shape[2]
        s = np.einsum("bhqd,bhkd->bhqk", q, k_all) * self.scale
        qpos = q_offset + np.arange(Tq)[:, None]
        kpos = np.arange(Tk)[None, :]
        s = np.where(kpos <= qpos, s, -1e30)
        s = s - s.max(-1, keepdims=True)
        a = np.exp(s); a /= a.sum(-1, keepdims=True)
        return np.einsum("bhqk,bhkd->bhqd", a, v_all)

    def _layer(self, h, lw: DecoderLayerWeights, k_cache, v_cache, offset):
        """One pre-norm layer; returns (h_out, roped_k_new, v_new, layer_output)."""
        cfg = self.cfg
        B, T, _ = h.shape
        xn = _rms(h, lw.input_layernorm, cfg.rms_norm_eps)
        q = self.rope(self._project_heads(xn, lw.q_proj), offset)
        k_new = self.rope(self._project_heads(xn, lw.k_proj), offset)
        v_new = self._project_heads(xn, lw.v_proj)
        k_all = k_new if k_cache is None else np.concatenate([k_cache, k_new], axis=2)
        v_all = v_new if v_cache is None else np.concatenate([v_cache, v_new], axis=2)
        o = self._attend(q, k_all, v_all, offset)
        o = o.transpose(0, 2, 1, 3).reshape(B, T, cfg.num_heads * cfg.head_dim) @ lw.o_proj
        h = np.asarray(h) + o
        hn = _rms(h, lw.post_attention_layernorm, cfg.rms_norm_eps)
        mlp = (_silu(hn @ lw.mlp_gate) * (hn @ lw.mlp_up)) @ lw.mlp_down
        h = h + mlp
        return h, k_new, v_new

    def _run(self, tokens, *, offset, k_cache, v_cache, update_cache):
        """Core loop shared by forward()/step(). Returns (logits, hidden_concat)."""
        cfg = self.cfg
        h = self.embed[np.asarray(tokens, dtype=np.int64)]
        taps = []
        for i, lw in enumerate(self.layers):
            kc = None if k_cache is None else k_cache[i]
            vc = None if v_cache is None else v_cache[i]
            h, k_new, v_new = self._layer(h, lw, kc, vc, offset)
            if i in cfg.target_layer_ids:
                taps.append(h)
            if update_cache:
                self._k[i] = k_new if self._k[i] is None else np.concatenate([self._k[i], k_new], axis=2)
                self._v[i] = v_new if self._v[i] is None else np.concatenate([self._v[i], v_new], axis=2)
        hidden = np.concatenate(taps, axis=-1) if taps else None
        norm = _rms(h, self.final_norm, cfg.rms_norm_eps)
        lm = self.embed.T if self.lm_head is None else self.lm_head
        logits = norm @ lm
        return logits, hidden

    # -- stateless full-sequence forward (greedy-AR ground truth) -----------
    def forward(self, tokens):
        return self._run(tokens, offset=0, k_cache=None, v_cache=None, update_cache=False)

    # -- stateful KV-cached path (#3) ---------------------------------------
    def reset(self) -> None:
        self._k = [None] * self.cfg.num_layers
        self._v = [None] * self.cfg.num_layers
        self._len = 0

    @property
    def cache_len(self) -> int:
        return self._len

    def step(self, new_tokens):
        """Process ``new_tokens`` against the cache, append them, and return
        ``(logits, hidden)`` for the new positions only."""
        toks = np.asarray(new_tokens, dtype=np.int64)
        offset = self._len
        logits, hidden = self._run(toks, offset=offset, k_cache=self._k,
                                   v_cache=self._v, update_cache=True)
        self._len += toks.shape[-1]
        return logits, hidden

    def rollback(self, n: int) -> None:
        """Drop the last ``n`` tokens from the cache (discard over-speculation)."""
        n = int(n)
        if n <= 0:
            return
        keep = self._len - n
        for i in range(self.cfg.num_layers):
            ki, vi = self._k[i], self._v[i]
            if ki is not None and vi is not None:
                self._k[i] = ki[:, :, :keep, :]
                self._v[i] = vi[:, :, :keep, :]
        self._len = keep


def random_decoder_lm(cfg: DecoderLMConfig, rng) -> ReferenceDecoderLM:
    """Build a ReferenceDecoderLM with small random weights (for tests/examples)."""
    D, H, Dh, I, V = cfg.hidden_size, cfg.num_heads, cfg.head_dim, cfg.intermediate_size, cfg.vocab_size
    s = lambda *sh: (rng.standard_normal(sh).astype(np.float32) * 0.08)
    layers = [DecoderLayerWeights(
        q_proj=s(D, H * Dh), k_proj=s(D, H * Dh), v_proj=s(D, H * Dh), o_proj=s(H * Dh, D),
        input_layernorm=s(D) + 1.0, post_attention_layernorm=s(D) + 1.0,
        mlp_gate=s(D, I), mlp_up=s(D, I), mlp_down=s(I, D),
    ) for _ in range(cfg.num_layers)]
    lm = None if cfg.tie_lm_head else s(D, V)
    return ReferenceDecoderLM(cfg, s(V, D), layers, s(D) + 1.0, lm)


__all__ = ["DecoderLMConfig", "DecoderLayerWeights", "ReferenceDecoderLM", "random_decoder_lm"]
