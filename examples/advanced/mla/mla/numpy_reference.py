"""Dependency-light Multi-Latent Attention reference path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MLAConfig


@dataclass(frozen=True)
class MLAResult:
    output: np.ndarray
    kv_latent: np.ndarray
    attn_probs: np.ndarray
    cache_bytes_full_kv: int
    cache_bytes_latent: int

    @property
    def kv_cache_reduction(self) -> float:
        return 1.0 - (self.cache_bytes_latent / self.cache_bytes_full_kv)


class MultiLatentAttentionNumpy:
    """Tiny MLA forward pass with compressed KV cache accounting."""

    def __init__(self, cfg: MLAConfig, *, seed: int = 0) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        d = cfg.model_dim
        c = cfg.latent_dim
        q = cfg.num_q_heads * cfg.head_dim
        kv = cfg.num_kv_heads * cfg.head_dim
        scale = 0.02
        self.q_down = rng.normal(0.0, scale, size=(d, c)).astype(np.float32)
        self.q_up = rng.normal(0.0, scale, size=(c, q)).astype(np.float32)
        self.kv_down = rng.normal(0.0, scale, size=(d, c)).astype(np.float32)
        self.k_up = rng.normal(0.0, scale, size=(c, kv)).astype(np.float32)
        self.v_up = rng.normal(0.0, scale, size=(c, kv)).astype(np.float32)
        self.out = rng.normal(0.0, scale, size=(q, d)).astype(np.float32)

    def forward(self, hidden_states: np.ndarray) -> MLAResult:
        x = np.asarray(hidden_states, dtype=np.float32)
        if x.shape != (self.cfg.batch_size, self.cfg.seq_len, self.cfg.model_dim):
            raise ValueError(
                "hidden_states must have shape "
                f"{(self.cfg.batch_size, self.cfg.seq_len, self.cfg.model_dim)}"
            )
        cfg = self.cfg
        b, s, _ = x.shape
        q_latent = x @ self.q_down
        q_full = q_latent @ self.q_up
        q = q_full.reshape(b, s, cfg.num_q_heads, cfg.head_dim).transpose(0, 2, 1, 3)

        kv_latent = _rmsnorm(x @ self.kv_down, cfg.rms_norm_eps)
        k_full = kv_latent @ self.k_up
        v_full = kv_latent @ self.v_up
        k = k_full.reshape(b, s, cfg.num_kv_heads, cfg.head_dim)
        v = v_full.reshape(b, s, cfg.num_kv_heads, cfg.head_dim)

        repeat = cfg.num_q_heads // cfg.num_kv_heads
        k = np.repeat(k, repeat, axis=2).transpose(0, 2, 1, 3)
        v = np.repeat(v, repeat, axis=2).transpose(0, 2, 1, 3)

        scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(cfg.head_dim)
        probs = _softmax(scores, axis=-1)
        context = probs @ v
        context = context.transpose(0, 2, 1, 3).reshape(b, s, cfg.num_q_heads * cfg.head_dim)
        output = context @ self.out

        full_kv_elems = b * s * cfg.num_kv_heads * cfg.head_dim * 2
        latent_elems = b * s * cfg.latent_dim
        return MLAResult(
            output=output.astype(np.float32),
            kv_latent=kv_latent.astype(np.float32),
            attn_probs=probs.astype(np.float32),
            cache_bytes_full_kv=full_kv_elems * 2,
            cache_bytes_latent=latent_elems * 2,
        )


def _rmsnorm(x: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(variance + eps))


def _softmax(x: np.ndarray, *, axis: int) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)
