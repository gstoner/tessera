"""Small NumPy reference path for the Nemotron Nano example.

This is a shape/progress smoke, not a numerical parity implementation of the
full 12B checkpoint.  It keeps the sample runnable in the default Tessera venv
while the heavyweight PyTorch/HuggingFace path remains optional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import NemotronNanoConfig


@dataclass
class NemotronNanoWeights:
    embed: np.ndarray
    m_in: np.ndarray
    m_out: np.ndarray
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    o: np.ndarray
    mlp_in: np.ndarray
    mlp_out: np.ndarray
    norm: np.ndarray
    lm_head: np.ndarray


class NemotronNanoNumpy:
    """Tiny deterministic Mamba/attention/MLP hybrid reference."""

    def __init__(self, cfg: NemotronNanoConfig, *, seed: int = 0) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        h = cfg.hidden_size
        ff = cfg.intermediate_size
        kv = cfg.num_key_value_heads * cfg.head_dim
        scale = 0.02
        self.weights = NemotronNanoWeights(
            embed=rng.normal(0.0, scale, size=(cfg.vocab_size, h)).astype(np.float32),
            m_in=rng.normal(0.0, scale, size=(h, ff)).astype(np.float32),
            m_out=rng.normal(0.0, scale, size=(ff, h)).astype(np.float32),
            q=rng.normal(0.0, scale, size=(h, h)).astype(np.float32),
            k=rng.normal(0.0, scale, size=(h, kv)).astype(np.float32),
            v=rng.normal(0.0, scale, size=(h, kv)).astype(np.float32),
            o=rng.normal(0.0, scale, size=(h, h)).astype(np.float32),
            mlp_in=rng.normal(0.0, scale, size=(h, ff)).astype(np.float32),
            mlp_out=rng.normal(0.0, scale, size=(ff, h)).astype(np.float32),
            norm=np.ones((h,), dtype=np.float32),
            lm_head=rng.normal(0.0, scale, size=(h, cfg.vocab_size)).astype(np.float32),
        )

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        x = self.weights.embed[ids]
        for kind in self.cfg.hybrid_override_pattern:
            if kind == "M":
                x = x + self._mamba_stub(x)
            elif kind == "*":
                x = x + self._attention_gqa(x)
            elif kind == "-":
                x = x + self._mlp_relu2(x)
            else:
                raise ValueError(f"unsupported hybrid block kind {kind!r}")
        x = self._rmsnorm(x, self.weights.norm)
        return x @ self.weights.lm_head

    def _mamba_stub(self, x: np.ndarray) -> np.ndarray:
        hidden = _relu(x @ self.weights.m_in)
        return hidden @ self.weights.m_out

    def _attention_gqa(self, x: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        batch, seq, _ = x.shape
        q = (x @ self.weights.q).reshape(batch, seq, cfg.num_attention_heads, cfg.head_dim)
        k = (x @ self.weights.k).reshape(batch, seq, cfg.num_key_value_heads, cfg.head_dim)
        v = (x @ self.weights.v).reshape(batch, seq, cfg.num_key_value_heads, cfg.head_dim)
        repeat = cfg.num_attention_heads // cfg.num_key_value_heads
        k = np.repeat(k, repeat, axis=2)
        v = np.repeat(v, repeat, axis=2)
        q = np.transpose(q, (0, 2, 1, 3))
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))
        logits = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(cfg.head_dim)
        probs = _softmax(logits, axis=-1)
        y = probs @ v
        y = np.transpose(y, (0, 2, 1, 3)).reshape(batch, seq, cfg.hidden_size)
        return y @ self.weights.o

    def _mlp_relu2(self, x: np.ndarray) -> np.ndarray:
        hidden = _relu(x @ self.weights.mlp_in)
        return (hidden * hidden) @ self.weights.mlp_out

    def _rmsnorm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        variance = np.mean(x * x, axis=-1, keepdims=True)
        return x * (1.0 / np.sqrt(variance + self.cfg.rms_norm_eps)) * weight


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(x: np.ndarray, *, axis: int) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)
