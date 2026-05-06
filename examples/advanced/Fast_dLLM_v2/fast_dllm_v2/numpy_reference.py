"""Dependency-light Fast dLLM v2 reference path.

The real Fast-dLLM algorithm combines block-wise approximate KV reuse with
confidence-aware parallel decoding.  This module keeps the same control shape
but uses tiny NumPy tensors so the example can run in the repository venv.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FastDLLMConfig


@dataclass(frozen=True)
class DecodeResult:
    tokens: np.ndarray
    accepted_prefix: int
    confidences: np.ndarray
    cache_blocks: np.ndarray


class FastDLLMNumpy:
    """Tiny deterministic confidence-aware parallel decoder."""

    def __init__(self, cfg: FastDLLMConfig, *, seed: int = 0) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        h = cfg.hidden_size
        ff = cfg.intermediate_size
        scale = 0.02
        self.embed = rng.normal(0.0, scale, size=(cfg.vocab_size, h)).astype(np.float32)
        self.step = rng.normal(0.0, scale, size=(h, h)).astype(np.float32)
        self.ff_in = rng.normal(0.0, scale, size=(h, ff)).astype(np.float32)
        self.ff_out = rng.normal(0.0, scale, size=(ff, h)).astype(np.float32)
        self.lm_head = rng.normal(0.0, scale, size=(h, cfg.vocab_size)).astype(np.float32)

    def decode(self, prompt: np.ndarray) -> DecodeResult:
        prompt = np.asarray(prompt, dtype=np.int64)
        if prompt.ndim != 1:
            raise ValueError("prompt must be a rank-1 token vector")
        branch_tokens = []
        branch_conf = []
        cache_rows = []

        for branch in range(self.cfg.branch_count):
            tokens = [int(x) for x in prompt]
            conf = []
            state = self._state_from_tokens(tokens)
            for step in range(self.cfg.decode_steps):
                state = self._denoise_step(state, branch=branch, step=step)
                logits = state @ self.lm_head
                probs = _softmax(logits)
                order = np.argsort(probs)
                best = int(order[-1])
                second = int(order[-2])
                margin = float(probs[best] - probs[second])
                confidence = float(probs[best])
                conf.append(confidence + margin)
                tokens.append((best + branch + step) % self.cfg.vocab_size)
                cache_rows.append(self._pack_cache_block(state))
            branch_tokens.append(tokens)
            branch_conf.append(conf)

        token_array = np.asarray(branch_tokens, dtype=np.int64)
        confidence_array = np.asarray(branch_conf, dtype=np.float32)
        accepted = self._accepted_prefix(token_array, confidence_array)
        cache = np.asarray(cache_rows, dtype=np.float32).reshape(self.cfg.branch_count, self.cfg.decode_steps, -1)
        return DecodeResult(token_array, accepted, confidence_array, cache)

    def _state_from_tokens(self, tokens: list[int]) -> np.ndarray:
        x = self.embed[np.asarray(tokens, dtype=np.int64)]
        return _rmsnorm(np.mean(x, axis=0), self.cfg.rms_norm_eps)

    def _denoise_step(self, state: np.ndarray, *, branch: int, step: int) -> np.ndarray:
        perturb = (branch + 1) * (step + 1) * 1.0e-3
        hidden = _relu((state + perturb) @ self.ff_in)
        return _rmsnorm((hidden @ self.ff_out) + (state @ self.step), self.cfg.rms_norm_eps)

    def _pack_cache_block(self, state: np.ndarray) -> np.ndarray:
        block = state[: self.cfg.block_tokens].astype(np.float32)
        scale = np.maximum(np.max(np.abs(block)), 1.0e-6)
        return np.round(block / scale * 127.0) / 127.0 * scale

    def _accepted_prefix(self, tokens: np.ndarray, confidences: np.ndarray) -> int:
        prompt_len = tokens.shape[1] - self.cfg.decode_steps
        accepted = 0
        for offset in range(self.cfg.decode_steps):
            column = tokens[:, prompt_len + offset]
            agreement = bool(np.all(column == column[0]))
            confident = bool(np.mean(confidences[:, offset]) >= self.cfg.confidence_tau)
            margin_ok = bool(np.min(confidences[:, offset]) >= self.cfg.topk_margin_tau)
            if not (agreement or (confident and margin_ok)):
                break
            accepted += 1
        return accepted


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _rmsnorm(x: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(x * x, keepdims=True)
    return x * (1.0 / np.sqrt(variance + eps))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)
