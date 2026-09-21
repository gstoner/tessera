#!/usr/bin/env python3
"""A small Hugging Face-shaped configuration adapter on canonical Tessera.

Familiar configuration objects select a real Tessera attention graph. Weight
and tokenizer conversion remain application concerns. The example executes on
the portable reference target, validates against NumPy, and checks every
compiler artifact.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

import tessera as ts  # noqa: E402


@dataclass(frozen=True)
class PretrainedConfig:
    model_type: str
    hidden_size: int
    num_attention_heads: int
    is_decoder: bool = False

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BertConfig(PretrainedConfig):
    model_type: str = "bert"
    hidden_size: int = 64
    num_attention_heads: int = 4
    is_decoder: bool = False


@dataclass(frozen=True)
class GPT2Config(PretrainedConfig):
    model_type: str = "gpt2"
    hidden_size: int = 64
    num_attention_heads: int = 4
    is_decoder: bool = True


@dataclass(frozen=True)
class LlamaConfig(PretrainedConfig):
    model_type: str = "llama"
    hidden_size: int = 64
    num_attention_heads: int = 4
    is_decoder: bool = True


@ts.jit
def encoder_attention(q, k, v):
    return ts.ops.flash_attn(q, k, v, causal=False)


@ts.jit
def decoder_attention(q, k, v):
    return ts.ops.flash_attn(q, k, v, causal=True)


def _numpy_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, *, causal: bool
) -> np.ndarray:
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        mask = np.triu(np.ones(scores.shape[-2:], dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return np.matmul(probs, v)


def run_attention(
    config: PretrainedConfig, *, sequence_length: int = 8
) -> tuple[np.ndarray, object]:
    rng = np.random.default_rng(7)
    shape = (1, config.num_attention_heads, sequence_length, config.head_dim)
    q = rng.standard_normal(shape, dtype=np.float32)
    k = rng.standard_normal(shape, dtype=np.float32)
    v = rng.standard_normal(shape, dtype=np.float32)
    compiled = decoder_attention if config.is_decoder else encoder_attention
    actual = compiled(q, k, v)
    expected = _numpy_attention(q, k, v, causal=config.is_decoder)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    artifacts = (
        compiled.ir_text(),
        compiled.schedule_ir,
        compiled.tile_ir,
        compiled.target_ir,
    )
    if not all(artifacts):
        raise RuntimeError(
            "HF adapter expected non-empty Graph, Schedule, Tile, and Target IR"
        )
    return actual, compiled


def main() -> int:
    summaries: list[dict[str, object]] = []
    for config in (BertConfig(), GPT2Config(), LlamaConfig()):
        output, compiled = run_attention(config)
        summaries.append(
            {
                "config": config.to_dict(),
                "output_shape": list(output.shape),
                "execution_kind": compiled.execution_kind,
                "oracle": "pass",
            }
        )
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
