#!/usr/bin/env python3
"""Canonical Jet-Nemotron linear-attention compiler smoke at D=128.

This is host-portable compiler and numerical evidence. It intentionally does
not claim to validate the open ROCm load/wait scheduling problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "python"))

import tessera as ts  # noqa: E402


@ts.jit
def jet_linear_attention(q, k, v):
    return ts.ops.linear_attn(q, k, v, feature_map="identity", causal=False)


def _reference(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.matmul(q, np.matmul(np.swapaxes(k, -1, -2), v))


def main() -> int:
    rng = np.random.default_rng(11)
    shape = (1, 2, 8, 128)
    q = rng.standard_normal(shape, dtype=np.float32) * 0.05
    k = rng.standard_normal(shape, dtype=np.float32) * 0.05
    v = rng.standard_normal(shape, dtype=np.float32) * 0.05
    actual = jet_linear_attention(q, k, v)
    expected = _reference(q, k, v)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    artifacts = (
        jet_linear_attention.ir_text(),
        jet_linear_attention.schedule_ir,
        jet_linear_attention.tile_ir,
        jet_linear_attention.target_ir,
    )
    if not all(artifacts):
        raise RuntimeError("Jet linear-attention smoke expected all compiler artifacts")
    if "linear_attn" not in artifacts[0]:
        raise RuntimeError("Graph IR did not preserve the linear_attn operation")
    print("OK Jet linear_attn:", shape, jet_linear_attention.execution_kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
