#!/usr/bin/env python3
"""Compile and validate a small scheduled matrix multiplication in Tessera."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import tessera as ts  # noqa: E402


@ts.jit(cpu_tile=(16, 16, 8))
def tiled_matmul(a, b):
    return ts.ops.matmul(a, b)


def main() -> int:
    rng = np.random.default_rng(19)
    a = rng.standard_normal((32, 24), dtype=np.float32)
    b = rng.standard_normal((24, 40), dtype=np.float32)
    actual = tiled_matmul(a, b)
    np.testing.assert_allclose(actual, a @ b, rtol=2e-5, atol=2e-5)
    artifacts = (
        tiled_matmul.ir_text(),
        tiled_matmul.schedule_ir,
        tiled_matmul.tile_ir,
        tiled_matmul.target_ir,
    )
    if not all(artifacts):
        raise RuntimeError("optimization example expected all compiler artifacts")
    if "16" not in tiled_matmul.schedule_ir:
        raise RuntimeError("requested CPU tile was not represented in Schedule IR")
    print("OK scheduled matmul:", actual.shape, tiled_matmul.execution_kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
