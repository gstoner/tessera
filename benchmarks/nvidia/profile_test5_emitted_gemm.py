"""Launch the retained compiler-emitted NVIDIA GEMM for Nsight resources."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def main() -> int:
    from tessera.compiler.emit import nvidia_cuda  # noqa: F401 - registers routes
    from tessera.compiler.emit.candidate import OP_MATMUL, candidates_for
    from tessera.compiler.fusion import MatmulRegion
    rng = np.random.default_rng(8)
    a = (rng.standard_normal((128, 64)) * .1).astype(np.float32)
    b = (rng.standard_normal((64, 256)) * .1).astype(np.float32)
    candidate = next(c for c in candidates_for("nvidia", OP_MATMUL)
                     if c.name == "nvidia_mma_gemm_emitted")
    out, tag = candidate.run(MatmulRegion(dtype="float16"), a, b)
    if tag == "reference" or np.asarray(out).shape != (128, 256):
        raise RuntimeError("compiler-emitted GEMM did not execute natively")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
