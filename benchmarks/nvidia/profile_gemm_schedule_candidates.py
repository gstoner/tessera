"""Launch production fused candidates for targeted Nsight resource capture."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def main() -> int:
    from tessera.compiler.emit import nvidia_cuda  # noqa: F401
    from tessera.compiler.emit.candidate import OP_FUSED_REGION, candidates_for
    from tessera.compiler.fusion import FusedRegion

    rng = np.random.default_rng(1208)
    a = (rng.standard_normal((256, 256)) * .1).astype(np.float32)
    b = (rng.standard_normal((256, 256)) * .1).astype(np.float32)
    bias = (rng.standard_normal(256) * .05).astype(np.float32)
    region = FusedRegion(epilogue=("bias", "gelu"))
    wanted = {"nvidia_generic_cuda", "nvidia_mma_fused"}
    for candidate in candidates_for("nvidia", OP_FUSED_REGION):
        if candidate.name in wanted:
            _, tag = candidate.run(region, a, b, bias)
            if tag == "reference":
                raise RuntimeError(f"{candidate.name} declined profiling inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
