"""Record SM120 evidence for the canonical FP16/BF16/TF32 GEMM K-loop.

The packet covers square, rectangular, ragged-K, and fully misaligned logical
shapes. It reuses the E2E spine recorder's two interleaved repeated-median
cohorts, CUDA-event and complete-call timing, discarded first complete call,
resource/spill record, and cold/warm image/cache identity. This recorder never
changes a selector; WSL observations are evidence, not promotion input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.nvidia.record_e2e_spine_dtype_matrix import record


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_canonical_k_loop.json"
STORAGES = ("fp16", "bf16", "tf32")
SHAPES = (
    (256, 256, 256),  # square
    (128, 384, 256),  # rectangular
    (128, 192, 263),  # ragged K tail
    (131, 197, 269),  # M/N/K misaligned
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--device-reps", type=int, default=100)
    parser.add_argument("--e2e-reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    payload = record(
        samples=args.samples,
        device_reps=args.device_reps,
        e2e_reps=args.e2e_reps,
        warmup=args.warmup,
        storages=STORAGES,
        shapes=SHAPES,
        work_item="CORE-GEMM-KLOOP-2026-07-25",
    )
    payload["schema"] = "tessera.nvidia.canonical-k-loop.v1"
    payload["selector_policy"] = {
        "host": "wsl",
        "eligible": False,
        "decision": "retain",
        "reason": "controlled native-Linux cross-domain stability required",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
