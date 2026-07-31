"""Record a fail-closed SM120 raster-order decision packet.

This is intentionally a diagnostic recorder: it measures the same emitted
fused-GEMM implementation under its carried raster permutations, verifies each
result against the row-major output, and records CUDA-event plus host end-to-end
times separately.  It never changes selector state.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def _shape(text: str) -> tuple[int, int, int]:
    parts = tuple(int(x) for x in text.split("x"))
    if len(parts) != 3 or min(parts) < 1:
        raise argparse.ArgumentTypeError("shapes must be positive MxNxK")
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", type=_shape,
                        default=((512, 512, 512), (128, 256, 64), (127, 259, 63)))
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 2 or args.reps < 1 or args.warmup < 0:
        raise ValueError("requires at least two runs, positive reps, and nonnegative warmup")

    from tessera.compiler.emit.nvidia_cuda import _mma_fused_device_fn, _mma_fused_fn, _ptr

    variants = (("row_major", 1), ("grouped_m", 4))
    rng = np.random.default_rng(20260730)
    rows: list[dict[str, object]] = []
    for m, n, k in args.shapes:
        a = np.ascontiguousarray((rng.normal(size=(m, k)) * .1).astype(np.float16))
        b = np.ascontiguousarray((rng.normal(size=(k, n)) * .1).astype(np.float16))
        # Keep the oracle independent of the native wrapper, whose output
        # array lifetime is intentionally not part of this recorder's contract.
        expected = (np.asarray(a, np.float32) @ np.asarray(b, np.float32)).copy()
        expected.setflags(write=False)
        for order, group in variants:
            out = np.empty((m, n), np.float32)
            run = _mma_fused_fn(False, None, "f16", raster_order=order, raster_group=group)
            if run(_ptr(a), _ptr(b), None, _ptr(out), m, n, k) != 1:
                raise RuntimeError(f"{order}/{group} execution failed")
            if out.shape != expected.shape:
                raise RuntimeError(
                    f"{order}/{group} changed output shape {out.shape}; "
                    f"oracle remains {expected.shape}"
                )
            if not np.isfinite(out).all():
                raise RuntimeError(f"{order}/{group} produced non-finite output")
            max_error = float(np.max(np.abs(out - expected)))
            tolerance = float(2e-2 + 2e-2 * np.max(np.abs(expected)))
            if max_error > tolerance:
                raise RuntimeError(
                    f"{order}/{group} max error {max_error} exceeds {tolerance}"
                )
            device = _mma_fused_device_fn(False, None, "f16", raster_order=order, raster_group=group)
            device_ms = [float(device(_ptr(a), _ptr(b), None, m, n, k, args.warmup, args.reps))
                         for _ in range(args.runs)]
            e2e_ms: list[float] = []
            for _ in range(args.runs):
                start = time.perf_counter_ns()
                if run(_ptr(a), _ptr(b), None, _ptr(out), m, n, k) != 1:
                    raise RuntimeError(f"{order}/{group} end-to-end execution failed")
                e2e_ms.append((time.perf_counter_ns() - start) / 1e6)
            rows.append({"shape": [m, n, k], "raster_order": order,
                         "raster_group": group, "device_ms": device_ms,
                         "device_median_ms": statistics.median(device_ms),
                         "end_to_end_ms": e2e_ms,
                         "end_to_end_median_ms": statistics.median(e2e_ms),
                         "correctness": "matches_f32_oracle",
                         "max_abs_error": max_error})
    args.output.write_text(json.dumps({"schema": "tessera.nvidia.raster.v1",
        "device": "nvidia:sm_120", "runs": args.runs, "reps": args.reps,
        "warmup": args.warmup, "rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
