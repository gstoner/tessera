#!/usr/bin/env python3
"""What every admitted `math.*` op in a row-program kernel costs in accuracy.

`math.sqrt` on the NVVM route was found (2026-09-16) to reach libdevice's
*approximate* `__nv_sqrtf`, because the precise branch is gated on a reflect
value MLIR never sets — one row of the row-normalization proof was 1 ulp off on
sm_120 and exact on both RDNA parts. Every other `math.*` op on both device
routes is subject to the same vendor default, so this recorder measures each
admitted op against the host on the owning device instead of assuming it.

Per op it builds the smallest possible row program (`y[r, f] = op(x[r, f])`),
runs it through the same packaging route the domain loops use, and compares the
device result with numpy's f32 result over the op's declared input domains,
reporting the maximum ulp distance and where it occurred. The verdict is
recorded, not asserted: an op measured at 0 ulp on this device is exact *here*,
and the packet says which device and which toolchain produced that. A non-zero
bound is a fact about the lane, not a failure — what would be a failure is
calling a result exact without this measurement.

The emitter refuses any `math.*` op outside its admission table, so the set this
sweeps is exactly the set a row program can contain.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from tessera.compiler.llvm_tools import llvm_bin_dir  # noqa: E402
from tessera.compiler.native_gpu_tensor import TensorSpec  # noqa: E402
from tessera.compiler.native_row_program import (  # noqa: E402
    ADMITTED_MATH, MATH_AUDIT_DOMAINS, declared_math, row_program_kernel, row_program_device,
    row_unary_math_module,
)

ROWS, FEATURES = 16, 1024

#: The host reference for each admitted op, at f32 in and f32 out.
HOST = {
    "math.sqrt": np.sqrt,
    "math.absf": np.abs,
    "math.exp": np.exp,
    "math.log": np.log,
    "math.cos": np.cos,
}


def sweep(domain: tuple[float, float], count: int) -> np.ndarray:
    """`count` f32 inputs across `domain`: geometric where the domain spans
    decades (so the small end is sampled at all), linear otherwise, always
    including both endpoints."""
    lo, hi = domain
    if lo > 0.0 and hi / lo > 1e3:
        values = np.geomspace(lo, hi, count)
    else:
        values = np.linspace(lo, hi, count)
    values[0], values[-1] = lo, hi
    return values.astype(np.float32)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-element ulp distance between two f32 arrays, and a mask of elements
    whose finiteness disagrees (which no ulp count describes)."""
    both_nan = np.isnan(a) & np.isnan(b)
    same_inf = np.isinf(a) & np.isinf(b) & (np.signbit(a) == np.signbit(b))
    comparable = np.isfinite(a) & np.isfinite(b)
    disagree = ~(comparable | both_nan | same_inf)
    # Map f32 bits to a monotone integer ordering so the distance is a count of
    # representable values, sign-crossings included.
    def order(x):
        bits = x.astype(np.float32).view(np.int32).astype(np.int64)
        return np.where(bits < 0, np.int64(-0x80000000) - bits, bits)
    distance = np.zeros(a.shape, dtype=np.int64)
    distance[comparable] = np.abs(order(a[comparable]) - order(b[comparable]))
    return distance, disagree


def measure(op: str, *, backend: str, chip: str, compiler: Path, llvm: Path) -> dict:
    module = row_unary_math_module(ROWS, FEATURES, op)
    program = row_program_device(module, entry="unary_math", rows=ROWS,
                                specs=(TensorSpec("x", "fp32", (ROWS, FEATURES), False),
                                       TensorSpec("y", "fp32", (ROWS, FEATURES), True)),
                                backend=backend, chip=chip, compiler=compiler, llvm_bin=llvm,
                                name=f"row program {op}")
    host = HOST[op]
    worst = dict(ulp=0, at=None, device=None, host=None)
    non_finite_disagreements = 0
    points = 0
    for domain in MATH_AUDIT_DOMAINS[op]:
        x = sweep(domain, ROWS * FEATURES).reshape(ROWS, FEATURES)
        got = np.asarray(program.run(x), dtype=np.float32)
        want = host(x).astype(np.float32)
        distance, disagree = ulp_distance(got, want)
        non_finite_disagreements += int(disagree.sum())
        points += x.size
        index = int(np.argmax(distance))
        if int(distance.flat[index]) > worst["ulp"]:
            worst = dict(ulp=int(distance.flat[index]), at=float(x.flat[index]),
                         device=float(got.flat[index]), host=float(want.flat[index]))
    kernel, _ = row_program_kernel(module, entry="unary_math", backend=backend, compiler=compiler)
    return dict(op=op, plan=ADMITTED_MATH[op], declared=list(declared_math(kernel)),
                domains=[list(d) for d in MATH_AUDIT_DOMAINS[op]], points=points,
                max_ulp=worst["ulp"], worst_input=worst["at"], device_result=worst["device"],
                host_result=worst["host"], non_finite_disagreements=non_finite_disagreements)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["nvidia", "rocm"], required=True)
    parser.add_argument("--chip", required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    llvm = llvm_bin_dir()
    if llvm is None:
        raise SystemExit("matched LLVM tools are required")
    rows = [measure(op, backend=args.backend, chip=args.chip, compiler=args.compiler, llvm=llvm)
            for op in sorted(ADMITTED_MATH)]
    for row in rows:
        if row["declared"] != [f"{row['op']}:{row['plan']}"]:
            raise SystemExit(f"{row['op']}: the kernel declares {row['declared']}, not its admission plan")
        if row["non_finite_disagreements"]:
            raise SystemExit(f"{row['op']}: {row['non_finite_disagreements']} results disagree on finiteness, "
                             "which is a lowering defect rather than a rounding difference")
    exact = [row["op"] for row in rows if row["max_ulp"] == 0]
    inexact = {row["op"]: row["max_ulp"] for row in rows if row["max_ulp"] != 0}
    packet = dict(
        schema=1, backend=args.backend, chip=args.chip, host=platform.node(),
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        llvm_bin=str(llvm), recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows, exact_ops=exact, inexact_ops=inexact,
        proofs=["every op the row-program emitter admits is measured against numpy f32 on this device "
                "over its declared input domains, at 16384 points per domain",
                "each kernel declares its own admission plan in tessera.row_program.math, and the "
                "recorder refuses a kernel whose declaration does not match the table",
                "no result disagrees with the host on finiteness (a lowering defect would show here, "
                "not as a rounding difference)"],
        envelope="f32 unary math ops inside one cooperative row-program kernel; accuracy against the "
                 "host reference on this device and this toolchain only",
        promotion_eligible=False, measured_performance=False,
        rocm_chip_env=os.environ.get("TESSERA_ROCM_CHIP"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), exact=exact, inexact=inexact)))


if __name__ == "__main__":
    main()
