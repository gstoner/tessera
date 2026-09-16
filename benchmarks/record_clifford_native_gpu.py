#!/usr/bin/env python3
"""Clifford product family through the native GPU storage route, on the owning device.

Records, per op, the max absolute deviation from the standalone GA reference
for single / batched / rank-3 inputs, the grade-pruned term count taken from
the arena IR (the pruning is in the emitted device code), and the identities
of the compiler, the Clifford driver and the LLVM tools that produced the
kernels. Correctness evidence only: per-call host transfers are not a
performance path and no promotion is claimed. gfx1151 and gfx1201 packets are
separate proofs and never transfer.
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
from tessera.compiler import native_clifford_gpu as ncg  # noqa: E402
from tessera.compiler.llvm_tools import llvm_bin_dir  # noqa: E402
from tessera.compiler.native_gpu_storage import replay_arena_ir  # noqa: E402
import tessera._clifford_ops as ref  # noqa: E402

REF = {
    "geo_product": "clifford_geometric_product", "wedge": "clifford_wedge",
    "left_contract": "clifford_left_contraction", "inner": "clifford_inner", "norm": "clifford_norm",
    "rotor_sandwich": "clifford_rotor_sandwich", "reverse": "clifford_reverse",
    "grade_involute": "clifford_grade_involution", "conjugate": "clifford_conjugate",
    "hodge_star": "clifford_hodge_star",
}


def reference(op, *arrays):
    fn = getattr(ref, REF[op])
    flat = [x.reshape(-1, 8) for x in arrays]
    rows = [np.asarray(fn(*[f[i] for f in flat]), dtype=np.float32) for i in range(flat[0].shape[0])]
    return np.stack(rows).reshape(arrays[0].shape[:-1] + rows[0].shape).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["nvidia", "rocm"], required=True)
    parser.add_argument("--chip", required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    llvm, clifford = llvm_bin_dir(), ncg.find_ts_clifford_opt()
    if llvm is None or clifford is None:
        raise SystemExit("matched LLVM tools and ts-clifford-opt are required")
    rng = np.random.default_rng(20260916)
    rows = []
    for op in sorted(REF):
        arity = ncg.CLIFFORD_GPU_OPS[op][0]
        for shape in [(8,), (37, 8), (3, 5, 8)]:
            arrays = [rng.standard_normal(shape).astype(np.float32) for _ in range(arity)]
            if op == "rotor_sandwich":
                rotor = np.zeros(shape, np.float32); rotor[..., 0] = np.cos(0.3); rotor[..., 3] = np.sin(0.3)
                arrays[0] = rotor
            program = ncg.clifford_gpu_program(op, shape, backend=args.backend, chip=args.chip,
                                               compiler=args.compiler, llvm_bin=llvm, clifford_opt=clifford)
            out = program.run(*arrays)
            expect = reference(op, *arrays)
            if ncg.CLIFFORD_GPU_OPS[op][1]:
                out = out.reshape(shape[:-1])
            error = float(np.max(np.abs(out - expect))) if out.size else 0.0
            if not np.allclose(out, expect, rtol=1e-5, atol=1e-5):
                raise SystemExit(f"{op} {shape} disagrees with the reference (max abs error {error})")
            rows.append(dict(op=op, shape=list(shape), max_abs_error=error,
                             binding=program.package.binding_digest))
    # Grade pruning is in the device code: count the emitted products.
    def mulf(grades):
        skeleton, _ = ncg.clifford_gpu_skeleton("geo_product", (4, 8), grades=grades)
        arena = replay_arena_ir(args.compiler, ncg.expand_clifford_source(skeleton, clifford_opt=clifford))
        return sum("arith.mulf" in line for line in arena.splitlines())
    pruning = dict(full=mulf(None), grade2=mulf([2]))
    if pruning != dict(full=64, grade2=24):
        raise SystemExit(f"unexpected emitted term counts {pruning}")
    a, b = (rng.standard_normal((11, 8)).astype(np.float32) for _ in range(2))
    pruned = ncg.clifford_gpu_program("geo_product", (11, 8), backend=args.backend, chip=args.chip,
                                      compiler=args.compiler, llvm_bin=llvm, clifford_opt=clifford, grades=[2]).run(a, b)
    full = reference("geo_product", a, b)
    if not (np.allclose(pruned[:, [3, 5, 6]], full[:, [3, 5, 6]], rtol=1e-5, atol=1e-5)
            and np.all(pruned[:, [0, 1, 2, 4, 7]] == 0)):
        raise SystemExit("grade-2 pruned product disagrees with the projection")
    packet = dict(schema=1, backend=args.backend, chip=args.chip, host=platform.node(),
                  compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
                  clifford_opt_sha256=hashlib.sha256(Path(clifford).read_bytes()).hexdigest(),
                  llvm_bin=str(llvm), recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=rows, emitted_products=pruning,
                  proofs=["every op of the family matches the standalone GA reference for single, batched and rank-3 inputs",
                          "grade-2 restriction emits 24 of 64 products and writes the other coefficients as zero",
                          "the device kernel is the dialect's own lowering folded to scalar code; no Python-emitted kernel"],
                  promotion_eligible=False, measured_performance=False,
                  envelope="f32, Cl(3,0), static shapes; per-call host transfers; correctness only",
                  rocm_chip_env=os.environ.get("TESSERA_ROCM_CHIP"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), rows=len(rows), emitted_products=pruning)))


if __name__ == "__main__":
    main()
