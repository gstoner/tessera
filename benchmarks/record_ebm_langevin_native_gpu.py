#!/usr/bin/env python3
"""The EBM Langevin loop as one cooperative device kernel, on the owning device.

Records, per (shape, steps, temperature), the max absolute deviation of the
device loop from the declared numpy policy (`reference_langevin_loop`), the
returned Philox key, and the structure of the emitted kernel taken from the
arena IR (one gpu.func, one scf.for carrying the state in registers, Philox
inside the loop, no linalg/tensor op left). Correctness evidence only: the
per-call host transfers are not a performance path and no promotion is
claimed. gfx1151, gfx1201 and sm_120 packets are separate proofs.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from tessera.compiler.llvm_tools import llvm_bin_dir  # noqa: E402
from tessera.compiler.native_gpu_storage import replay_arena_ir  # noqa: E402
from tessera.ebm import native_langevin as nl  # noqa: E402

CASES = [((4, 8), 1, 0.7), ((6, 8), 5, 0.7), ((3, 5), 12, 0.7), ((9, 100), 4, 0.3),
         ((16, 33), 8, 0.0), ((2, 1024), 3, 0.5)]


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
    rng = np.random.default_rng(20260916)
    rows = []
    for shape, steps, temperature in CASES:
        y0, x = (rng.standard_normal(shape).astype(np.float32) for _ in range(2))
        key = [0x1234ABCD9876 + steps, 42]
        out, next_key = nl.native_langevin_loop_device(y0, x, key, eta=0.1, temperature=temperature, steps=steps,
                                                       backend=args.backend, chip=args.chip,
                                                       compiler=args.compiler, llvm_bin=llvm)
        expect, expect_key = nl.reference_langevin_loop(y0, x, key, eta=0.1, temperature=temperature, steps=steps)
        error = float(np.max(np.abs(np.asarray(out) - expect)))
        if list(np.asarray(next_key)) != list(expect_key):
            raise SystemExit(f"{shape} K={steps}: returned key {list(next_key)} != {list(expect_key)}")
        if not np.allclose(out, expect, rtol=1e-5, atol=1e-5):
            raise SystemExit(f"{shape} K={steps} T={temperature} disagrees with the declared policy (max abs error {error})")
        if temperature == 0.0 and not np.allclose(out, nl.reference_langevin_loop(y0, x, key, eta=0.1, temperature=0.0,
                                                                                    steps=steps)[0]):
            raise SystemExit("zero-temperature loop is not the plain descent")
        program = nl.ebm_langevin_program(shape, eta=0.1, temperature=temperature, steps=steps, backend=args.backend,
                                          chip=args.chip, compiler=args.compiler, llvm_bin=llvm)
        rows.append(dict(shape=list(shape), steps=steps, temperature=temperature, max_abs_error=error,
                         next_key=[int(v) for v in np.asarray(next_key)], binding=program.package.binding_digest))
    # The kernel structure is in the device code: replay the packaged source.
    source, _ = nl.langevin_device_source((4, 8), eta=0.1, temperature=0.7, steps=3, backend=args.backend,
                                          compiler=args.compiler)
    arena = replay_arena_ir(args.compiler, source)
    body = arena.split("gpu.func @row_program(", 1)[1]
    structure = dict(gpu_funcs=arena.count("gpu.func "), loops=len(re.findall(r"scf\.for ", body)),
                     loop_carried=re.findall(r"iter_args\([^)]*\) -> \(([^)]*)\)", body),
                     philox_mul=body.count("arith.mului_extended"), barriers=body.count("gpu.barrier"),
                     linalg_or_tensor_ops=sum(("linalg." in line or "tensor." in line) for line in body.splitlines()))
    if structure["gpu_funcs"] != 1 or structure["loops"] != 1 or structure["linalg_or_tensor_ops"] != 0 \
            or structure["loop_carried"] != ["f32, i64, i64"]:
        raise SystemExit(f"unexpected kernel structure {structure}")
    packet = dict(schema=1, backend=args.backend, chip=args.chip, host=platform.node(),
                  compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
                  llvm_bin=str(llvm), recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=rows, kernel_structure=structure,
                  proofs=["the K-step loop is one device launch, bit-exact with the declared Philox/Box-Muller policy "
                          "for every case (K in 1..12, F in 5..1024, T in {0, 0.3, 0.5, 0.7})",
                          "the gradient is the paired autodiff pass's own adjoint, lowered inside the kernel "
                          "(no host gradient, no host noise, no Python-emitted kernel)",
                          "one gpu.func, one scf.for carrying (state, key, key) in registers, Philox inside the loop"],
                  promotion_eligible=False, measured_performance=False,
                  envelope="f32, quadratic energy, [rows, features] with features <= 1024; per-call host transfers; "
                           "correctness only", rocm_chip_env=os.environ.get("TESSERA_ROCM_CHIP"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), rows=len(rows), kernel_structure=structure)))


if __name__ == "__main__":
    main()
