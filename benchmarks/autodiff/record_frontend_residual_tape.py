#!/usr/bin/env python3
"""Bounded public-frontend residual capture through native MLIR AD products.

FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1: compiler-derived products,
immutable saved inputs, repeated backward and closed-frame refusal. Timing is
synchronized host wall and never confers promotion eligibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import tessera as ts
from benchmarks.record_device_ring_protocol import Device
from benchmarks.record_pool_resident_ssd import Memory
from tessera.compiler.native_gpu_storage import NativeGPUStoragePackage
from tessera.compiler.native_persistent_tape import PersistentTapePair


def record(*, backend: str, chip: str, compiler: Path, llvm_bin: Path,
           width: int = 17, samples: int = 11, case: str = "cubic") -> dict:
    if width <= 0 or samples <= 0:
        raise ValueError("positive width and samples required")
    if case not in {"cubic", "coupled"}:
        raise ValueError("unknown residual case")

    def cubic(theta, x):
        return x * x * x - theta

    def coupled(theta, x):
        return (x * x - theta) * (x + theta)

    residual = ts.jit(target="nvidia_sm120" if backend == "nvidia" else "rocm",
                      autodiff="reverse")(cubic if case == "cubic" else coupled)
    x = np.linspace(0.25, 1.25, width, dtype=np.float32)
    theta = ((x * x * x) if case == "cubic" else (0.4 * x + 0.1)).astype(np.float32)
    seed = np.linspace(-0.75, 0.75, width, dtype=np.float32)
    pair = residual.compile_persistent_device_tape(theta, x, compiler=compiler,
                llvm_bin=llvm_bin, backend=backend, chip=chip)
    # Exercise persistence before any device launch, including image and
    # checked tensor ABI identity for both products.
    pair = PersistentTapePair(*(
        NativeGPUStoragePackage.from_json(p.to_json(), expected_digest=p.binding_digest)
        for p in (pair.forward, pair.backward)), pair.lineage_digest)
    forward_contract, backward_contract = pair.validate()
    device = Device(backend)
    memory = Memory(device)
    rows = []
    try:
        dtheta, dx, dy = (memory.put(v) for v in (theta, x, seed))
        with pair.capture(dtheta, dx) as frame:
            primal = (x * x * x - theta) if case == "cubic" else ((x * x - theta) * (x + theta))
            np.testing.assert_allclose(memory.get(frame.primals[0]), primal, rtol=1e-5, atol=1e-6)
            # Mutate caller buffers after capture: derivatives must consume
            # the frame's saved inputs and residuals, not live aliases.
            memory.write(dx, np.full_like(x, 9))
            memory.write(dtheta, np.full_like(theta, -7))
            expected = ((-seed, 3 * x * x * seed) if case == "cubic" else
                        ((x * x - x - 2 * theta) * seed,
                         (3 * x * x + 2 * x * theta - theta) * seed))
            for _ in range(samples):
                start = time.perf_counter_ns()
                outputs = frame.backward(dy)
                elapsed = time.perf_counter_ns() - start
                actual = tuple(memory.get(v) for v in outputs)
                errors = []
                for value, oracle in zip(actual, expected, strict=True):
                    np.testing.assert_allclose(value, oracle, rtol=1e-5, atol=1e-6)
                    errors.append(float(np.max(np.abs(value - oracle))))
                rows.append({"backward_ns": elapsed, "max_abs_error": max(errors)})
            residual_bytes = sum(np.prod(v.__cuda_array_interface__["shape"], dtype=int)
                                 * np.dtype(v.__cuda_array_interface__["typestr"]).itemsize
                                 for v in frame.residuals)
        try:
            frame.backward(dy)
        except ValueError as error:
            if "closed" not in str(error):
                raise
        else:
            raise AssertionError("closed residual frame accepted backward")
    finally:
        memory.close()
    module = residual._traced_autodiff_module((theta, x), {})
    if module.module_attrs.get("tessera.frontend.authority") != '"tracer"':
        raise AssertionError("residual source lost tracer authority")
    return {
        "work_items": ["FRONTEND-IR-MEDIUM-1", "AD-RESIDUAL-EVAL-1"],
        "backend": backend, "chip": chip, "width": width, "case": case,
        "compiler_boundary": "public_jit_to_mlir_split_products",
        "frontend_authority": "tracer", "promotion_eligible": False,
        "timing_domain": "synchronized_host_wall",
        "timing_scope": "backward allocation, native launch and synchronization; readback excluded",
        "graph_sha256": hashlib.sha256(module.to_mlir(canonical=True).encode()).hexdigest(),
        "lineage_digest": pair.lineage_digest,
        "compiler_digest": pair.forward.compiler_digest,
        "llvm_digest": pair.forward.llvm_digest,
        "forward_binding_digest": pair.forward.binding_digest,
        "backward_binding_digest": pair.backward.binding_digest,
        "forward_abi": forward_contract, "backward_abi": backward_contract,
        "residual_bytes": int(residual_bytes),
        "saved_input_bytes": theta.nbytes + x.nbytes,
        "caller_mutation_isolated": True,
        "closed_frame_refused": True, "samples": rows,
        "backward_median_ms": statistics.median(row["backward_ns"] for row in rows) / 1e6,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("nvidia", "rocm"), required=True)
    parser.add_argument("--chip", required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, default=Path("/usr/lib/llvm-23/bin"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("cubic", "coupled"), default="cubic")
    args = parser.parse_args()
    packet = record(backend=args.backend, chip=args.chip, compiler=args.compiler,
                    llvm_bin=args.llvm_bin, case=args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps({key: packet[key] for key in ("chip", "width", "backward_median_ms", "residual_bytes")}))


if __name__ == "__main__":
    main()
