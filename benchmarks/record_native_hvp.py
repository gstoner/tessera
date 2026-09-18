#!/usr/bin/env python3
"""Compiler-owned Hessian-vector products through the native tape route, on the
owning device.

Records, per source function and shape, the max absolute deviation of the
device-resident gradient and Hessian-vector product from their closed forms,
plus the identities of the compiler and LLVM tools that produced the package.
Correctness evidence only: no performance is measured and no promotion is
claimed. sm_120, gfx1151 and gfx1201 packets are separate proofs and never
transfer (the queue owed the sm_120 one: "CUDA higher-order package binding and
exact SM120 execution remain open", NVIDIA queue 2026-09-14).
"""
import argparse
import ctypes as ct
import hashlib
import inspect
import json
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
import tessera as ts  # noqa: E402
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.native_storage_contract import (  # noqa: E402
    generate_tensor_binding, read_tensor_contract, tensor_contract_specs)


@ts.jit(target="cpu", autodiff="reverse", wrt=("x",))
def cubic(x):
    return ts.ops.mul(ts.ops.mul(x, x), x)


@ts.jit(target="cpu", autodiff="reverse", wrt=("x",))
def repeated_square(x):
    y = ts.ops.mul(x, x)
    return ts.ops.mul(y, y)


CASES = [(cubic, "cubic", 3), (repeated_square, "repeated_square", 4)]
SHAPES = [(4,), (2, 2), (3, 5)]


def run_case(function, power, shape, *, device, backend, chip, compiler, llvm_bin):
    x = np.linspace(-1.5, 2.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    package = function.compile_native_hvp(x, compiler=compiler, llvm_bin=llvm_bin,
                                          backend=backend, chip=chip)
    specs = tensor_contract_specs(read_tensor_contract(package))
    signature = inspect.Signature(
        [inspect.Parameter(s.name, inspect.Parameter.POSITIONAL_ONLY) for s in specs])
    binding = generate_tensor_binding(package, signature)
    arrays = [x, np.ones_like(x), np.ones_like(x), np.zeros_like(x), np.zeros_like(x)]
    pointers, tensors = [], []
    try:
        for a in arrays:
            p = ct.c_void_p()
            device.check(device.alloc(ct.byref(p), a.nbytes))
            pointers.append(p)
            device.check(device.htod(p, a.ctypes.data, a.nbytes) if device.cuda
                         else device.copy(p, a.ctypes.data, a.nbytes, 1))
            tensors.append(SimpleNamespace(__cuda_array_interface__=dict(
                version=3, shape=a.shape, typestr=a.dtype.str, data=(p.value, False))))
        binding(*tensors, 1)
        for a, p in zip(arrays[-2:], pointers[-2:]):
            device.check(device.dtoh(a.ctypes.data, p, a.nbytes) if device.cuda
                         else device.copy(a.ctypes.data, p, a.nbytes, 2))
    finally:
        binding.close()
        for p in pointers:
            device.check(device.free(p))
    grad_ref = power * x ** (power - 1)
    hvp_ref = power * (power - 1) * x ** (power - 2)
    grad_err = float(np.max(np.abs(arrays[-2] - grad_ref)))
    hvp_err = float(np.max(np.abs(arrays[-1] - hvp_ref)))
    if not (np.allclose(arrays[-2], grad_ref, rtol=1e-5, atol=1e-5)
            and np.allclose(arrays[-1], hvp_ref, rtol=1e-5, atol=1e-5)):
        raise SystemExit(f"{function.__name__} {shape}: device HVP disagrees with the closed form "
                         f"(grad err {grad_err}, hvp err {hvp_err})")
    return dict(shape=list(shape), grad_max_abs_error=grad_err, hvp_max_abs_error=hvp_err,
                binding_digest=getattr(package, "binding_digest", None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["nvidia", "rocm"], required=True)
    parser.add_argument("--chip", required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = Device(args.backend)
    try:
        rows = []
        for function, name, power in CASES:
            for shape in SHAPES:
                row = run_case(function, power, shape, device=device, backend=args.backend,
                               chip=args.chip, compiler=args.compiler, llvm_bin=args.llvm_bin)
                rows.append(dict(function=name, power=power, **row))
    finally:
        if device.cuda:
            destroy = device.lib.cuCtxDestroy_v2
            destroy.argtypes, destroy.restype = [ct.c_void_p], ct.c_int
            device.check(destroy(device.context))
    tools = {name: hashlib.sha256((args.llvm_bin / name).read_bytes()).hexdigest()
             for name in ("mlir-opt", "mlir-translate")
             if (args.llvm_bin / name).is_file()}
    packet = dict(
        schema="tessera.native_hvp.v1",
        backend=args.backend, chip=args.chip, host=platform.platform(),
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        llvm_tools_sha256=tools,
        rows=rows,
        evidence_scope="exact_device",
        promotion=dict(correctness_eligible=True, performance_eligible=False,
                       reason="per-call host transfers; no timer; WSL2 wall clock does not promote"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} ({len(rows)} rows, worst hvp err "
          f"{max(r['hvp_max_abs_error'] for r in rows):.3e})")


if __name__ == "__main__":
    main()
