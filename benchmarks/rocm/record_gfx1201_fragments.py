"""Owning-device proof for compiler-emitted RDNA4 fragments, not tuned GEMM."""

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np
import ml_dtypes
from benchmarks.rocm.benchmark_rocm_arch_fragments import Case, _source, _lower
from tessera.compiler.native_gpu_storage import _decode_image


def record(output):
    hip = ct.CDLL("libamdhip64.so")
    P = ct.c_void_p

    def call(name, types, *args):
        fn = getattr(hip, name)
        fn.argtypes = types
        fn.restype = ct.c_int
        rc = fn(*args)
        if rc:
            raise RuntimeError(f"{name} returned {rc}")

    call("hipInit", [ct.c_uint], 0)
    llvm = Path(os.environ["TESSERA_LLVM_BIN"])
    compiler = Path(os.environ["TESSERA_OPT"])
    agents = subprocess.check_output(["rocminfo"], text=True)
    if not re.search(r"Name:\s+gfx1201\b", agents):
        raise RuntimeError("requires owning gfx1201 host")
    rows = []
    for dtype, storage in [
        ("f16", np.float16),
        ("bf16", ml_dtypes.bfloat16),
        ("e4m3", ml_dtypes.float8_e4m3fn),
        ("e5m2", ml_dtypes.float8_e5m2),
        ("int8", np.int8),
        ("int4", np.int8),
    ]:
        case = Case("gfx1201", "rdna4_wmma", dtype)
        lowered, _ = _lower(case, _source(case))
        toolkit = Path(os.environ["ROCM_PATH"]).resolve()
        pipeline = "builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,reconcile-unrealized-casts),rocdl-attach-target{chip=gfx1201},gpu-module-to-binary{toolkit={toolkit}})"
        binary = subprocess.check_output(
            [str(llvm / "mlir-opt"), "--pass-pipeline=" + pipeline.replace("{toolkit}", str(toolkit))],
            input=lowered,
            text=True,
        )
        encoded = re.search(r'bin = "((?:\\.|[^"\\])*)"', binary)
        literals = re.findall(r'"((?:\\.|[^"\\])*)"', binary)
        image = _decode_image(encoded[1] if encoded else literals[-1])
        module, fn = P(), P()
        blob = ct.create_string_buffer(image)
        call("hipModuleLoadData", [ct.POINTER(P), P], ct.byref(module), ct.cast(blob, P))
        call(
            "hipModuleGetFunction",
            [ct.POINTER(P), P, ct.c_char_p],
            ct.byref(fn),
            module,
            b"architecture_fragment_store",
        )
        k = 32 if dtype == "int4" else 16
        for m, n, active_k in [(16, 16, k), (13, 11, k - 3), (1, 7, 5)]:
            rng = np.random.default_rng(17)
            a = np.zeros((16, k), dtype=storage)
            b = np.zeros((k, 16), dtype=storage)
            a[:m, :active_k] = rng.integers(-3, 4, (m, active_k))
            b[:active_k, :n] = rng.integers(-3, 4, (active_k, n))
            result = np.zeros((16, 16), dtype=np.int32 if dtype.startswith("int") else np.float32)
            arrays = [np.ascontiguousarray(a), np.ascontiguousarray(b.T), result]
            pointers = []
            values = []
            for array in arrays:
                ptr = P()
                call("hipMalloc", [ct.POINTER(P), ct.c_size_t], ct.byref(ptr), array.nbytes)
                pointers.append(ptr)
                call("hipMemcpy", [P, P, ct.c_size_t, ct.c_int], ptr, P(array.ctypes.data), array.nbytes, 1)
                values.extend([P(ptr.value), P(ptr.value), ct.c_int64(0), ct.c_int64(array.size), ct.c_int64(1)])
            arguments = (P * len(values))(*(ct.cast(ct.pointer(v), P) for v in values))
            call(
                "hipModuleLaunchKernel",
                [P] + [ct.c_uint] * 7 + [P, ct.POINTER(P), ct.POINTER(P)],
                fn,
                1,
                1,
                1,
                32,
                1,
                1,
                0,
                None,
                arguments,
                None,
            )
            call("hipDeviceSynchronize", [])
            call("hipMemcpy", [P, P, ct.c_size_t, ct.c_int], P(result.ctypes.data), pointers[2], result.nbytes, 2)
            for ptr in pointers:
                call("hipFree", [P], ptr)
            expected = a.astype(np.float32) @ b.astype(np.float32)
            error = float(np.max(np.abs(result - expected)))
            rows.append(
                dict(
                    dtype=dtype,
                    shape=[m, n, active_k],
                    max_abs_error=error,
                    passed=error == 0,
                    image_sha256=hashlib.sha256(image).hexdigest(),
                    tile_source_sha256=hashlib.sha256(_source(case).encode()).hexdigest(),
                )
            )
        call("hipModuleUnload", [P], module)
    packet = dict(
        chip="gfx1201",
        board="RX 9070 XT",
        compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
        rows=rows,
        scope="one native WMMA tile, externally zero-padded ragged inputs; no arbitrary-size GEMM or performance promotion",
    )
    output.write_text(json.dumps(packet, indent=2) + "\n")
    if not all(r["passed"] for r in rows):
        raise RuntimeError("matrix numerical comparison failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    record(p.parse_args().output)
