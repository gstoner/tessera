"""Native GPU-IR rounding microprobe; no Graph/Schedule coverage claim."""
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
from tessera import runtime as rt
from tessera.compiler.native_gpu_storage import _run, _binary_pass, _decode_image, _reject_bodyless_image

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
source = (folder / "rounding-probe.mlir").read_text()
source = source.replace("module {", "module attributes {gpu.container_module} {", 1)
assert rt._rocm_live_arch() == "gfx1201"
compiler, llvm = Path(os.environ["TESSERA_OPT"]), Path(os.environ["LLVM_BIN"])
verified = _run(compiler, "--verify-each", source=source)
pipeline = ("builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
            "convert-math-to-llvm,reconcile-unrealized-casts),"
            "rocdl-attach-target{chip=gfx1201}," + _binary_pass(None) + ")")
binary = _run(llvm / "mlir-opt", "--pass-pipeline=" + pipeline, source=verified)
encoded = re.search(r'bin = "((?:\\.|[^"\\])*)"', binary)
literal = encoded[1] if encoded else re.findall(r'"((?:\\.|[^"\\])*)"', binary)[-1]
image = _decode_image(literal)
_reject_bodyless_image(image, "rocm", "scale_rounding", llvm)
tiny, normal = math.ldexp(1., -149), math.ldexp(1., -126)
maximum = float(np.finfo(np.float32).max)
overflow = math.ldexp(1., 128) - math.ldexp(1., 103)
cases = [(1., 0.), (1., -0.), (-0., 1.), (1., tiny), (.5, tiny),
         (1.5, tiny), (2.5, tiny), (3.5, tiny), (1., normal),
         (1., normal-math.ldexp(1., -150)), (1., normal-3*math.ldexp(1., -150)),
         (1., maximum), (1., np.nextafter(overflow, 0.)), (1., overflow),
         (1., -tiny), (448., maximum/448.), (2.5, 1.+math.ldexp(1., -24)),
         (.5, math.ldexp(1., -150))]
cases = (cases * 4)[:64]
a = np.array([x for x, y in cases], np.float64)
b = np.array([y for x, y in cases], np.float64)
with np.errstate(over="ignore", under="ignore"):
    expected = (a*b).astype(np.float32).view(np.uint32)
output = np.full(64, 0xdeadbeef, np.uint32)
hip = rt._load_hip_for_launch()


def check(rc):
    if rc:
        raise RuntimeError("HIP rounding probe status " + str(rc))


check(hip.hipInit(0))
module, function = C.c_void_p(), C.c_void_p()
blob, pointers = C.create_string_buffer(image), []
check(hip.hipModuleLoadData(C.byref(module), blob))
try:
    check(hip.hipModuleGetFunction(C.byref(function), module, b"scale_rounding"))
    for host in (a, b, output):
        pointer = C.c_void_p()
        check(hip.hipMalloc(C.byref(pointer), host.nbytes))
        pointers.append(pointer)
        check(hip.hipMemcpy(pointer, C.c_void_p(host.ctypes.data), host.nbytes, 1))
    params = (C.c_void_p*3)(*[C.cast(C.byref(pointer), C.c_void_p) for pointer in pointers])
    check(hip.hipModuleLaunchKernel(function, 1, 1, 1, 64, 1, 1, 0, None, params, None))
    check(hip.hipDeviceSynchronize())
    check(hip.hipMemcpy(C.c_void_p(output.ctypes.data), pointers[-1], output.nbytes, 2))
finally:
    hip.hipDeviceSynchronize()
    for pointer in pointers:
        hip.hipFree(pointer)
    hip.hipModuleUnload(module)
receipt = dict(
    architecture=rt._rocm_live_arch(),
    scope="native GPU-IR rounding microprobe only; not Graph/Schedule or NVFP4 production execution",
    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
    mlir_opt_sha256=hashlib.sha256((llvm / "mlir-opt").read_bytes()).hexdigest(),
    image_sha256=hashlib.sha256(image).hexdigest(),
    cases=[dict(scale=x, global_value=y, expected_bits=int(e), native_bits=int(g))
           for (x, y), e, g in zip(cases, expected, output, strict=True)],
    mismatch_count=int(np.count_nonzero(expected != output)),
)
(folder / "rounding-probe.mlir").write_text(source)
(folder / "rounding-probe.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(mismatches=receipt["mismatch_count"], cases=len(cases))))
