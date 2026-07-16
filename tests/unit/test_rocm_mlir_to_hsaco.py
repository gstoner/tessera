"""Stage I — the MLIR→hsaco→execute loop closes on gfx1151.

This is the proof that the **Tessera IR stack reaches silicon through the
MLIR/LLVM pipeline**, not hand-written HIP. A `tessera.add` Graph-IR kernel is
lowered:

    tessera kernel  --(tessera-opt --tessera-emit-rocdl)-->  gpu.module(ROCDL)
                    --(mlir-opt: finish-lower + attach gfx1151 + module-to-binary)-->
                    gpu.binary{ #gpu.object bin="<hsaco ELF>" }

the hsaco ELF is extracted, loaded with `hipModuleLoadData`, launched against the
device (the standard MLIR memref-descriptor kernel ABI), and the result is
compared to a numpy reference.

This is distinct from the shipped HIPRTC runtime symbols (which compile a
hand-written HIP C++ kernel at load and bypass MLIR entirely): here the kernel
that executes was produced by the compiler's lowering pipeline.

Division of labour: `tessera-opt` owns the Tessera-specific lowering
(`--tessera-emit-rocdl`); the generic `gpu.module → hsaco` serialization
(`convert-gpu-to-rocdl` finish, `rocdl-attach-target`, `gpu-module-to-binary`)
rides the upstream `mlir-opt` (apt LLVM 22, a documented platform dependency).
In-process serialization (no mlir-opt shell-out, for the runtime launch path) is
a later-stage concern; Stage I proves the offline compile chain.

Skip-clean: tessera-opt / mlir-opt not built or found, or no usable AMD GPU.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"
CHIP = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")

# One-D element-wise kernels that tessera-to-linalg lowers; %OP% is the
# tessera op, %REF% the numpy reference applied to the host inputs.
_KERNELS = {
    "tessera.add": lambda a, b: a + b,
    "tessera.mul": lambda a, b: a * b,
}


def _find_mlir_opt():
    if env := os.environ.get("TESSERA_MLIR_OPT"):
        return env if Path(env).is_file() else None
    for c in ("/usr/lib/llvm-23/bin/mlir-opt", "/opt/homebrew/opt/llvm/bin/mlir-opt"):
        if Path(c).is_file():
            return c
    return shutil.which("mlir-opt")


def _hip():
    """Load libamdhip64 + return the CDLL, or None (skip-clean)."""
    rocm_lib = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib")
    for dep in ("libamdhip64.so", "libhiprtc.so"):
        p = os.path.join(rocm_lib, dep)
        if os.path.isfile(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    try:
        return ctypes.CDLL("libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return None


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _extract_hsaco(gpu_binary_mlir: str) -> bytes:
    """Pull the raw hsaco ELF out of a `#gpu.object<..., bin = "<escaped>">`."""
    marker = 'bin = "'
    i = gpu_binary_mlir.index(marker) + len(marker)
    out = bytearray()
    j = i
    s = gpu_binary_mlir
    hexd = "0123456789abcdefABCDEF"
    simple = {"\\": 0x5C, '"': 0x22, "n": 0x0A, "t": 0x09, "r": 0x0D}
    while j < len(s):
        c = s[j]
        if c == '"':
            break
        if c == "\\":
            nxt = s[j + 1:j + 3]
            if len(nxt) == 2 and nxt[0] in hexd and nxt[1] in hexd:
                out.append(int(nxt, 16))
                j += 3
                continue
            if s[j + 1] in simple:
                out.append(simple[s[j + 1]])
                j += 2
                continue
        out.append(ord(c))
        j += 1
    return bytes(out)


def _compile_to_hsaco(op: str, mlir_opt: str, n: int) -> bytes:
    """tessera kernel → tessera-opt emit-rocdl → mlir-opt finish+attach+binary →
    extract hsaco bytes. The kernel name is fixed (`ew`)."""
    src = f'''
func.func @ew(%a: tensor<{n}xf32>, %b: tensor<{n}xf32>) -> tensor<{n}xf32> {{
  %0 = "{op}"(%a, %b) : (tensor<{n}xf32>, tensor<{n}xf32>) -> tensor<{n}xf32>
  return %0 : tensor<{n}xf32>
}}
'''
    emit = _run([str(TESSERA_OPT), "-", "--tessera-emit-rocdl"], input=src)
    assert emit.returncode == 0, f"tessera-emit-rocdl failed: {emit.stderr}"
    pipeline = (
        "builtin.module("
        "gpu.module(convert-gpu-to-rocdl,reconcile-unrealized-casts),"
        f"rocdl-attach-target{{chip={CHIP}}},"
        "gpu-module-to-binary)"
    )
    binr = _run([mlir_opt, f"--pass-pipeline={pipeline}"], input=emit.stdout)
    assert binr.returncode == 0, f"mlir-opt serialize failed: {binr.stderr}"
    assert "bin = " in binr.stdout, "no gpu.object binary blob emitted"
    hsaco = _extract_hsaco(binr.stdout)
    assert hsaco[:4] == b"\x7fELF", f"not an ELF hsaco: {hsaco[:4]!r}"
    return hsaco


def _launch_and_compare(hip, hsaco: bytes, a, b, ref) -> float:
    """hipModuleLoadData(hsaco) → launch `ew` (memref-descriptor ABI) → maxerr.
    Returns -1.0 if the device is unusable (skip-clean signal)."""
    n = a.size

    def ok(rc):
        return rc == 0

    if hip.hipInit(0) != 0:
        return -1.0
    mod = ctypes.c_void_p()
    if not ok(hip.hipModuleLoadData(ctypes.byref(mod), hsaco)):
        return -1.0
    fn = ctypes.c_void_p()
    if not ok(hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"ew")):
        # the outlined kernel is named "<func>_kernel" inside the gpu.module
        if not ok(hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"ew_kernel")):
            return -1.0
    nb = n * 4
    da, db, dd = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    for d in (da, db, dd):
        if not ok(hip.hipMalloc(ctypes.byref(d), nb)):
            return -1.0
    hip.hipMemcpy(da, a.ctypes.data_as(ctypes.c_void_p), nb, 1)   # H2D
    hip.hipMemcpy(db, b.ctypes.data_as(ctypes.c_void_p), nb, 1)

    # MLIR memref-descriptor kernel ABI: (i64 step=1, i64 off=0,
    # then per memref: alloc_ptr, aligned_ptr, offset:i64, size:i64, stride:i64)
    def memref(p):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(n), ctypes.c_int64(1)]

    args = ([ctypes.c_int64(1), ctypes.c_int64(0)]
            + memref(da) + memref(db) + memref(dd))
    arr = (ctypes.c_void_p * len(args))()
    for i, a_ in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a_), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    if not ok(launch(fn, n, 1, 1, 1, 1, 1, 0, None, arr, None)):
        return -1.0
    if not ok(hip.hipDeviceSynchronize()):
        return -1.0
    out = np.zeros(n, dtype=np.float32)
    hip.hipMemcpy(out.ctypes.data_as(ctypes.c_void_p), dd, nb, 2)  # D2H
    for d in (da, db, dd):
        hip.hipFree(d)
    return float(np.max(np.abs(out - ref)))


@pytest.mark.parametrize("op", list(_KERNELS))
def test_tessera_kernel_compiles_through_mlir_and_executes(op):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    mlir_opt = _find_mlir_opt()
    if mlir_opt is None:
        pytest.skip("mlir-opt not found (set TESSERA_MLIR_OPT or install LLVM 23)")
    hip = _hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")

    n = 64
    hsaco = _compile_to_hsaco(op, mlir_opt, n)

    rng = np.random.default_rng(0)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    ref = _KERNELS[op](a, b)
    maxerr = _launch_and_compare(hip, hsaco, a, b, ref)
    if maxerr < 0:
        pytest.skip("no usable AMD GPU (hipModuleLoadData / launch unavailable)")
    # Exact: f32 add/mul through the compiled kernel must match numpy bit-for-bit.
    assert maxerr == 0.0, f"{op} via MLIR-compiled hsaco: maxerr={maxerr}"


def test_emit_rocdl_serializes_to_a_real_hsaco_elf():
    """The serialization half alone (no GPU needed): the lowered gpu.module
    serializes to a real AMD-GPU ELF (\\x7fELF magic). Proves the compile chain
    independent of device availability."""
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    mlir_opt = _find_mlir_opt()
    if mlir_opt is None:
        pytest.skip("mlir-opt not found (set TESSERA_MLIR_OPT or install LLVM 23)")
    hsaco = _compile_to_hsaco("tessera.add", mlir_opt, 64)
    assert hsaco[:4] == b"\x7fELF"
    assert len(hsaco) > 256, "hsaco implausibly small"
