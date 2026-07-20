"""The RUNNABLE ROCm async_copy lowering executes a global→LDS→global round-trip
on gfx1151.

`tessera_rocm.async_copy` previously lowered only to an artifact-only contract
marker (`llvm.amdgcn.raw.buffer.copy.contract`). The `--lower-rocm-async-copy`
pass lowers it to a real cooperative copy loop (global load → LDS store; on RDNA
there is no GLOBAL_LOAD_LDS DMA, confirmed from the ISA archive) and
`tessera_rocm.wait` → `gpu.barrier`. This test builds a kernel that async_copies
a global buffer into LDS, barriers, then writes the LDS tile back out, and checks
out == src on device — proving the staging path is executable (the runnable half
of the Fork-A pipeline; the markers stay for the IR-contract path).

Skip-clean: tessera-rocm-opt / mlir-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[2]
ROP = REPO / "build/src/compiler/codegen/Tessera_ROCM_Backend/tools/tessera-rocm-opt"
CHIP = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")

_KERNEL = """
module {
  gpu.module @m {
    gpu.func @copy_demo(%src: memref<?xf32>, %out: memref<?xf32>, %n: i64)
        workgroup(%lds: memref<256xf32, #gpu.address_space<workgroup>>) kernel {
      %tok = tessera_rocm.async_copy %lds, %src, %n
          : memref<256xf32, #gpu.address_space<workgroup>>, memref<?xf32>
            -> !tessera_rocm.token
      tessera_rocm.wait %tok : !tessera_rocm.token
      %tid = gpu.thread_id x
      %bdim = gpu.block_dim x
      %ni = arith.index_cast %n : i64 to index
      scf.for %i = %tid to %ni step %bdim {
        %v = memref.load %lds[%i]
            : memref<256xf32, #gpu.address_space<workgroup>>
        memref.store %v, %out[%i] : memref<?xf32>
      }
      gpu.return
    }
  }
}
"""


def _find_mlir_opt():
    if env := os.environ.get("TESSERA_MLIR_OPT"):
        return env if Path(env).is_file() else None
    for c in ("/usr/lib/llvm-23/bin/mlir-opt",
              "/opt/homebrew/opt/llvm@23/bin/mlir-opt"):
        if Path(c).is_file():
            return c
    import shutil
    return shutil.which("mlir-opt")


def _extract_hsaco(text):
    i = text.index('bin = "') + len('bin = "')
    out = bytearray(); j = i
    hexd = "0123456789abcdefABCDEF"
    simple = {"\\": 0x5C, '"': 0x22, "n": 0x0A, "t": 0x09, "r": 0x0D}
    while j < len(text):
        c = text[j]
        if c == '"':
            break
        if c == "\\":
            nx = text[j + 1:j + 3]
            if len(nx) == 2 and nx[0] in hexd and nx[1] in hexd:
                out.append(int(nx, 16)); j += 3; continue
            if text[j + 1] in simple:
                out.append(simple[text[j + 1]]); j += 2; continue
        out.append(ord(c)); j += 1
    return bytes(out)


def _hip():
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


def _mr(p, n):
    return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
            ctypes.c_int64(0), ctypes.c_int64(n), ctypes.c_int64(1)]


def test_async_copy_runnable_global_to_lds_roundtrip():
    if not ROP.is_file():
        pytest.skip("build tessera-rocm-opt")
    mlir_opt = _find_mlir_opt()
    if mlir_opt is None:
        pytest.skip("mlir-opt not found")
    lowered = subprocess.run([str(ROP), "-", "--lower-rocm-async-copy"],
                             input=_KERNEL, capture_output=True, text=True)
    assert lowered.returncode == 0, lowered.stderr
    assert "tessera_rocm.async_copy" not in lowered.stdout  # consumed
    pl = ("builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
          f"reconcile-unrealized-casts),rocdl-attach-target{{chip={CHIP}}},"
          "gpu-module-to-binary)")
    ser = subprocess.run([mlir_opt, f"--pass-pipeline={pl}"],
                         input=lowered.stdout, capture_output=True, text=True)
    if ser.returncode != 0 or "gpu.binary" not in ser.stdout:
        pytest.skip(f"serialize unavailable: {ser.stderr[:200]}")
    hsaco = _extract_hsaco(ser.stdout)
    assert hsaco[:4] == b"\x7fELF"

    hip = _hip()
    if hip is None or hip.hipInit(0) != 0:
        pytest.skip("no ROCm host")
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        pytest.skip("no usable AMD GPU")
    fn = ctypes.c_void_p()
    assert hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"copy_demo") == 0

    n = 200
    src = (np.random.default_rng(3).standard_normal(n) * 2).astype(np.float32)
    out = np.zeros(n, np.float32)
    dsrc, dout = ctypes.c_void_p(), ctypes.c_void_p()
    for d in (dsrc, dout):
        if hip.hipMalloc(ctypes.byref(d), 4 * n) != 0:
            pytest.skip("hipMalloc failed")
    hip.hipMemcpy(dsrc, src.ctypes.data_as(ctypes.c_void_p), 4 * n, 1)

    args = _mr(dsrc, n) + _mr(dout, n) + [ctypes.c_int64(n)]
    arr = (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    assert launch(fn, 1, 1, 1, 32, 1, 1, 0, None, arr, None) == 0
    assert hip.hipDeviceSynchronize() == 0
    hip.hipMemcpy(out.ctypes.data_as(ctypes.c_void_p), dout, 4 * n, 2)
    for d in (dsrc, dout):
        hip.hipFree(d)

    # The async_copy staged src into LDS; the read-back wrote it to out.
    assert np.array_equal(out, src), "global->LDS->global round-trip mismatch"
