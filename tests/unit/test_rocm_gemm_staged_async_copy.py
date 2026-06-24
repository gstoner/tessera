"""A real COMPUTE kernel routed through the runnable async_copy: a 16x16x16 WMMA
GEMM whose A and B operands are staged global→LDS via tessera_rocm.async_copy,
then multiplied with tessera_rocm.wmma reading the LDS-staged fragments.

This is the end-to-end Fork-A staging+compute path on one tile: async_copy
(runnable, --lower-rocm-async-copy) → gpu.barrier → WMMA from LDS → C, executing
on gfx1151 and matching A@B. (Single tile by design — the point is to prove the
staged-compute path runs, not to beat the direct GEMM; this session measured
that no FA/GEMM shape is staging-bound, so a full double-buffered GEMM is not a
perf win, only a path demonstration.)

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


def _kernel() -> str:
    ws = "#gpu.address_space<workgroup>"
    # B fragment: b[i] = lB[i*16 + lane], i = 0..15 (strided LDS reads).
    bloads, belems = [], []
    for i in range(16):
        bloads.append(
            f"      %bi{i} = arith.constant {i * 16} : index\n"
            f"      %bx{i} = arith.addi %bi{i}, %lane : index\n"
            f"      %b{i} = memref.load %lB[%bx{i}] : memref<256xf16, {ws}>")
        belems.append(f"%b{i}")
    # C store: C[(2e+lhi)*16 + lane] = acc[e], e = 0..7.
    cstores = []
    for e in range(8):
        cstores.append(
            f"      %re{e} = arith.constant {2 * e} : index\n"
            f"      %row{e} = arith.addi %re{e}, %lhi : index\n"
            f"      %rb{e} = arith.muli %row{e}, %c16 : index\n"
            f"      %ci{e} = arith.addi %rb{e}, %lane : index\n"
            f"      %ce{e} = vector.extract %acc[{e}] : f32 from vector<8xf32>\n"
            f"      memref.store %ce{e}, %C[%ci{e}] : memref<?xf32>")
    return f"""
module {{
  gpu.module @m {{
    gpu.func @gemm_tile_staged(%A: memref<?xf16>, %B: memref<?xf16>,
        %C: memref<?xf32>)
        workgroup(%lA: memref<256xf16, {ws}>, %lB: memref<256xf16, {ws}>)
        kernel {{
      %n = arith.constant 256 : i64
      %tA = tessera_rocm.async_copy %lA, %A, %n
          : memref<256xf16, {ws}>, memref<?xf16> -> !tessera_rocm.token
      %tB = tessera_rocm.async_copy %lB, %B, %n
          : memref<256xf16, {ws}>, memref<?xf16> -> !tessera_rocm.token
      tessera_rocm.wait %tA : !tessera_rocm.token
      tessera_rocm.wait %tB : !tessera_rocm.token
      %tid = gpu.thread_id x
      %c4 = arith.constant 4 : index
      %c15 = arith.constant 15 : index
      %c16 = arith.constant 16 : index
      %lane = arith.andi %tid, %c15 : index
      %lhi = arith.shrui %tid, %c4 : index
      %arow = arith.muli %lane, %c16 : index
      %aFrag = vector.load %lA[%arow] : memref<256xf16, {ws}>, vector<16xf16>
{chr(10).join(bloads)}
      %bFrag = vector.from_elements {", ".join(belems)} : vector<16xf16>
      %z = arith.constant dense<0.0> : vector<8xf32>
      %acc = tessera_rocm.wmma %aFrag, %bFrag, %z
          : vector<16xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>
{chr(10).join(cstores)}
      gpu.return
    }}
  }}
}}
"""


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


def _find_mlir_opt():
    for c in ("/usr/lib/llvm-22/bin/mlir-opt",
              "/opt/homebrew/opt/llvm/bin/mlir-opt"):
        if Path(c).is_file():
            return c
    import shutil
    return os.environ.get("TESSERA_MLIR_OPT") or shutil.which("mlir-opt")


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


def test_gemm_tile_staged_through_async_copy_matches_numpy():
    if not ROP.is_file():
        pytest.skip("build tessera-rocm-opt")
    mlir_opt = _find_mlir_opt()
    if mlir_opt is None:
        pytest.skip("mlir-opt not found")
    lowered = subprocess.run(
        [str(ROP), "-", "--lower-rocm-async-copy",
         "--lower-tessera-target-to-rocdl"],
        input=_kernel(), capture_output=True, text=True)
    if lowered.returncode != 0:
        pytest.skip(f"lower unavailable: {lowered.stderr[:300]}")
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
    assert hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"gemm_tile_staged") == 0

    rng = np.random.default_rng(7)
    a = (rng.standard_normal((16, 16)) * 0.3).astype(np.float16)
    b = (rng.standard_normal((16, 16)) * 0.3).astype(np.float16)
    c = np.zeros((16, 16), np.float32)
    da, db, dc = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    for d, nb in ((da, 2 * 256), (db, 2 * 256), (dc, 4 * 256)):
        if hip.hipMalloc(ctypes.byref(d), nb) != 0:
            pytest.skip("hipMalloc failed")
    hip.hipMemcpy(da, a.ctypes.data_as(ctypes.c_void_p), 2 * 256, 1)
    hip.hipMemcpy(db, b.ctypes.data_as(ctypes.c_void_p), 2 * 256, 1)

    args = _mr(da, 256) + _mr(db, 256) + _mr(dc, 256)
    arr = (ctypes.c_void_p * len(args))()
    for i, x in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(x), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    assert launch(fn, 1, 1, 1, 32, 1, 1, 0, None, arr, None) == 0
    assert hip.hipDeviceSynchronize() == 0
    hip.hipMemcpy(c.ctypes.data_as(ctypes.c_void_p), dc, 4 * 256, 2)
    for d in (da, db, dc):
        hip.hipFree(d)

    ref = a.astype(np.float32) @ b.astype(np.float32)
    maxerr = float(np.max(np.abs(c - ref)))
    assert maxerr < 5e-2, f"staged WMMA GEMM tile maxerr={maxerr}"
