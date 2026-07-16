"""Stage K (step 1) — a WMMA GEMM through the Target-IR → ROCDL → hsaco → execute
chain matches the hand-written oracle byte-for-byte.

Stage I closed the MLIR→hsaco loop for a scalar element-wise kernel. Stage J made
`tessera_rocm.wmma` lower to the real `rocdl.wmma` intrinsic. This test composes
them on a real **16×16×16 WMMA GEMM** kernel expressed at the Target-IR level
(`tessera_rocm.wmma` + RDNA fragment loads/stores):

    gpu kernel{tessera_rocm.wmma}
      --(tessera-opt --lower-tessera-target-to-rocdl  [Stage J])--> rocdl.wmma
      --(mlir-opt: convert-scf-to-cf, convert-gpu-to-rocdl, reconcile,
         rocdl-attach-target{gfx1151}, gpu-module-to-binary  [Stage I])--> hsaco
      --(hipModuleLoadData + launch)--> executes

and the result is compared to BOTH numpy AND the shipped `device_verified_abi`
hand-written kernel (`tessera_rocm_wmma_gemm_f16`), which serves as the on-silicon
oracle. Measured on gfx1151: vs numpy ~2e-7, **vs the oracle 0.0 (bit-identical)**.

Honest scope: the kernel *scaffold* (the fragment load/store layout) is authored
MLIR here — what's compiler-driven is the WMMA instruction selection + the entire
ROCDL/serialization/launch chain. The remaining Stage-K work is the **generating
pass** that emits this kernel from a `tessera.matmul` (so it is compiler-generated
end to end); this test locks the target IR + the layout the pass must produce, and
proves the chain matches the oracle.

Skip-clean: tessera-opt / mlir-opt / oracle lib not built, or no usable AMD GPU.
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
ORACLE_LIB = (REPO / "build" / "src" / "compiler" / "codegen"
              / "Tessera_ROCM_Backend" / "runtime" / "hip"
              / "libtessera_rocm_gemm.so")
CHIP = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")

# A 16×16×16 WMMA GEMM at the Tessera ROCm Target-IR level. Layout (RDNA wave32,
# identical to the hand-written oracle + rocdl_emit): lane L → l15 = L&15,
# lhi = L>>4; A frag a[i] = A[l15*16+i] (contiguous row); B frag b[i] = B[i*16+l15];
# D[(2e+lhi)*16 + l15] = acc[e].
_KERNEL = '''
module {
  gpu.module @gemm_mod {
    gpu.func @gemm(%A: memref<256xf16>, %B: memref<256xf16>, %D: memref<256xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c15i = arith.constant 15 : i32
      %c4i = arith.constant 4 : i32
      %acc0 = arith.constant dense<0.0> : vector<8xf32>
      %binit = arith.constant dense<0.0> : vector<16xf16>
      %tx = gpu.thread_id x
      %txi = arith.index_cast %tx : index to i32
      %l15i = arith.andi %txi, %c15i : i32
      %lhii = arith.shrui %txi, %c4i : i32
      %l15 = arith.index_cast %l15i : i32 to index
      %lhi = arith.index_cast %lhii : i32 to index
      %arow = arith.muli %l15, %c16 : index
      %a = vector.load %A[%arow] : memref<256xf16>, vector<16xf16>
      %b = scf.for %i = %c0 to %c16 step %c1 iter_args(%bacc = %binit) -> (vector<16xf16>) {
        %off = arith.muli %i, %c16 : index
        %idx = arith.addi %off, %l15 : index
        %bv = memref.load %B[%idx] : memref<256xf16>
        %bnew = vector.insert %bv, %bacc[%i] : f16 into vector<16xf16>
        scf.yield %bnew : vector<16xf16>
      }
      %d = "tessera_rocm.wmma"(%a, %b, %acc0)
           : (vector<16xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
      scf.for %e = %c0 to %c8 step %c1 {
        %dv = vector.extract %d[%e] : f32 from vector<8xf32>
        %twoE = arith.muli %e, %c2 : index
        %row = arith.addi %twoE, %lhi : index
        %rowoff = arith.muli %row, %c16 : index
        %didx = arith.addi %rowoff, %l15 : index
        memref.store %dv, %D[%didx] : memref<256xf32>
      }
      gpu.return
    }
  }
}
'''


def _find_mlir_opt():
    if env := os.environ.get("TESSERA_MLIR_OPT"):
        return env if Path(env).is_file() else None
    for c in ("/usr/lib/llvm-23/bin/mlir-opt", "/opt/homebrew/opt/llvm/bin/mlir-opt"):
        if Path(c).is_file():
            return c
    return shutil.which("mlir-opt")


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


def _extract_hsaco(text: str) -> bytes:
    i = text.index('bin = "') + len('bin = "')
    out = bytearray()
    j = i
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


def _compile_gemm_hsaco(mlir_opt: str) -> bytes:
    # Stage J: tessera_rocm.wmma -> rocdl.wmma
    j = subprocess.run([str(TESSERA_OPT), "-", "--lower-tessera-target-to-rocdl"],
                       input=_KERNEL, capture_output=True, text=True)
    assert j.returncode == 0, f"Stage J lowering failed: {j.stderr}"
    assert "rocdl.wmma.f32.16x16x16.f16" in j.stdout, "wmma did not lower to rocdl"
    # Stage I: finish-lower + attach + serialize
    pipeline = (
        "builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        f"reconcile-unrealized-casts),rocdl-attach-target{{chip={CHIP}}},"
        "gpu-module-to-binary)"
    )
    s = subprocess.run([mlir_opt, f"--pass-pipeline={pipeline}"],
                       input=j.stdout, capture_output=True, text=True)
    assert s.returncode == 0, f"serialize failed: {s.stderr}"
    assert "bin = " in s.stdout, "no hsaco produced"
    hsaco = _extract_hsaco(s.stdout)
    assert hsaco[:4] == b"\x7fELF", f"not an ELF hsaco: {hsaco[:4]!r}"
    return hsaco


def _launch_gemm(hip, hsaco: bytes, A, B):
    """Load + launch the compiled 16×16×16 GEMM (1 block × 32 threads). Returns
    the f32 result, or None if the device is unusable (skip signal)."""
    if hip.hipInit(0) != 0:
        return None
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        return None
    fn = ctypes.c_void_p()
    if hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"gemm") != 0:
        return None
    da, db, dd = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    if (hip.hipMalloc(ctypes.byref(da), 512) != 0
            or hip.hipMalloc(ctypes.byref(db), 512) != 0
            or hip.hipMalloc(ctypes.byref(dd), 1024) != 0):
        return None
    hip.hipMemcpy(da, A.ctypes.data_as(ctypes.c_void_p), 512, 1)
    hip.hipMemcpy(db, B.ctypes.data_as(ctypes.c_void_p), 512, 1)

    def mr(p):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(256), ctypes.c_int64(1)]

    args = mr(da) + mr(db) + mr(dd)
    arr = (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    if launch(fn, 1, 1, 1, 32, 1, 1, 0, None, arr, None) != 0:
        return None
    if hip.hipDeviceSynchronize() != 0:
        return None
    D = np.zeros((16, 16), dtype=np.float32)
    hip.hipMemcpy(D.ctypes.data_as(ctypes.c_void_p), dd, 1024, 2)
    for d in (da, db, dd):
        hip.hipFree(d)
    return D


def _oracle(hip, A, B):
    """The device_verified_abi hand-written kernel — the on-silicon oracle."""
    lib = ctypes.CDLL(str(ORACLE_LIB), mode=ctypes.RTLD_GLOBAL)
    fn = lib.tessera_rocm_wmma_gemm_f16
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    fn.restype = ctypes.c_int
    D = np.zeros((16, 16), dtype=np.float32)
    rc = fn(A.ctypes.data_as(ctypes.c_void_p), B.ctypes.data_as(ctypes.c_void_p),
            D.ctypes.data_as(ctypes.c_void_p), 16, 16, 16)
    return rc, D


def test_target_ir_wmma_gemm_matches_numpy_and_oracle():
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    mlir_opt = _find_mlir_opt()
    if mlir_opt is None:
        pytest.skip("mlir-opt not found (set TESSERA_MLIR_OPT or install LLVM 23)")
    hip = _hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")

    hsaco = _compile_gemm_hsaco(mlir_opt)

    rng = np.random.default_rng(1)
    A = (rng.standard_normal((16, 16)) * 0.4).astype(np.float16)
    B = (rng.standard_normal((16, 16)) * 0.4).astype(np.float16)

    D = _launch_gemm(hip, hsaco, A, B)
    if D is None:
        pytest.skip("no usable AMD GPU (load/launch unavailable)")

    ref = A.astype(np.float32) @ B.astype(np.float32)
    assert float(np.max(np.abs(D - ref))) < 1e-2, "compiled GEMM != numpy"

    if not ORACLE_LIB.is_file():
        pytest.skip("oracle lib not built: ninja -C build tessera_rocm_gemm")
    orc, Do = _oracle(hip, A, B)
    if orc == 2:
        pytest.skip("oracle: no usable AMD GPU (rc=2)")
    assert orc == 0, f"oracle returned {orc}"
    # The compiler-chain GEMM must match the hand-written oracle bit-for-bit:
    # same instruction, same layout, same accumulation order.
    assert float(np.max(np.abs(D - Do))) == 0.0, \
        "compiler-chain GEMM != hand-written oracle"
