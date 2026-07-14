"""Stage L1 — the compiler-generated WMMA GEMM is *problem-size-generic*.

Stage K proved a single 16x16x16 tile, compiler-generated and bit-identical to
the hand-written oracle. L1 generalizes the emitted kernel: the `tessera_rocm.
wmma_gemm` directive still carries the WMMA *instruction* tile (16x16x16), but
`generate-wmma-gemm-kernel` now emits a kernel that takes the runtime `(M,N,K)`
as `index` arguments, runs a 2-D grid of one wave per 16x16 output tile, an
`scf.for` K-loop, and ragged-edge masking — so ONE compiled kernel computes any
shape:

    "tessera_rocm.wmma_gemm"{m=n=k=16}
      --(generate-wmma-gemm-kernel)--> gpu.func(A,B,D:memref<?>, M,N,K:index)
                                       { grid + scf.for K-loop + masked wmma }
      --(--lower-tessera-target-to-rocdl  [Stage J])--> rocdl.wmma
      --(mlir-opt finish-lower + attach{gfx1151} + gpu-module-to-binary)--> hsaco
      --(launch grid=(ceilN/16, ceilM/16))--> executes for arbitrary (M,N,K)

Each shape is compared to BOTH numpy and the shipped `device_verified_abi`
hand-written kernel (`tessera_rocm_wmma_gemm_f16`, the on-silicon oracle, itself
general-shape). The compiler kernel uses the identical instruction, fragment
layout, accumulation order, and masking, so it matches the oracle bit-for-bit
(0.0) including on ragged (non-multiple-of-16) shapes.

`test_rocm_wmma_gemm_generated.py` is the single-tile companion (the Stage K
milestone). Skip-clean: tools / oracle lib not built, or no usable AMD GPU.
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

def _directive(mt=1, nt=1):
    return (
        'module {\n'
        '  "tessera_rocm.wmma_gemm"() {name = "gemm", m = 16 : i64, '
        'n = 16 : i64, k = 16 : i64, '
        f'mt = {mt} : i64, nt = {nt} : i64}} : () -> ()\n'
        '}\n'
    )


# (M, N, K): square-aligned, rectangular multi-tile, and ragged edges — incl.
# ragged K (K%16!=0: 31, 40) which exercises the aligned-main-loop + masked-tail
# split, and a multi-tile ragged-K case (96x80x40).
_SHAPES = [(16, 16, 16), (32, 32, 32), (48, 64, 32), (40, 24, 48),
           (17, 15, 31), (96, 80, 40)]
# (mt, nt) register-blocked macro-tiles: 1x1 (L1), small (2x4), oracle-best 3x4.
_MACRO_TILES = [(1, 1), (2, 2), (2, 4), (3, 4)]


def _find_mlir_opt():
    if env := os.environ.get("TESSERA_MLIR_OPT"):
        return env if Path(env).is_file() else None
    for c in ("/usr/lib/llvm-22/bin/mlir-opt", "/opt/homebrew/opt/llvm/bin/mlir-opt"):
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


def _build_hsaco(mt=1, nt=1) -> bytes:
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    mlir_opt = _find_mlir_opt()
    if mlir_opt is None:
        pytest.skip("mlir-opt not found (set TESSERA_MLIR_OPT or install LLVM 22)")
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-wmma-gemm-kernel",
         "--lower-tessera-target-to-rocdl"],
        input=_directive(mt, nt), capture_output=True, text=True)
    assert gen.returncode == 0, gen.stderr
    assert "rocdl.wmma.f32.16x16x16.f16" in gen.stdout
    pipeline = (
        "builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        f"reconcile-unrealized-casts),rocdl-attach-target{{chip={CHIP}}},"
        "gpu-module-to-binary)"
    )
    ser = subprocess.run([mlir_opt, f"--pass-pipeline={pipeline}"],
                         input=gen.stdout, capture_output=True, text=True)
    assert ser.returncode == 0, ser.stderr
    hsaco = _extract_hsaco(ser.stdout)
    assert hsaco[:4] == b"\x7fELF"
    return hsaco


def _mr(p, size):
    return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
            ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int64(1)]


def _launch(hip, fn, A, B, M, N, K, mt=1, nt=1):
    """Run the compiler-generated kernel; returns the MxN f32 result."""
    da, db, dd = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    nbytes = (2 * M * K, 2 * K * N, 4 * M * N)
    for d, nb in ((da, nbytes[0]), (db, nbytes[1]), (dd, nbytes[2])):
        if hip.hipMalloc(ctypes.byref(d), nb) != 0:
            pytest.skip("hipMalloc failed")
    hip.hipMemcpy(da, A.ctypes.data_as(ctypes.c_void_p), nbytes[0], 1)
    hip.hipMemcpy(db, B.ctypes.data_as(ctypes.c_void_p), nbytes[1], 1)
    args = (_mr(da, M * K) + _mr(db, K * N) + _mr(dd, M * N)
            + [ctypes.c_int64(M), ctypes.c_int64(N), ctypes.c_int64(K)])
    arr = (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    # One wave per mt x nt macro-tile.
    gx, gy = (N + 16 * nt - 1) // (16 * nt), (M + 16 * mt - 1) // (16 * mt)
    assert launch(fn, gx, gy, 1, 32, 1, 1, 0, None, arr, None) == 0
    assert hip.hipDeviceSynchronize() == 0
    D = np.zeros((M, N), dtype=np.float32)
    hip.hipMemcpy(D.ctypes.data_as(ctypes.c_void_p), dd, nbytes[2], 2)
    for d in (da, db, dd):
        hip.hipFree(d)
    return D


def _oracle(A, B, M, N, K):
    if not ORACLE_LIB.is_file():
        pytest.skip("oracle lib not built: ninja -C build tessera_rocm_gemm")
    lib = ctypes.CDLL(str(ORACLE_LIB), mode=ctypes.RTLD_GLOBAL)
    ofn = lib.tessera_rocm_wmma_gemm_f16
    ofn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    ofn.restype = ctypes.c_int
    Do = np.zeros((M, N), dtype=np.float32)
    rc = ofn(A.ctypes.data_as(ctypes.c_void_p), B.ctypes.data_as(ctypes.c_void_p),
             Do.ctypes.data_as(ctypes.c_void_p), M, N, K)
    if rc == 2:
        pytest.skip("oracle: no usable AMD GPU")
    assert rc == 0
    return Do


@pytest.mark.parametrize("M,N,K", _SHAPES)
def test_general_shape_matches_numpy_and_oracle(M, N, K):
    hsaco = _build_hsaco()
    hip = _hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    if hip.hipInit(0) != 0:
        pytest.skip("hipInit failed")
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        pytest.skip("no usable AMD GPU (module load failed)")
    fn = ctypes.c_void_p()
    assert hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"gemm") == 0

    rng = np.random.default_rng(11 + M + N + K)
    A = (rng.standard_normal((M, K)) * 0.4).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.4).astype(np.float16)

    D = _launch(hip, fn, A, B, M, N, K)

    ref = A.astype(np.float32) @ B.astype(np.float32)
    assert float(np.max(np.abs(D - ref))) < 5e-2, f"vs numpy at {M}x{N}x{K}"

    Do = _oracle(A, B, M, N, K)
    # The compiler-generated general-shape kernel matches the hand-written
    # oracle bit-for-bit, including on ragged (non-multiple-of-16) edges.
    assert float(np.max(np.abs(D - Do))) == 0.0, \
        f"compiler-generated GEMM != hand-written oracle at {M}x{N}x{K}"


@pytest.mark.parametrize("mt,nt", _MACRO_TILES)
def test_register_blocked_matches_oracle(mt, nt):
    """Stage L2 — each wave computes an mt x nt grid of 16x16 output tiles
    (register blocking, fragment reuse). Register blocking changes only data
    reuse, not the per-output accumulation order, so the result stays
    bit-identical to the hand-written oracle. Shape spans full + ragged
    macro-tiles to exercise the grid math and edge masking."""
    hsaco = _build_hsaco(mt, nt)
    hip = _hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    if hip.hipInit(0) != 0:
        pytest.skip("hipInit failed")
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        pytest.skip("no usable AMD GPU (module load failed)")
    fn = ctypes.c_void_p()
    assert hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"gemm") == 0

    # A spread that is not a clean multiple of any macro-tile -> ragged edges.
    M, N, K = 100, 96, 64
    rng = np.random.default_rng(23 + mt * 7 + nt)
    A = (rng.standard_normal((M, K)) * 0.4).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.4).astype(np.float16)

    D = _launch(hip, fn, A, B, M, N, K, mt, nt)
    ref = A.astype(np.float32) @ B.astype(np.float32)
    assert float(np.max(np.abs(D - ref))) < 5e-2, f"vs numpy at {mt}x{nt}"

    Do = _oracle(A, B, M, N, K)
    assert float(np.max(np.abs(D - Do))) == 0.0, \
        f"register-blocked {mt}x{nt} GEMM != hand-written oracle"
