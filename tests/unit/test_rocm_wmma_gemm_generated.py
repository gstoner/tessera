"""Stage K (step 2) — the compiler GENERATES the executing WMMA GEMM.

The milestone: "the Tessera compiler, not a hand-written kernel, produced the
executing GEMM." A `tessera_rocm.wmma_gemm` matmul directive is expanded by the
`generate-wmma-gemm-kernel` pass into a fragment-materialized RDNA WMMA kernel
(fragment loads + `tessera_rocm.wmma` + accumulator stores — the kernel BODY is
emitted by the pass, not authored), which then lowers through Stage J (real
`rocdl.wmma`) and Stage I (→ hsaco) and executes:

    "tessera_rocm.wmma_gemm"{m=16,n=16,k=16}
      --(tessera-opt --generate-wmma-gemm-kernel)--> gpu.func{frag loads + wmma + stores}
      --(--lower-tessera-target-to-rocdl  [Stage J])--> rocdl.wmma
      --(mlir-opt: convert-scf-to-cf, convert-gpu-to-rocdl, reconcile,
         rocdl-attach-target{gfx1151}, gpu-module-to-binary  [Stage I])--> hsaco
      --(hipModuleLoadData + launch)--> executes

The result is compared to BOTH numpy AND the shipped `hardware_verified`
hand-written kernel (`tessera_rocm_wmma_gemm_f16`), the on-silicon oracle. On
gfx1151: vs numpy ~2e-7, **vs the oracle 0.0 (bit-identical)** — same instruction,
layout, and accumulation order, but compiler-generated.

`test_rocm_wmma_gemm_via_mlir.py` is the companion that locks the target IR with a
hand-authored kernel; this test proves the pass reproduces it.

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

_DIRECTIVE = '''
module {
  "tessera_rocm.wmma_gemm"() {name = "gemm", m = 16 : i64, n = 16 : i64, k = 16 : i64} : () -> ()
}
'''


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


def _need_tools():
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    mo = _find_mlir_opt()
    if mo is None:
        pytest.skip("mlir-opt not found (set TESSERA_MLIR_OPT or install LLVM 22)")
    return mo


def test_generate_pass_emits_fragment_materialized_kernel():
    """No GPU needed: the pass turns the directive into a gpu.func whose body is
    the problem-size-generic, fragment-materialized WMMA kernel — a runtime
    (M,N,K) signature, an scf.for K-loop, masked fragment loads + tessera_rocm.
    wmma + masked stores."""
    _need_tools()
    r = subprocess.run([str(TESSERA_OPT), "-", "--generate-wmma-gemm-kernel"],
                       input=_DIRECTIVE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    # Problem-size-generic ABI: dynamic memrefs + runtime M,N,K index args.
    assert "gpu.func @gemm" in out and "kernel" in out
    assert "memref<?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf32>" in out
    assert "index, %arg4: index, %arg5: index" in out
    # Interior fast path / masked edge path split (Stage L2):
    #   - vector.load: the contiguous A fragment on the fast (unmasked) path
    #   - scf.if: the fastCond branch + the masked-path scf.if-guarded stores
    #   - scf.for: the K-loop appears in both paths
    assert "vector.load" in out                 # coalesced A fragment, fast path
    assert out.count("scf.for") == 2            # K-loop in fast + masked path
    assert "scf.if" in out                       # fast/masked split + edge stores
    assert "tessera_rocm.wmma" in out          # the matrix op, generated
    # For mt=nt=1: fast B (16 inserts) + masked A+B (32) = 48; stores 8 + 8 = 16.
    assert out.count("vector.insert") == 48
    assert out.count("memref.store") == 16
    assert '"tessera_rocm.wmma_gemm"' not in out   # directive consumed


def test_compiler_generated_gemm_matches_numpy_and_oracle():
    mlir_opt = _need_tools()
    hip = _hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")

    # directive -> generate kernel -> Stage J (real rocdl.wmma)
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-wmma-gemm-kernel",
         "--lower-tessera-target-to-rocdl"],
        input=_DIRECTIVE, capture_output=True, text=True)
    assert gen.returncode == 0, gen.stderr
    assert "rocdl.wmma.f32.16x16x16.f16" in gen.stdout
    # Stage I: finish-lower + attach + serialize
    pipeline = (
        "builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        f"reconcile-unrealized-casts),rocdl-attach-target{{chip={CHIP}}},"
        "gpu-module-to-binary)"
    )
    ser = subprocess.run([mlir_opt, f"--pass-pipeline={pipeline}"],
                         input=gen.stdout, capture_output=True, text=True)
    assert ser.returncode == 0, ser.stderr
    assert "bin = " in ser.stdout
    hsaco = _extract_hsaco(ser.stdout)
    assert hsaco[:4] == b"\x7fELF"

    rng = np.random.default_rng(7)
    A = (rng.standard_normal((16, 16)) * 0.4).astype(np.float16)
    B = (rng.standard_normal((16, 16)) * 0.4).astype(np.float16)

    if hip.hipInit(0) != 0:
        pytest.skip("hipInit failed")
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        pytest.skip("no usable AMD GPU (module load failed)")
    fn = ctypes.c_void_p()
    assert hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"gemm") == 0
    da, db, dd = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    for d, n in ((da, 512), (db, 512), (dd, 1024)):
        if hip.hipMalloc(ctypes.byref(d), n) != 0:
            pytest.skip("hipMalloc failed")
    hip.hipMemcpy(da, A.ctypes.data_as(ctypes.c_void_p), 512, 1)
    hip.hipMemcpy(db, B.ctypes.data_as(ctypes.c_void_p), 512, 1)

    # Problem-size-generic kernel ABI: 3 dynamic-memref descriptors (5 fields:
    # alloc, aligned, offset, size, stride) + runtime M,N,K as i64. The grid
    # is one wave per 16x16 output tile; M=N=K=16 -> grid (1,1,1).
    def mr(p, size):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int64(1)]

    M = N = Kd = 16
    args = (mr(da, M * Kd) + mr(db, Kd * N) + mr(dd, M * N)
            + [ctypes.c_int64(M), ctypes.c_int64(N), ctypes.c_int64(Kd)])
    arr = (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    gx, gy = (N + 15) // 16, (M + 15) // 16
    assert launch(fn, gx, gy, 1, 32, 1, 1, 0, None, arr, None) == 0
    assert hip.hipDeviceSynchronize() == 0
    D = np.zeros((16, 16), dtype=np.float32)
    hip.hipMemcpy(D.ctypes.data_as(ctypes.c_void_p), dd, 1024, 2)
    for d in (da, db, dd):
        hip.hipFree(d)

    assert float(np.max(np.abs(D - A.astype(np.float32) @ B.astype(np.float32)))) < 1e-2

    if not ORACLE_LIB.is_file():
        pytest.skip("oracle lib not built: ninja -C build tessera_rocm_gemm")
    lib = ctypes.CDLL(str(ORACLE_LIB), mode=ctypes.RTLD_GLOBAL)
    ofn = lib.tessera_rocm_wmma_gemm_f16
    ofn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    ofn.restype = ctypes.c_int
    Do = np.zeros((16, 16), dtype=np.float32)
    orc = ofn(A.ctypes.data_as(ctypes.c_void_p), B.ctypes.data_as(ctypes.c_void_p),
              Do.ctypes.data_as(ctypes.c_void_p), 16, 16, 16)
    if orc == 2:
        pytest.skip("oracle: no usable AMD GPU")
    assert orc == 0
    # The compiler-GENERATED GEMM matches the hand-written oracle bit-for-bit.
    assert float(np.max(np.abs(D - Do))) == 0.0, \
        "compiler-generated GEMM != hand-written oracle"


def test_generate_pass_rejects_non_16_tile():
    """Honest scope: only the 16x16x16 tile is implemented — other extents are a
    named error, not a silent wrong kernel."""
    _need_tools()
    bad = _DIRECTIVE.replace("k = 16", "k = 32")
    r = subprocess.run([str(TESSERA_OPT), "-", "--generate-wmma-gemm-kernel"],
                       input=bad, capture_output=True, text=True)
    assert r.returncode != 0
    assert "16x16x16" in r.stderr
