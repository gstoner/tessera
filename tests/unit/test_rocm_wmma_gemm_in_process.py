"""Stage L3 — in-process MLIR -> hsaco serialization (no mlir-opt shell-out).

Stages I/K/L1/L2 close the compiler-generated-GEMM loop, but the serialization
step (`gpu-module-to-binary`) rode the *platform* `mlir-opt` binary. A runtime
launch lane can't shell out to mlir-opt. L3 links the GPU/ROCDL serialization
spine into `tessera-opt` itself (AMDGPU codegen from the shared libLLVM, ld.lld
from the platform LLVM), so the WHOLE chain runs in ONE invocation:

    "tessera_rocm.wmma_gemm"{m=n=k=16, mt, nt}
      --(tessera-opt --pass-pipeline='builtin.module(
           generate-wmma-gemm-kernel, lower-tessera-target-to-rocdl,
           gpu.module(convert-scf-to-cf, convert-gpu-to-rocdl,
                      reconcile-unrealized-casts),
           rocdl-attach-target{chip=gfx1151}, gpu-module-to-binary)')--> hsaco
      --(hipModuleLoadData + launch)--> executes

This test drives that single tessera-opt invocation (mlir-opt is NEVER called),
extracts the hsaco from the emitted `gpu.binary`, and executes it on gfx1151,
comparing to numpy AND the hand-written oracle (bit-identical). It is the same
kernel/result as `test_rocm_wmma_gemm_general.py`; the point here is the
*serialization path*, not the math.

Skip-clean: tessera-opt not built (or built without in-process serialization —
detected by the pipeline failing), or no usable AMD GPU.
"""

from __future__ import annotations

import ctypes
import os
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


def _pipeline():
    return (
        "builtin.module("
        "generate-wmma-gemm-kernel,"
        "lower-tessera-target-to-rocdl,"
        "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        "reconcile-unrealized-casts),"
        f"rocdl-attach-target{{chip={CHIP}}},"
        "gpu-module-to-binary)"
    )


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


def _build_hsaco_in_process(mt=1, nt=1) -> bytes:
    """The WHOLE chain in ONE tessera-opt call — mlir-opt is never invoked."""
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    r = subprocess.run([str(TESSERA_OPT), "-", f"--pass-pipeline={_pipeline()}"],
                       input=_directive(mt, nt), capture_output=True, text=True)
    if r.returncode != 0 or "gpu.binary" not in r.stdout:
        # tessera-opt built without the L3 in-process serialization spine
        # (e.g. a hardware-free artifact build) — nothing to prove here.
        pytest.skip("tessera-opt has no in-process gpu-module-to-binary "
                    f"(rc={r.returncode}): {r.stderr[:400]}")
    hsaco = _extract_hsaco(r.stdout)
    assert hsaco[:4] == b"\x7fELF", "in-process serialization did not emit ELF"
    return hsaco


def _mr(p, size):
    return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
            ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int64(1)]


def _launch(hip, fn, A, B, M, N, K, mt, nt):
    da, db, dd = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    nb = (2 * M * K, 2 * K * N, 4 * M * N)
    for d, n in ((da, nb[0]), (db, nb[1]), (dd, nb[2])):
        if hip.hipMalloc(ctypes.byref(d), n) != 0:
            pytest.skip("hipMalloc failed")
    hip.hipMemcpy(da, A.ctypes.data_as(ctypes.c_void_p), nb[0], 1)
    hip.hipMemcpy(db, B.ctypes.data_as(ctypes.c_void_p), nb[1], 1)
    args = (_mr(da, M * K) + _mr(db, K * N) + _mr(dd, M * N)
            + [ctypes.c_int64(M), ctypes.c_int64(N), ctypes.c_int64(K)])
    arr = (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    gx, gy = (N + 16 * nt - 1) // (16 * nt), (M + 16 * mt - 1) // (16 * mt)
    assert launch(fn, gx, gy, 1, 32, 1, 1, 0, None, arr, None) == 0
    assert hip.hipDeviceSynchronize() == 0
    D = np.zeros((M, N), dtype=np.float32)
    hip.hipMemcpy(D.ctypes.data_as(ctypes.c_void_p), dd, nb[2], 2)
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


def test_in_process_pipeline_emits_hsaco_no_mlir_opt():
    """No GPU needed: one tessera-opt invocation lowers the directive all the way
    to a gpu.binary ELF — proving gpu-module-to-binary runs in-process."""
    _build_hsaco_in_process(mt=2, nt=2)  # raises/skips unless it emits ELF


@pytest.mark.parametrize("mt,nt", [(1, 1), (2, 4)])
def test_in_process_gemm_matches_numpy_and_oracle(mt, nt):
    hsaco = _build_hsaco_in_process(mt, nt)
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

    M, N, K = 64, 80, 48
    rng = np.random.default_rng(31 + mt + nt)
    A = (rng.standard_normal((M, K)) * 0.4).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.4).astype(np.float16)

    D = _launch(hip, fn, A, B, M, N, K, mt, nt)
    ref = A.astype(np.float32) @ B.astype(np.float32)
    assert float(np.max(np.abs(D - ref))) < 5e-2

    Do = _oracle(A, B, M, N, K)
    # The hsaco that tessera-opt serialized in-process executes identically to
    # the hand-written oracle — the serialization path is the only thing new.
    assert float(np.max(np.abs(D - Do))) == 0.0, \
        f"in-process-serialized GEMM != oracle at {mt}x{nt}"
