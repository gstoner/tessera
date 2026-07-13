"""Non-Darwin Apple-GPU runtime *stub* must COMPUTE, not zero-fill.

The stub (`apple_gpu_runtime_stub.cpp`) is the portable fallback used on Linux CI,
where the real Metal runtime is unavailable. It is `#if !defined(__APPLE__)`
guarded, so on a Mac it compiles to nothing and its half-precision references are
never exercised — they can silently rot to zero-fill and no Darwin test would
notice (Darwin runs the real Metal path; Linux skips the execution tests).

This test closes that gap on *every* platform: it compiles the stub (flipping the
guard via a tiny wrapper), then checks the f16 bmm / bsmm / conv2d / conv3d stubs
produce correct, **non-zero** references vs numpy. It also guards against the
`static inline` forward-decl/definition split that made Clang reject the stub.
Skips only when no C++ compiler is available.
"""

import ctypes
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

_STUB = (Path(__file__).resolve().parents[2] / "src" / "compiler" / "codegen"
         / "Tessera_Apple_Backend" / "runtime" / "apple_gpu_runtime_stub.cpp")
_ROOT = Path(__file__).resolve().parents[2]


def test_shared_runtime_build_is_safe_under_parallel_cache_misses(tmp_path, monkeypatch):
    """A worker must never load another worker's partially linked runtime."""
    if shutil.which("c++") is None and shutil.which("clang++") is None and shutil.which("g++") is None:
        pytest.skip("no C++ compiler available")
    monkeypatch.setenv("TESSERA_CACHE_DIR", str(tmp_path / "cache"))
    from tessera.runtime import _build_apple_gpu_runtime_shared

    with ThreadPoolExecutor(max_workers=4) as pool:
        built = list(pool.map(lambda _: _build_apple_gpu_runtime_shared(_ROOT), range(4)))

    assert len({path.resolve() for path in built}) == 1
    assert built[0].is_file()
    ctypes.CDLL(str(built[0]))


@pytest.fixture(scope="module")
def stub_lib(tmp_path_factory):
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no C++ compiler available")
    if not _STUB.exists():
        pytest.skip("stub source not found")
    tmp = tmp_path_factory.mktemp("apple_gpu_stub")
    # Pull in the std headers (with __APPLE__ as-is so system headers are happy),
    # then undefine __APPLE__ so the stub's `#if !defined(__APPLE__)` body compiles.
    wrap = tmp / "wrap.cpp"
    wrap.write_text(
        "#include <algorithm>\n#include <cmath>\n#include <cstdint>\n"
        "#include <cstring>\n#include <limits>\n#include <vector>\n#include <cstdlib>\n"
        "#undef __APPLE__\n"
        f'#include "{_STUB}"\n'
    )
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    lib = tmp / f"libstub{ext}"
    r = subprocess.run([cxx, "-std=c++17", "-shared", "-fPIC", str(wrap), "-o", str(lib)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"stub failed to compile:\n{r.stderr[:2000]}")
    return ctypes.CDLL(str(lib))


_U16 = ctypes.POINTER(ctypes.c_uint16)
_FP = ctypes.POINTER(ctypes.c_float)
_I32 = ctypes.c_int32


def _u16(a):
    return np.ascontiguousarray(a, np.float16).view(np.uint16).ctypes.data_as(_U16)


def _rel(x, ref):
    ref = np.asarray(ref, np.float64)
    return float(np.abs(np.asarray(x, np.float64) - ref).max() / (np.abs(ref).max() + 1e-9))


def test_stub_bmm_f16_computes(stub_lib):
    fn = stub_lib.tessera_apple_gpu_bmm_f16
    fn.argtypes = [_U16, _U16, _U16, _I32, _I32, _I32, _I32, _I32]
    fn.restype = None
    rng = np.random.default_rng(0)
    B, M, N, K = 2, 4, 5, 6
    A = (rng.standard_normal((B, M, K)) * 0.3).astype(np.float16)
    Bm = (rng.standard_normal((B, K, N)) * 0.3).astype(np.float16)
    O = np.zeros((B, M, N), np.float16)
    fn(_u16(A), _u16(Bm), O.view(np.uint16).ctypes.data_as(_U16),
       _I32(B), _I32(M), _I32(N), _I32(K), _I32(0))
    assert np.any(O != 0)                                     # not zero-filled
    assert _rel(O.astype(np.float32), np.matmul(A.astype(np.float64), Bm.astype(np.float64))) < 1e-2


def test_stub_bsmm_f16_computes(stub_lib):
    # Regression: this stub used to zero-fill despite the f32 bsmm reference
    # existing right beside it.
    fn = stub_lib.tessera_apple_gpu_mpsgraph_bsmm_f16
    fn.argtypes = [_U16, _U16, _U16, _U16, _I32, _I32, _I32, _I32, _I32, ctypes.c_float]
    fn.restype = None
    rng = np.random.default_rng(1)
    B, M, N, P, K = 2, 3, 4, 5, 6
    A = (rng.standard_normal((B, M, K)) * 0.3).astype(np.float16)
    Bm = (rng.standard_normal((B, K, N)) * 0.3).astype(np.float16)
    C = (rng.standard_normal((B, N, P)) * 0.3).astype(np.float16)
    O = np.zeros((B, M, P), np.float16)
    fn(_u16(A), _u16(Bm), _u16(C), O.view(np.uint16).ctypes.data_as(_U16),
       _I32(B), _I32(M), _I32(N), _I32(P), _I32(K), ctypes.c_float(0.5))
    assert np.any(O != 0)
    s = (A.astype(np.float64) @ Bm.astype(np.float64)) * 0.5
    s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
    assert _rel(O.astype(np.float64), s @ C.astype(np.float64)) < 1e-2


def test_stub_conv2d_f16_computes(stub_lib):
    fn = stub_lib.tessera_apple_gpu_conv2d_f16
    fn.argtypes = [_U16, _U16, _U16, _U16] + [_I32] * 14
    fn.restype = None
    rng = np.random.default_rng(2)
    N, H, W, Cin, Cout, kH, kW = 1, 5, 5, 2, 3, 3, 3
    X = (rng.standard_normal((N, H, W, Cin)) * 0.3).astype(np.float16)
    Wt = (rng.standard_normal((kH, kW, Cin, Cout)) * 0.2).astype(np.float16)
    oH = oW = 3
    O = np.zeros((N, oH, oW, Cout), np.float16)
    fn(_u16(X), _u16(Wt), None, O.view(np.uint16).ctypes.data_as(_U16),
       _I32(N), _I32(H), _I32(W), _I32(Cin), _I32(Cout), _I32(kH), _I32(kW),
       _I32(1), _I32(1), _I32(0), _I32(0), _I32(1), _I32(1), _I32(1))
    assert np.any(O != 0)


def test_stub_conv3d_f16_computes(stub_lib):
    # Regression: conv3d f16 used to zero-fill despite reference_conv3d_f32_stub.
    fn = stub_lib.tessera_apple_gpu_conv3d_f16
    fn.argtypes = [_U16, _U16, _U16, _U16] + [_I32] * 19
    fn.restype = None
    rng = np.random.default_rng(3)
    N, iD, iH, iW, Cin, Cout = 1, 4, 4, 4, 2, 3
    X = (rng.standard_normal((N, iD, iH, iW, Cin)) * 0.3).astype(np.float16)
    Wt = (rng.standard_normal((2, 2, 2, Cin, Cout)) * 0.2).astype(np.float16)
    oD = oH = oW = 3
    O = np.zeros((N, oD, oH, oW, Cout), np.float16)
    fn(_u16(X), _u16(Wt), None, O.view(np.uint16).ctypes.data_as(_U16),
       _I32(N), _I32(iD), _I32(iH), _I32(iW), _I32(Cin), _I32(Cout),
       _I32(2), _I32(2), _I32(2), _I32(1), _I32(1), _I32(1), _I32(0), _I32(0),
       _I32(0), _I32(1), _I32(1), _I32(1), _I32(1))
    assert np.any(O != 0)
