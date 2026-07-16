"""Conv2d native-multi-tile spike — regression lock for the cracked MPP path.

The previously-blocked native ``mpp::tensor_ops::convolution2d`` multi-tile lane
now works end-to-end on Apple GPU via grid-of-threadgroups + per-tile slice on
both source and destination. This test pins:

- The single-tile baseline ``tessera_apple_gpu_spike_conv2d_single_tile_f16``
  (one threadgroup, 8×8 dest, VALID 3×3, Cin=Cout=4) is bit-correct vs numpy.
- The multi-tile entry ``tessera_apple_gpu_spike_conv2d_multi_tile_f16`` is
  bit-correct across aligned grid sizes (1×1, 2×2, 3×2, 4×4).

It is a **spike** (narrow scope: hardcoded Cin=Cout=4, K=3, stride=1, aligned
tile, f16-in/f32-out). The proof is locked so the cracked-but-not-shipped lane
can't silently regress while we work on the productionization follow-ups
(Cin/Cout templating, non-aligned boundary, dtype variants).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _on_darwin() -> bool:
    return os.uname().sysname == "Darwin"


def _runtime_lib_path() -> Path | None:
    """Try the locations the runtime build leaves the shared lib in."""
    for cand in (
        REPO_ROOT / "build" / "src" / "compiler" / "codegen" / "Tessera_Apple_Backend"
        / "runtime" / "libTesseraAppleRuntime.dylib",
        Path("/tmp/ts_conv_spike.dylib"),
    ):
        if cand.is_file():
            return cand
    return None


pytestmark = pytest.mark.hardware_apple_gpu


def _compile_runtime_dylib_if_needed(tmp_path: Path) -> Path:
    """Compile the runtime .mm into a standalone dylib if no pre-built one
    exists. Avoids depending on the ninja build state."""
    lib = _runtime_lib_path()
    if lib is not None:
        return lib
    cxx = shutil.which("c++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler available")
    src = REPO_ROOT / "src" / "compiler" / "codegen" / "Tessera_Apple_Backend" / "runtime" / "apple_gpu_runtime.mm"
    out = tmp_path / "spike.dylib"
    r = subprocess.run(
        [cxx, "-std=c++17", "-shared", "-fPIC", "-fobjc-arc", "-O2",
         "-x", "objective-c++", str(src), "-o", str(out),
         "-framework", "Foundation", "-framework", "Metal",
         "-framework", "MetalPerformanceShaders",
         "-framework", "MetalPerformanceShadersGraph"],
        capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        pytest.fail(f"runtime compile failed:\n{r.stderr[:4000]}")
    return out


@pytest.fixture(scope="module")
def runtime_lib(tmp_path_factory):
    import ctypes
    tmp = tmp_path_factory.mktemp("conv2d_spike")
    libpath = _compile_runtime_dylib_if_needed(tmp)
    lib = ctypes.CDLL(str(libpath))
    return lib


_Cin, _Cout, _kH, _kW, _B = 4, 4, 3, 3, 1


def _conv_ref(X: np.ndarray, Wt: np.ndarray, dstH: int, dstW: int) -> np.ndarray:
    """VALID conv reference in f64 (numpy NHWC / HWIO)."""
    Xf = X.astype(np.float64); Wf = Wt.astype(np.float64)
    out = np.zeros((_B, dstH, dstW, _Cout), np.float64)
    for n in range(_B):
        for oh in range(dstH):
            for ow in range(dstW):
                for co in range(_Cout):
                    s = 0.0
                    for ky in range(_kH):
                        for kx in range(_kW):
                            for ci in range(_Cin):
                                s += Xf[n, oh + ky, ow + kx, ci] * Wf[ky, kx, ci, co]
                    out[n, oh, ow, co] = s
    return out


def _make_inputs(dstH: int, dstW: int, seed: int):
    srcH, srcW = dstH + (_kH - 1), dstW + (_kW - 1)
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((_B, srcH, srcW, _Cin)) * 0.2).astype(np.float16)
    Wt = (rng.standard_normal((_kH, _kW, _Cin, _Cout)) * 0.3).astype(np.float16)
    return X, Wt, srcH, srcW


def test_conv2d_native_single_tile_matches_numpy(runtime_lib):
    """Single threadgroup, 8×8 dest, VALID 3×3 — was the spike's baseline."""
    import ctypes
    sym = runtime_lib.tessera_apple_gpu_spike_conv2d_single_tile_f16
    u16 = ctypes.POINTER(ctypes.c_uint16)
    fp = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [u16, u16, fp]
    sym.restype = ctypes.c_int32
    X, Wt, _, _ = _make_inputs(8, 8, seed=0)
    Y = np.zeros((_B, 8, 8, _Cout), np.float32)
    rc = sym(X.view(np.uint16).ctypes.data_as(u16),
             Wt.view(np.uint16).ctypes.data_as(u16),
             Y.ctypes.data_as(fp))
    assert rc == 1, "single-tile spike did not run on the MTL4 lane"
    ref = _conv_ref(X, Wt, 8, 8)
    rel = float(np.abs(Y - ref).max() / (np.abs(ref).max() + 1e-9))
    assert rel < 5e-6, f"single-tile rel error {rel:.3e} (expected < 5e-6)"


@pytest.mark.parametrize("dstH,dstW", [(8, 8), (16, 16), (24, 16), (32, 32)])
def test_conv2d_native_multi_tile_matches_numpy(runtime_lib, dstH, dstW):
    """Grid-of-threadgroups multi-tile, aligned (dstH,dstW multiples of 8).

    The previously-blocked tiling path. Validates that per-tile X.slice() +
    Y.slice() with innermost-first offsets produces bit-correct output across
    1×1 / 2×2 / 3×2 / 4×4 grids.
    """
    import ctypes
    sym = runtime_lib.tessera_apple_gpu_spike_conv2d_multi_tile_f16
    u16 = ctypes.POINTER(ctypes.c_uint16)
    fp = ctypes.POINTER(ctypes.c_float)
    i32 = ctypes.c_int32
    sym.argtypes = [u16, u16, fp, i32, i32]
    sym.restype = i32
    X, Wt, _, _ = _make_inputs(dstH, dstW, seed=dstH * 1000 + dstW)
    Y = np.zeros((_B, dstH, dstW, _Cout), np.float32)
    rc = sym(X.view(np.uint16).ctypes.data_as(u16),
             Wt.view(np.uint16).ctypes.data_as(u16),
             Y.ctypes.data_as(fp), i32(dstH), i32(dstW))
    assert rc == 1, f"multi-tile spike did not run for ({dstH},{dstW})"
    ref = _conv_ref(X, Wt, dstH, dstW)
    rel = float(np.abs(Y - ref).max() / (np.abs(ref).max() + 1e-9))
    assert rel < 5e-6, f"multi-tile rel error {rel:.3e} at ({dstH},{dstW})"


def test_conv2d_native_multi_tile_rejects_unaligned(runtime_lib):
    """The spike scope is aligned tiles only — unaligned must fail cleanly,
    not silently emit garbage (the discipline that protected SIMD-reduction)."""
    import ctypes
    sym = runtime_lib.tessera_apple_gpu_spike_conv2d_multi_tile_f16
    u16 = ctypes.POINTER(ctypes.c_uint16)
    fp = ctypes.POINTER(ctypes.c_float); i32 = ctypes.c_int32
    sym.argtypes = [u16, u16, fp, i32, i32]; sym.restype = i32
    X, Wt, _, _ = _make_inputs(8, 8, seed=1)
    Y = np.zeros((_B, 13, 8, _Cout), np.float32)   # 13 isn't a multiple of 8
    rc = sym(X.view(np.uint16).ctypes.data_as(u16),
             Wt.view(np.uint16).ctypes.data_as(u16),
             Y.ctypes.data_as(fp), i32(13), i32(8))
    assert rc == 0, "spike must refuse unaligned tile shapes (it returned 1)"
