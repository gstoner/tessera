"""MLA-1 + MLA-2 — DeepSeek MLA decode fusion (Schedule IR + Apple GPU runtime).

Forward correctness of the fused op (`ops.mla_decode_fused`) and the
runtime symbol contract for `tessera_apple_gpu_mla_decode_f32`.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tessera as ts


ROOT = Path(__file__).resolve().parents[2]


def _np_mla_decode_reference(x, w_dkv, w_uk, w_uv, q):
    """Hand-walked reference matching the fused op's contract."""
    c = x @ w_dkv
    K = c @ w_uk
    V = c @ w_uv
    # Standard scaled-dot-product attention with broadcast across batch.
    d = q.shape[-1]
    scale = 1.0 / np.sqrt(d)
    # Expand K/V to match Q's batch dim if Q is (B, S_q, D).
    if q.ndim == 3 and K.ndim == 2:
        K_b = np.broadcast_to(K, (q.shape[0],) + K.shape)
        V_b = np.broadcast_to(V, (q.shape[0],) + V.shape)
    else:
        K_b, V_b = K, V
    scores = np.matmul(q, np.swapaxes(K_b, -1, -2)) * scale
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    return np.matmul(p, V_b)


# ─────────────────────────────────────────────────────────────────────────────
# Forward correctness
# ─────────────────────────────────────────────────────────────────────────────


class TestMLADecodeFusedForward:
    def test_forward_matches_chain_reference(self):
        np.random.seed(0)
        S_kv, D_x, D_lat, D_h, B, S_q = 8, 16, 32, 16, 1, 4
        x = np.random.randn(S_kv, D_x).astype(np.float32) * 0.3
        Wdkv = np.random.randn(D_x, D_lat).astype(np.float32) * 0.3
        Wuk = np.random.randn(D_lat, D_h).astype(np.float32) * 0.3
        Wuv = np.random.randn(D_lat, D_h).astype(np.float32) * 0.3
        Q = np.random.randn(B, S_q, D_h).astype(np.float32) * 0.3

        out = ts.ops.mla_decode_fused(x, Wdkv, Wuk, Wuv, Q, causal=False)
        ref = _np_mla_decode_reference(x, Wdkv, Wuk, Wuv, Q)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Apple GPU runtime symbol contract
# ─────────────────────────────────────────────────────────────────────────────


def test_apple_gpu_mla_decode_runtime_shim_exposes_symbol(tmp_path):
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler is not available")

    backend = ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime"
    if sys.platform == "darwin":
        source = backend / "apple_gpu_runtime.mm"
        lib = tmp_path / "libtessera_apple_gpu_runtime.dylib"
        cmd = [
            cxx, "-std=c++17", "-shared", "-fPIC", "-fobjc-arc",
            "-x", "objective-c++", str(source), "-o", str(lib),
            "-framework", "Foundation",
            "-framework", "Metal",
            "-framework", "MetalPerformanceShaders",
            "-framework", "MetalPerformanceShadersGraph",
        ]
    else:
        source = backend / "apple_gpu_runtime_stub.cpp"
        lib = tmp_path / "libtessera_apple_gpu_runtime.so"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    runtime = ctypes.CDLL(str(lib))
    sym = runtime.tessera_apple_gpu_mla_decode_f32
    sym.argtypes = [ctypes.POINTER(ctypes.c_float)] * 6 + [ctypes.c_int32] * 6
    sym.restype = None

    # Verify numerical match against the Python reference.
    np.random.seed(0)
    S_kv, D_x, D_lat, D_h, B, S_q = 4, 8, 16, 8, 1, 3
    x = np.random.randn(S_kv, D_x).astype(np.float32) * 0.3
    Wdkv = np.random.randn(D_x, D_lat).astype(np.float32) * 0.3
    Wuk = np.random.randn(D_lat, D_h).astype(np.float32) * 0.3
    Wuv = np.random.randn(D_lat, D_h).astype(np.float32) * 0.3
    Q = np.random.randn(B, S_q, D_h).astype(np.float32) * 0.3
    O = np.zeros((B, S_q, D_h), dtype=np.float32)

    sym(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Wdkv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Wuk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Wuv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        O.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B, S_kv, D_x, D_lat, S_q, D_h,
    )

    ref = _np_mla_decode_reference(x, Wdkv, Wuk, Wuv, Q)
    np.testing.assert_allclose(O, ref, rtol=1e-4, atol=1e-5)
