from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tessera as ts
from tessera.runtime import launch


ROOT = Path(__file__).resolve().parents[2]


def test_apple_cpu_target_reports_accelerate_execution_mode_and_runtime_pipeline():
    @ts.jit(target="apple_cpu")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    artifact = mm.runtime_artifact()

    assert artifact.metadata["execution_mode"] == "cpu_accelerate"
    assert artifact.metadata["runtime_status"] == "artifact_only"
    assert artifact.metadata["compiler_path"] == "target_ir_artifact"
    assert artifact.metadata["artifact_hashes"]["backend"]
    assert 'execution_mode = "cpu_accelerate"' in mm.target_ir
    assert "tessera-lower-to-apple_cpu-runtime" in mm.compile_bundle.artifact("backend").text

    result = launch(artifact, args=(np.eye(2, dtype=np.float32), np.ones((2, 2), dtype=np.float32)))
    assert result["ok"] is False
    assert result["runtime_status"] in {"missing_backend", "unimplemented"}
    assert result["compiler_path"] == "target_ir_artifact"


def test_apple_gpu_tiny_decode_artifact_covers_rope_softmax_matmul_and_kv_cache():
    @ts.jit(target="apple_gpu")
    def tiny_decode(x, wq, wk, wv, wo, theta, cache):
        q = ts.ops.matmul(x, wq)
        k = ts.ops.matmul(x, wk)
        v = ts.ops.matmul(x, wv)
        q_rot = ts.ops.rope(q, theta)
        k_rot = ts.ops.rope(k, theta)
        cache_next = ts.ops.kv_cache_append(cache, k_rot, v)
        scores = ts.ops.matmul(q_rot, ts.ops.transpose(k_rot))
        probs = ts.ops.softmax(scores)
        ctx = ts.ops.matmul(probs, v)
        return ts.ops.matmul(ctx, wo)

    target_ir = tiny_decode.target_ir
    artifact = tiny_decode.runtime_artifact()

    assert artifact.metadata["execution_mode"] == "metal_artifact"
    assert artifact.metadata["runtime_status"] == "artifact_only"
    assert "matmul_contract" in target_ir
    assert "rope_contract" in target_ir
    assert "softmax_contract" in target_ir
    assert "tessera_apple.diagnostic" in target_ir
    assert "KV-cache target lowering is not implemented for Apple GPU" in target_ir
    assert 'framework = "MPSGraph"' in target_ir
    assert 'execution_mode = "metal_artifact"' in target_ir
    assert not tiny_decode.uses_compiled_path


def test_rope_reference_path_executes_tiny_decode_proxy_on_cpu():
    @ts.jit
    def tiny_decode_cpu(x, wq, wk, wv, wo, theta):
        q = ts.ops.rope(ts.ops.matmul(x, wq), theta)
        k = ts.ops.rope(ts.ops.matmul(x, wk), theta)
        v = ts.ops.matmul(x, wv)
        scores = ts.ops.matmul(q, ts.ops.transpose(k))
        probs = ts.ops.softmax(scores)
        ctx = ts.ops.matmul(probs, v)
        return ts.ops.matmul(ctx, wo)

    x = np.arange(16, dtype=np.float32).reshape(4, 4) / 16.0
    w = np.eye(4, dtype=np.float32)
    theta = np.zeros((4, 4), dtype=np.float32)

    out = tiny_decode_cpu(x, w, w, w, w, theta)

    assert tiny_decode_cpu.uses_compiled_path
    assert "tessera.rope" in tiny_decode_cpu.ir_text()
    assert "tile.rotary_pair" in tiny_decode_cpu.tile_ir
    np.testing.assert_allclose(out, ts.ops.softmax(x @ x.T) @ x)


def test_apple_cpu_runtime_shim_gemm_f32_correctness(tmp_path):
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler is not available")

    source = ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_cpu_runtime.cpp"
    lib = tmp_path / ("libtessera_apple_cpu_runtime.dylib" if sys.platform == "darwin" else "libtessera_apple_cpu_runtime.so")
    cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    if sys.platform == "darwin":
        cmd.extend(["-framework", "Accelerate"])
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    runtime = ctypes.CDLL(str(lib))
    gemm = runtime.tessera_apple_cpu_gemm_f32
    gemm.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    gemm.restype = None

    for m, n, k in ((2, 2, 2), (2, 3, 4), (1, 5, 3)):
        a = np.arange(m * k, dtype=np.float32).reshape(m, k)
        b = (np.arange(k * n, dtype=np.float32).reshape(k, n) / 7.0)
        c = np.zeros((m, n), dtype=np.float32)
        gemm(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            m,
            n,
            k,
        )
        np.testing.assert_allclose(c, a @ b, rtol=1e-5, atol=1e-5)
