"""APPLE-MATMUL2D-1: the canonical GEMM reduction on the Metal 4 cooperative-tensor lane.

Graph IR matmul -> shared TilingPass -> `tessera_apple.gpu.tensor_view` +
`gpu.matmul2d` (storage pair, fp32 accumulator, MTLTensor layout quantum in
verified IR) -> `gpu.kernel_call` on the runtime's matmul2d symbols -> the
value-lane dispatcher, which consumes the projected view ABI and never
re-derives layout. Lowering tests are host-free; execution rows run on the
owning Mac and compare against a float64 oracle built from the exact quantized
bytes. Recognition is not promotion: nothing here changes route selection.
"""
from __future__ import annotations

import re
import subprocess

import numpy as np
import pytest

from tests._support.environment import CompilerToolchain

PASSES = ("tessera-tiling", "tessera-apple-canonical-gemm-matmul2d", "tessera-apple-matmul2d-to-call")


def _gemm_module(m: int, k: int, n: int, a_storage: str, b_storage: str) -> str:
    return f"""module {{
  func.func @gemm(%a: tensor<{m}x{k}x{a_storage}>, %b: tensor<{k}x{n}x{b_storage}>) -> tensor<{m}x{n}xf32> {{
    %0 = "tessera.matmul"(%a, %b) : (tensor<{m}x{k}x{a_storage}>, tensor<{k}x{n}x{b_storage}>) -> tensor<{m}x{n}xf32>
    return %0 : tensor<{m}x{n}xf32>
  }}
}}"""


def _lower(toolchain: CompilerToolchain, module: str, *passes: str) -> subprocess.CompletedProcess[str]:
    tessera_opt = toolchain.require_tessera_opt(*passes)
    return subprocess.run([str(tessera_opt), "-", *(f"--{p}" for p in passes), "--allow-unregistered-dialect"],
                          input=module, capture_output=True, text=True)


def _call(stdout: str) -> dict[str, str]:
    lines = [line for line in stdout.splitlines() if "tessera_apple.gpu.kernel_call" in line]
    assert lines, f"no kernel_call in:\n{stdout}"
    return {key: quoted or bare for key, quoted, bare in re.findall(r'([\w.]+) = (?:"([^"]+)"|([^,\s}]+))', lines[0])}


# ---------------------------------------------------------------- host-free lowering

def test_f16_canonical_loop_becomes_tensor_views_and_matmul2d(compiler_toolchain: CompilerToolchain):
    proc = _lower(compiler_toolchain, _gemm_module(64, 128, 96, "f16", "f16"), *PASSES[:2])
    assert proc.returncode == 0, proc.stderr
    assert "scf.for" not in proc.stdout
    assert proc.stdout.count("tessera_apple.gpu.tensor_view") == 2
    assert "tessera_apple.gpu.matmul2d" in proc.stdout
    assert 'accumulate = "f32"' in proc.stdout
    assert "extents = array<i64: 128, 64>, strides = array<i64: 1, 128>" in proc.stdout


@pytest.mark.parametrize("a,b,symbol,fmt", [
    ("f16", "f16", "tessera_apple_gpu_mtl4_matmul2d_f16", None),
    ("bf16", "bf16", "tessera_apple_gpu_mtl4_matmul2d_bf16", None),
    ("f8E4M3FN", "f8E4M3FN", "tessera_apple_gpu_mtl4_matmul2d_lowp", "0"),
    ("f8E5M2", "f8E5M2", "tessera_apple_gpu_mtl4_matmul2d_lowp", "1"),
    ("f4E2M1FN", "f4E2M1FN", "tessera_apple_gpu_mtl4_matmul2d_lowp", "2"),
    ("f16", "f8E4M3FN", "tessera_apple_gpu_mtl4_matmul2d_lowp", "3"),
    ("f16", "f4E2M1FN", "tessera_apple_gpu_mtl4_matmul2d_lowp", "5"),
])
def test_call_projects_symbol_and_view_abi_from_ir(compiler_toolchain, a, b, symbol, fmt):
    proc = _lower(compiler_toolchain, _gemm_module(64, 256, 256, a, b), *PASSES)
    assert proc.returncode == 0, proc.stderr
    call = _call(proc.stdout)
    assert call["symbol"] == symbol and call["op_kind"] == "mtl4_matmul2d"
    assert call["dtype"] == f"{a}x{b}" and call["tessera_apple.accumulate"] == "fp32"
    assert (call["tessera_apple.a_inner"], call["tessera_apple.a_outer"], call["tessera_apple.a_stride"]) == ("256", "64", "256")
    assert (call["tessera_apple.b_inner"], call["tessera_apple.b_outer"], call["tessera_apple.b_stride"]) == ("256", "256", "256")
    assert call.get("tessera_apple.lowp_format") == fmt


def test_packed_fp8_off_the_quantum_is_refused_not_reformed(compiler_toolchain):
    proc = _lower(compiler_toolchain, _gemm_module(64, 64, 128, "f8E4M3FN", "f8E4M3FN"), *PASSES[:2])
    assert proc.returncode != 0
    assert "APPLE_MATMUL2D_LAYOUT_UNSUPPORTED" in proc.stderr and "128 bytes" in proc.stderr


def test_mixed_bf16_f16_pair_is_refused(compiler_toolchain):
    proc = _lower(compiler_toolchain, _gemm_module(64, 128, 128, "bf16", "f16"), *PASSES[:2])
    assert proc.returncode != 0
    assert "APPLE_MATMUL2D_PAIR_UNSUPPORTED" in proc.stderr


def test_f32_storage_is_not_an_mpp_pair(compiler_toolchain):
    proc = _lower(compiler_toolchain, _gemm_module(64, 128, 128, "f32", "f32"), *PASSES[:2])
    assert proc.returncode != 0
    assert "APPLE_MATMUL2D_PAIR_UNSUPPORTED" in proc.stderr


# ---------------------------------------------------------------- owning-Mac execution

def _decode(codes, elem, np):
    from tessera import runtime as R
    fmt = {"f8E4M3FN": "fp8_e4m3", "f8E5M2": "fp8_e5m2", "f4E2M1FN": "fp4_e2m1"}[elem]
    return R.apple_lowp_decode(codes, fmt, np)


@pytest.mark.metal4
@pytest.mark.parametrize("a,b,m", [("f16", "f16", 64), ("bf16", "bf16", 100), ("f8E4M3FN", "f8E4M3FN", 64),
                                   ("f8E5M2", "f8E5M2", 100), ("f4E2M1FN", "f4E2M1FN", 64),
                                   ("f16", "f8E4M3FN", 64), ("f16", "f4E2M1FN", 100)])
def test_lowered_call_executes_through_the_value_lane_and_matches_the_oracle(compiler_toolchain, a, b, m):
    from tessera import runtime as R
    if not R._apple_gpu_mtl4_matmul2d_lane_available():
        pytest.fail("Metal 4 is up but the runtime lacks the matmul2d symbols: rebuild the dylib")
    ml = pytest.importorskip("ml_dtypes")
    K, N = 256, 256
    proc = _lower(compiler_toolchain, _gemm_module(m, K, N, a, b), *PASSES)
    assert proc.returncode == 0, proc.stderr
    call = _call(proc.stdout)
    rng = np.random.default_rng(m + K)
    kinds = {"f16": np.float16, "bf16": ml.bfloat16, "f8E4M3FN": ml.float8_e4m3fn,
             "f8E5M2": ml.float8_e5m2, "f4E2M1FN": ml.float4_e2m1fn}

    def operand(elem, shape):
        if elem in ("f16", "bf16"):
            return (rng.standard_normal(shape) * 0.25).astype(kinds[elem])
        # exact quantized bytes: draw codes with bounded exponents so K-long fp32 dot products stay exact
        if elem == "f4E2M1FN":
            return rng.integers(0, 16, size=shape, dtype=np.uint8).view(ml.float4_e2m1fn)
        c = rng.integers(0, 256, size=shape, dtype=np.uint8)
        c = ((c & 0x87) | (((c >> 3) & 0xF) % 12) << 3) if elem == "f8E4M3FN" else ((c & 0x83) | (((c >> 2) & 0x1F) % 24) << 2)
        return c.astype(np.uint8).view(kinds[elem])

    # The padded-M form the tiling pass emits: execute exactly what the IR states.
    Mp = int(call["tessera_apple.a_outer"])
    A = operand(a, (Mp, K))
    B = operand(b, (K, N))
    out = R._dispatch_gpu_mtl4_matmul2d([A, B], call, np)
    ref = A.astype(np.float64) @ B.astype(np.float64)
    assert out.dtype == np.float32 and out.shape == (Mp, N)
    scale = np.abs(ref).max() + 1.0
    tol = 4e-6 if a not in ("f16", "bf16") else 3e-2
    assert np.abs(out.astype(np.float64) - ref).max() <= tol * scale


@pytest.mark.metal4
def test_value_lane_refuses_a_call_without_the_view_abi():
    from tessera import runtime as R
    call = {"symbol": "tessera_apple_gpu_mtl4_matmul2d_f16", "dtype": "f16xf16", "op_kind": "mtl4_matmul2d"}
    with pytest.raises(ValueError, match="view ABI"):
        R._dispatch_gpu_mtl4_matmul2d([np.zeros((8, 8), np.float16)] * 2, call, np)
