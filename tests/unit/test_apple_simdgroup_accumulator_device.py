"""APPLE-ACCUM-1 -- every admitted simdgroup accumulator, re-derived on the GPU.

The Apple simdgroup lane admits an accumulator only when the device computes
exactly what the program declared. This file is the evidence behind
``apple_fragment.SIMDGROUP_ACCUMULATORS`` and the C++ verifier. The 16-bit
storage pairs (fp16/bf16 x fp32/fp16) run through the real TILE-1 runtime ABI
with the Tessera-emitted kernel. The f32-storage x fp16 pair, which only the IR
lane (``MatmulToAppleSimdgroup``) admits -- the TILE-1 ABI takes 16-bit
operands -- runs as a hand-written kernel on the cooperative-matmul runtime
symbol (f32 operands), since no Tessera executor exists for it yet. Every row
must be **bit-exact** with a numpy model of the declared accumulation. A tolerance would hide the one fact
that matters here -- whether the partial sums really live in the declared
type -- so none is used.

The models (all measured on the M1 Max, Apple7, macOS 27.0, Metal toolchain
32023.921, 2026-09-26):

* fp32 accumulator: a sequential fp32 fused multiply-add chain over K.
* fp16 accumulator, fp16 storage: a sequential fp16 fused multiply-add chain
  (every step rounds to fp16).
* fp16 accumulator, bf16 or f32 storage: fp32 inside each 8-deep MMA
  (starting from the fp16 accumulator), rounded to nearest-even fp16 after
  every MMA.

The bf16 accumulator is *refused*, and ``test_bf16_accumulator_is_not_bf16``
keeps the measurement that refuses it honest: on this part it is bit-exact
with fp32 accumulation truncated to bf16 at the store, which is a different
numeric class than accum=bf16 declares.

The error of each reduced-precision accumulator against the fp32-accumulated
result is recorded at K = 4096 and asserted against bounds derived from that
measurement, not loosened to pass.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from tessera.compiler.apple_fragment import SIMDGROUP_ACCUMULATORS
from tessera.compiler.apple_target import AppleGPUArch, AppleGPUTargetProfile
from tessera.compiler.msl_gemm_emit import (
    dispatch_apple_simdgroup_tile_f16,
    materialize_apple_simdgroup_tile_msl,
)

pytestmark = pytest.mark.hardware_apple_gpu

TARGET = AppleGPUTargetProfile(AppleGPUArch.APPLE7)
MMA_K = 8  # simdgroup_multiply_accumulate is an 8x8x8 MMA on Apple7


def _require_device():
    from tests._support.apple import require_apple_metal

    require_apple_metal()


def _storage(dtype: str):
    if dtype == "fp16":
        return np.float16
    ml_dtypes = pytest.importorskip("ml_dtypes")
    return ml_dtypes.bfloat16


def _operands(dtype: str, m: int, k: int, n: int, seed: int):
    """Operands exactly representable in the storage type, returned as float64."""
    rng = np.random.default_rng(seed)
    st = _storage(dtype)
    a = rng.standard_normal((m, k)).astype(st)
    b = rng.standard_normal((k, n)).astype(st)
    return a, b, a.astype(np.float64), b.astype(np.float64)


def _run(dtype: str, accum: str, a, b, block=(32, 32, 16)) -> np.ndarray:
    artifact = materialize_apple_simdgroup_tile_msl(
        TARGET, dtype, *block, accumulator_dtype=accum)
    out, native = dispatch_apple_simdgroup_tile_f16(artifact, a, b)
    assert native is True, "TILE-1 did not dispatch on the GPU"
    return np.asarray(out, dtype=np.float64)


def _fma_chain(a64, b64, inner, slab, outer) -> np.ndarray:
    """Model: within each `slab` of K, fused multiply-add in `inner` starting
    from the carried accumulator; round the carried value to `outer` after
    each slab. Products of storage-exact operands are exact in float64, so each
    step rounds exactly once (a fused FMA)."""
    m, k = a64.shape
    carried = np.zeros((m, b64.shape[1]), dtype=outer)
    for k0 in range(0, k, slab):
        t = carried.astype(inner)
        for kk in range(k0, min(k, k0 + slab)):
            t = (t.astype(np.float64) + np.outer(a64[:, kk], b64[kk])).astype(inner)
        carried = t.astype(outer)
    return carried.astype(np.float64)


def _model(dtype: str, accum: str, a64, b64) -> np.ndarray:
    if accum == "fp32":
        return _fma_chain(a64, b64, np.float32, a64.shape[1], np.float32)
    if dtype == "fp16":
        return _fma_chain(a64, b64, np.float16, 1, np.float16)
    return _fma_chain(a64, b64, np.float32, MMA_K, np.float16)


def _admitted():
    return [(s, a) for s, accums in SIMDGROUP_ACCUMULATORS.items() for a in accums]


@pytest.mark.parametrize("dtype, accum", _admitted())
@pytest.mark.parametrize("shape", [(64, 64, 64), (37, 29, 203)])
def test_admitted_accumulator_is_bit_exact_with_its_declared_model(dtype, accum, shape):
    """Aligned and ragged shapes (zero padding adds exact +0 products)."""
    _require_device()
    m, n, k = shape
    a, b, a64, b64 = _operands(dtype, m, k, n, seed=m + n + k)
    got = _run(dtype, accum, a, b)
    want = _model(dtype, accum, a64, b64)
    mismatches = int(np.count_nonzero(got != want))
    assert mismatches == 0, (
        f"storage={dtype} accum={accum} {shape}: {mismatches} of {got.size} "
        f"elements differ from the declared-accumulation model (max |diff| "
        f"{np.abs(got - want).max():.3e})")


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_fp16_accumulator_error_at_k4096_is_the_measured_cost(dtype):
    """Record the price of an fp16 accumulator at an ordinary K.

    Measured 2026-09-26 on the M1 Max (64x64, K=4096, seed 4096, max |err| /
    max |exact|):
      fp16 storage: fp16 accum 1.25e-02, fp32 accum 1.8e-06
      bf16 storage: fp16 accum 5.6e-03,  fp32 accum 1.2e-06
    (A seed-0 probe of the same shape measured 1.8e-02 for fp16/fp16, so the
    ceiling carries about 2x headroom over the worst observed draw.)
    (bf16 storage keeps fp32 inside each 8-deep MMA and rounds to fp16 once per
    MMA, so it rounds 8x less often than the fp16-storage per-FMA chain.)

    The bounds bracket those measurements: the fp16-accumulated error must be
    far above the fp32 one (so the accumulator really is fp16, not a silent
    fp32) and below a ceiling set from the measurement (so a regression that
    rounds more often than declared fails).
    """
    _require_device()
    a, b, a64, b64 = _operands(dtype, 64, 4096, 64, seed=4096)
    exact = a64 @ b64
    scale = np.abs(exact).max()
    fp32 = _run(dtype, "fp32", a, b)
    fp16 = _run(dtype, "fp16", a, b)
    err32 = np.abs(fp32 - exact).max() / scale
    err16 = np.abs(fp16 - exact).max() / scale
    err16_vs_fp32 = np.abs(fp16 - fp32).max() / scale
    print(f"APPLE-ACCUM-1 storage={dtype} K=4096: fp16-accum err {err16:.3e}, "
          f"fp32-accum err {err32:.3e}, fp16 vs fp32 {err16_vs_fp32:.3e}")
    assert err32 < 1e-5
    ceiling = {"fp16": 4e-2, "bf16": 1.5e-2}[dtype]
    assert 1e-3 < err16 < ceiling
    assert err16 > 100 * err32


def test_bf16_accumulator_is_not_bf16():
    """Why accum=bf16 is refused: keep the measurement that refuses it live.

    A bfloat simdgroup accumulator compiles (the refusal is not a compile
    limit), but on Apple7 the result equals fp32 accumulation over the whole K
    truncated toward zero to bf16 -- not bf16 accumulation, and not even
    round-to-nearest. If a future OS/toolchain makes this a genuine bf16
    accumulator, this test fails and the refusal must be re-derived.
    """
    _require_device()
    from tessera._apple_gpu_dispatch import bind_registered

    ml_dtypes = pytest.importorskip("ml_dtypes")
    bf16 = ml_dtypes.bfloat16
    src = """#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void probe(device const bfloat* A [[buffer(0)]], device const bfloat* B [[buffer(1)]],
                  device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
                  constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],
                  uint3 tg [[threadgroup_position_in_grid]],
                  uint3 tid [[thread_position_in_threadgroup]]) {
  const uint m0 = tg.y * 8, n0 = tg.x * 8;
  simdgroup_matrix<bfloat, 8, 8> acc = make_filled_simdgroup_matrix<bfloat, 8, 8>(bfloat(0));
  simdgroup_matrix<bfloat, 8, 8> a, b;
  for (uint k0 = 0; k0 < K; k0 += 8) {
    simdgroup_load(a, A + m0 * K + k0, K);
    simdgroup_load(b, B + k0 * N + n0, N);
    simdgroup_multiply_accumulate(acc, a, b, acc);
  }
  threadgroup bfloat Cs[64];
  simdgroup_store(acc, Cs, 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid.x; e < 64; e += 32) C[(m0 + e / 8) * N + n0 + e % 8] = float(Cs[e]);
}
"""
    rng = np.random.default_rng(7)
    m = n = 64
    k = 512
    a = rng.standard_normal((m, k)).astype(bf16)
    b = rng.standard_normal((k, n)).astype(bf16)
    out = np.empty((m, n), dtype=np.float32)
    sym = bind_registered("tessera_apple_gpu_tile_simdgroup_gemm_bf16")
    assert sym is not None
    rc = sym(src.encode(), b"probe",
             a.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
             b.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
             out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), m, n, k, 8, 8, 32)
    assert rc == 1
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    fp32_whole_k = _fma_chain(a64, b64, np.float32, k, np.float32).astype(np.float32)
    truncated = (fp32_whole_k.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
    per_step_bf16 = _fma_chain(a64, b64, bf16, 1, bf16)
    assert np.array_equal(out, truncated), "bf16 accumulator no longer fp32+RTZ"
    assert not np.array_equal(out.astype(np.float64), per_step_bf16)


_F32_FP16_KERNEL = """#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void probe(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                  device float* C [[buffer(2)]], constant int& M [[buffer(3)]],
                  constant int& N [[buffer(4)]], constant int& K [[buffer(5)]],
                  uint3 tg [[threadgroup_position_in_grid]],
                  uint sg [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half Cs[4][64];
  for (uint f = sg; f < 16; f += 4) {
    const uint m0 = tg.y * 32 + (f / 4) * 8, n0 = tg.x * 32 + (f % 4) * 8;
    simdgroup_matrix<half, 8, 8> acc = make_filled_simdgroup_matrix<half, 8, 8>(half(0));
    simdgroup_matrix<float, 8, 8> a, b;
    for (int k0 = 0; k0 < K; k0 += 8) {
      simdgroup_load(a, A + m0 * K + k0, K);
      simdgroup_load(b, B + k0 * N + n0, N);
      simdgroup_multiply_accumulate(acc, a, b, acc);
    }
    simdgroup_store(acc, Cs[sg], 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 64; e += 32) C[(m0 + e / 8) * N + n0 + e % 8] = float(Cs[sg][e]);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}
"""


def test_f32_storage_fp16_accumulator_is_bit_exact_with_its_model():
    """Re-derives the f32 x fp16 row the C++ verifier and MatmulToAppleSimdgroup
    admit (APPLE-ACCUM-1 review): fp32 inside each 8-deep MMA, RNE to fp16 after
    every MMA. Hand-written kernel, so this proves the hardware fact the IR lane
    relies on, not an end-to-end route (none exists for f32 storage)."""
    _require_device()
    from tessera._apple_gpu_dispatch import apple_gpu_runtime

    rt = apple_gpu_runtime()
    assert rt is not None
    sym = rt.tessera_apple_gpu_synth_matmul_epilogue_coopmat
    rng = np.random.default_rng(32)
    for m, n, k in ((64, 64, 64), (64, 64, 1024)):
        a = rng.standard_normal((m, k)).astype(np.float32)
        b = rng.standard_normal((k, n)).astype(np.float32)
        out = np.full((m, n), np.nan, dtype=np.float32)
        rc = sym(ctypes.c_char_p(_F32_FP16_KERNEL.encode()), ctypes.c_char_p(b"probe"),
                 a.ctypes.data_as(ctypes.c_void_p), b.ctypes.data_as(ctypes.c_void_p),
                 None, out.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int32(m), ctypes.c_int32(n), ctypes.c_int32(k),
                 ctypes.c_int32(0), ctypes.c_int32(4), ctypes.c_int32(32))
        assert rc == 1
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        want = _fma_chain(a64, b64, np.float32, MMA_K, np.float16)
        assert np.array_equal(out.astype(np.float64), want), (m, n, k)


def _value_lane_module(mlir: str, dt: str, m: int, k: int, n: int, policy):
    from tessera.compiler.graph_ir import (
        GraphIRFunction,
        GraphIRModule,
        IRArg,
        IROp,
        IRType,
    )

    ta = IRType(f"tensor<{m}x{k}x{mlir}>", (str(m), str(k)), dt)
    tb = IRType(f"tensor<{k}x{n}x{mlir}>", (str(k), str(n)), dt)
    tc = IRType(f"tensor<{m}x{n}x{mlir}>", (str(m), str(n)), dt)
    return GraphIRModule(functions=[GraphIRFunction(
        name="f", args=[IRArg("a", ta), IRArg("b", tb)], result_types=[tc],
        body=[IROp(result="c", op_name="tessera.matmul", operands=["%a", "%b"],
                   operand_types=[ta.mlir_str, tb.mlir_str], result_type=tc.mlir_str,
                   kwargs={"numeric_policy": policy})],
        return_values=["%c"])])


def _value_lane_artifact(module):
    from tessera.compiler.canonical_compile import canonical_compile

    return canonical_compile(
        module, target="apple_gpu", options={"apple_target_ir_mode": "value"},
    ).to_runtime_artifact()


def test_value_lane_executes_the_declared_fp16_accumulator():
    """End to end: Graph IR numeric_policy accum=fp16 -> TileToApple stamps
    tessera_apple.accumulate -> the runtime materializes an fp16-accumulator
    kernel -> the GPU result is bit-exact with the fp16 model, not the fp32 one,
    and is returned in the declared f16 result dtype."""
    _require_device()
    from tessera.runtime import launch

    m, k, n = 32, 512, 32
    art = _value_lane_artifact(_value_lane_module(
        "f16", "fp16", m, k, n, {"storage": "fp16", "accum": "fp16"}))
    calls = art.metadata.get("apple_value_calls") or []
    assert calls and calls[0]["op_kind"] == "tile_simdgroup_gemm", art.metadata
    assert calls[0]["accumulate"] == "fp16" and calls[0]["result_dtype"] == "fp16"
    a, b, a64, b64 = _operands("fp16", m, k, n, seed=11)
    result = launch(art, [a, b])
    assert result["ok"], result
    assert result["output"].dtype == np.float16
    got = np.asarray(result["output"], dtype=np.float64)
    assert np.array_equal(got, _model("fp16", "fp16", a64, b64))
    assert not np.array_equal(got, _model("fp16", "fp32", a64, b64).astype(np.float16)
                              .astype(np.float64))


@pytest.mark.parametrize("dtype, mlir", [("fp16", "f16"), ("bf16", "bf16")])
def test_value_lane_returns_the_declared_result_dtype_rounded_once(dtype, mlir):
    """APPLE-ACCUM-1 review: the TILE-1 lane used to return the kernel's fp32
    buffer for a declared f16/bf16 result -- a different dtype and unrounded
    values. It now returns the declared dtype, rounded once (RNE) from the fp32
    accumulator, the same single rounding MatmulToAppleSimdgroup performs."""
    _require_device()
    from tessera.runtime import launch

    m, k, n = 32, 256, 32
    art = _value_lane_artifact(_value_lane_module(
        mlir, dtype, m, k, n, {"storage": dtype, "accum": "fp32"}))
    a, b, a64, b64 = _operands(dtype, m, k, n, seed=21)
    result = launch(art, [a, b])
    assert result["ok"], result
    st = _storage(dtype)
    assert result["output"].dtype == np.dtype(st)
    want = _model(dtype, "fp32", a64, b64).astype(np.float32).astype(st)
    assert np.array_equal(np.asarray(result["output"]).view(np.uint16),
                          np.asarray(want).view(np.uint16))


def test_value_lane_refuses_an_fp16_accumulator_into_a_bf16_result():
    """APPLE-ACCUM-1 review P1-1: bf16 x bf16 -> bf16 with accum=fp16 used to
    dispatch and return fp16-precision values in a bf16-declared tensor. It is
    now refused at TileToApple (the same rule the IR lane enforces) and never
    reaches the GPU."""
    art = _value_lane_artifact(_value_lane_module(
        "bf16", "bf16", 16, 32, 16, {"storage": "bf16", "accum": "fp16"}))
    assert art.metadata.get("compiler_path") != "apple_value_target_ir"
    error = str(art.metadata.get("apple_value_target_ir_error"))
    assert "APPLE_SIMDGROUP_ACCUM_UNSUPPORTED" in error
    assert "single-rounding" in error
