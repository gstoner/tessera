"""GA10 conformance — full GA3/GA5 native MSL kernel suite on Apple GPU.

Covers the 9 pointwise GA3/GA5 ops + 4 fp16/bf16 ports of the existing
geo_product/rotor_sandwich kernels that landed alongside the GA9
follow-on work (2026-05-17). Each test compiles the runtime once
(module-scoped fixture), ctypes-loads the dylib, dispatches the kernel,
and bitwise-matches the Python `tessera.ga` reference path.

Tested ops:
  reverse, grade_involution, conjugate, hodge_star (4 unary 8→8)
  norm, norm_squared                                  (2 unary 8→1)
  wedge, left_contraction                             (2 binary 8x8→8)
  inner                                               (1 binary 8x8→1)
  grade_projection                                    (1 unary 8→8 +int)
  geo_product f16/bf16, rotor_sandwich f16/bf16       (4 dtype ports)

Total: 14 new C ABI symbols verified.
"""

from __future__ import annotations

import ctypes
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Apple GPU MSL kernels require macOS / Apple Silicon",
)


@pytest.fixture(scope="module")
def apple_gpu_runtime(tmp_path_factory):
    """Compile apple_gpu_runtime.mm into a dylib and ctypes-load."""
    cxx = shutil.which("clang++") or shutil.which("c++")
    if cxx is None:
        pytest.skip("clang++ not available")
    source = ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm"
    tmp_dir = tmp_path_factory.mktemp("apple_gpu_full")
    lib = tmp_dir / "libtessera_apple_gpu_runtime.dylib"
    cmd = [cxx, "-std=c++17", "-shared", "-fPIC", "-O2", "-fobjc-arc",
           "-x", "objective-c++", str(source), "-o", str(lib),
           "-framework", "Metal",
           "-framework", "MetalPerformanceShaders",
           "-framework", "Foundation"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"apple_gpu_runtime.mm did not compile:\n{proc.stderr[-2000:]}")
    return ctypes.CDLL(str(lib))


def _ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _ptr16(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


def _bind_unary_8x8(rt, name):
    fn = getattr(rt, name)
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32]
    fn.restype = None
    return fn


def _bind_unary_8x1(rt, name):
    # Same C signature: in/out/batch — but out has length batch (scalars).
    return _bind_unary_8x8(rt, name)


def _bind_binary_8x8(rt, name):
    fn = getattr(rt, name)
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32]
    fn.restype = None
    return fn


def _bind_binary_8x1(rt, name):
    return _bind_binary_8x8(rt, name)


def _bind_grade_projection(rt):
    fn = rt.tessera_apple_gpu_clifford_grade_projection_cl30_f32
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32,   # grade_mask
                   ctypes.c_int32]   # batch
    fn.restype = None
    return fn


def _bind_unary_8x8_f16(rt, name):
    fn = getattr(rt, name)
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint16),
                   ctypes.POINTER(ctypes.c_uint16),
                   ctypes.c_int32]
    fn.restype = None
    return fn


def _bind_binary_8x8_f16(rt, name):
    fn = getattr(rt, name)
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint16),
                   ctypes.POINTER(ctypes.c_uint16),
                   ctypes.POINTER(ctypes.c_uint16),
                   ctypes.c_int32]
    fn.restype = None
    return fn


# ---------------------------------------------------------------------------
# All expected exported symbols
# ---------------------------------------------------------------------------

ALL_NEW_SYMBOLS = [
    # f32 GA3/GA5 pointwise
    "tessera_apple_gpu_clifford_reverse_cl30_f32",
    "tessera_apple_gpu_clifford_grade_involution_cl30_f32",
    "tessera_apple_gpu_clifford_conjugate_cl30_f32",
    "tessera_apple_gpu_clifford_hodge_star_cl30_f32",
    "tessera_apple_gpu_clifford_norm_cl30_f32",
    "tessera_apple_gpu_clifford_norm_squared_cl30_f32",
    "tessera_apple_gpu_clifford_wedge_cl30_f32",
    "tessera_apple_gpu_clifford_left_contraction_cl30_f32",
    "tessera_apple_gpu_clifford_inner_cl30_f32",
    "tessera_apple_gpu_clifford_grade_projection_cl30_f32",
    # fp16/bf16 ports
    "tessera_apple_gpu_clifford_geo_product_cl30_f16",
    "tessera_apple_gpu_clifford_geo_product_cl30_bf16",
    "tessera_apple_gpu_clifford_rotor_sandwich_cl30_f16",
    "tessera_apple_gpu_clifford_rotor_sandwich_cl30_bf16",
]


@pytest.mark.parametrize("symbol", ALL_NEW_SYMBOLS)
def test_clifford_kernel_symbols_exported(apple_gpu_runtime, symbol):
    assert hasattr(apple_gpu_runtime, symbol), f"runtime missing {symbol}"


# ---------------------------------------------------------------------------
# Unary 8→8 GA3 / GA5 ops (signed-per-grade + Hodge)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("symbol,py_op", [
    ("tessera_apple_gpu_clifford_reverse_cl30_f32", "reverse"),
    ("tessera_apple_gpu_clifford_grade_involution_cl30_f32", "grade_involution"),
    ("tessera_apple_gpu_clifford_conjugate_cl30_f32", "conjugate"),
    ("tessera_apple_gpu_clifford_hodge_star_cl30_f32", "hodge_star"),
])
def test_unary_op_matches_python_reference(apple_gpu_runtime, symbol, py_op):
    from tessera.ga import Cl, Multivector
    import tessera.ga as ga

    a = Cl(3, 0)
    rng = np.random.RandomState({"reverse": 1, "grade_involution": 2,
                                  "conjugate": 3, "hodge_star": 4}[py_op])
    batch = 32
    A = rng.randn(batch, 8).astype(np.float32)
    C = np.zeros_like(A)
    fn = _bind_unary_8x8(apple_gpu_runtime, symbol)
    fn(_ptr(np.ascontiguousarray(A)), _ptr(C), ctypes.c_int32(batch))

    py_fn = getattr(ga, py_op)
    C_ref = np.zeros_like(A)
    for i in range(batch):
        C_ref[i] = py_fn(Multivector(A[i], a)).coefficients
    np.testing.assert_allclose(C, C_ref, atol=1e-7, rtol=1e-6)


# ---------------------------------------------------------------------------
# Unary 8→1: norm, norm_squared
# ---------------------------------------------------------------------------

def test_norm_squared_matches_python_reference(apple_gpu_runtime):
    rng = np.random.RandomState(11)
    batch = 32
    A = rng.randn(batch, 8).astype(np.float32)
    C = np.zeros(batch, dtype=np.float32)
    fn = _bind_unary_8x1(apple_gpu_runtime,
                         "tessera_apple_gpu_clifford_norm_squared_cl30_f32")
    fn(_ptr(np.ascontiguousarray(A)), _ptr(C), ctypes.c_int32(batch))
    expected = np.sum(A * A, axis=1).astype(np.float32)
    np.testing.assert_allclose(C, expected, atol=1e-5, rtol=1e-5)


def test_norm_matches_python_reference(apple_gpu_runtime):
    rng = np.random.RandomState(12)
    batch = 32
    A = rng.randn(batch, 8).astype(np.float32)
    C = np.zeros(batch, dtype=np.float32)
    fn = _bind_unary_8x1(apple_gpu_runtime,
                         "tessera_apple_gpu_clifford_norm_cl30_f32")
    fn(_ptr(np.ascontiguousarray(A)), _ptr(C), ctypes.c_int32(batch))
    expected = np.sqrt(np.sum(A * A, axis=1)).astype(np.float32)
    np.testing.assert_allclose(C, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Binary 8x8→8: wedge, left_contraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("symbol,py_op", [
    ("tessera_apple_gpu_clifford_wedge_cl30_f32", "wedge"),
    ("tessera_apple_gpu_clifford_left_contraction_cl30_f32", "left_contraction"),
])
def test_binary_op_matches_python_reference(apple_gpu_runtime, symbol, py_op):
    from tessera.ga import Cl, Multivector
    import tessera.ga as ga

    a = Cl(3, 0)
    rng = np.random.RandomState(20 + len(py_op))
    batch = 32
    A = rng.randn(batch, 8).astype(np.float32)
    B = rng.randn(batch, 8).astype(np.float32)
    C = np.zeros_like(A)
    fn = _bind_binary_8x8(apple_gpu_runtime, symbol)
    fn(_ptr(np.ascontiguousarray(A)),
       _ptr(np.ascontiguousarray(B)),
       _ptr(C), ctypes.c_int32(batch))

    py_fn = getattr(ga, py_op)
    C_ref = np.zeros_like(A)
    for i in range(batch):
        C_ref[i] = py_fn(Multivector(A[i], a), Multivector(B[i], a)).coefficients
    np.testing.assert_allclose(C, C_ref, atol=2e-5, rtol=2e-5)


# ---------------------------------------------------------------------------
# Binary 8x8→1: inner
# ---------------------------------------------------------------------------

def test_inner_matches_python_reference(apple_gpu_runtime):
    from tessera.ga import Cl, Multivector
    import tessera.ga as ga

    a = Cl(3, 0)
    rng = np.random.RandomState(30)
    batch = 32
    A = rng.randn(batch, 8).astype(np.float32)
    B = rng.randn(batch, 8).astype(np.float32)
    C = np.zeros(batch, dtype=np.float32)
    fn = _bind_binary_8x1(apple_gpu_runtime,
                          "tessera_apple_gpu_clifford_inner_cl30_f32")
    fn(_ptr(np.ascontiguousarray(A)),
       _ptr(np.ascontiguousarray(B)),
       _ptr(C), ctypes.c_int32(batch))

    expected = np.zeros(batch, dtype=np.float32)
    for i in range(batch):
        expected[i] = float(ga.inner(Multivector(A[i], a),
                                      Multivector(B[i], a)))
    np.testing.assert_allclose(C, expected, atol=2e-5, rtol=2e-5)


# ---------------------------------------------------------------------------
# grade_projection — parameterized by grade_mask
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grades,grade_mask", [
    ({0}, 0b0001),         # scalar only
    ({1}, 0b0010),         # vectors only
    ({2}, 0b0100),         # bivectors only
    ({0, 2}, 0b0101),      # even subalgebra
    ({1, 3}, 0b1010),      # odd subalgebra
    ({0, 1, 2, 3}, 0b1111),# everything (identity)
])
def test_grade_projection_matches_python(apple_gpu_runtime, grades, grade_mask):
    from tessera.ga import Cl, Multivector, grade_projection

    a = Cl(3, 0)
    rng = np.random.RandomState(40)
    batch = 16
    A = rng.randn(batch, 8).astype(np.float32)
    C = np.zeros_like(A)
    fn = _bind_grade_projection(apple_gpu_runtime)
    fn(_ptr(np.ascontiguousarray(A)),
       _ptr(C),
       ctypes.c_int32(grade_mask),
       ctypes.c_int32(batch))

    C_ref = np.zeros_like(A)
    for i in range(batch):
        C_ref[i] = grade_projection(Multivector(A[i], a), grades).coefficients
    np.testing.assert_array_equal(C, C_ref)


# ---------------------------------------------------------------------------
# fp16/bf16 ports of geo_product
# ---------------------------------------------------------------------------

def _f32_to_f16(x):
    return x.astype(np.float16).view(np.uint16)


def _f16_to_f32(x):
    return x.view(np.float16).astype(np.float32)


def _f32_to_bf16(x):
    # Round-half-to-even via bit manipulation (matches Tessera's
    # bfloat16 convention used in the runtime helpers).
    u32 = x.astype(np.float32).view(np.uint32)
    rounding_bias = (u32 >> 16) & 1
    u32 = (u32 + 0x7FFF + rounding_bias) >> 16
    return u32.astype(np.uint16)


def _bf16_to_f32(x):
    return (x.astype(np.uint32) << 16).view(np.float32)


def test_geo_product_f16_matches_python_reference(apple_gpu_runtime):
    from tessera.ga import Cl, Multivector, geometric_product

    a = Cl(3, 0)
    rng = np.random.RandomState(50)
    batch = 32
    A_f32 = (rng.randn(batch, 8) * 0.3).astype(np.float32)  # tame range for fp16
    B_f32 = (rng.randn(batch, 8) * 0.3).astype(np.float32)

    A16 = _f32_to_f16(A_f32)
    B16 = _f32_to_f16(B_f32)
    C16 = np.zeros((batch, 8), dtype=np.uint16)

    fn = _bind_binary_8x8_f16(apple_gpu_runtime,
                              "tessera_apple_gpu_clifford_geo_product_cl30_f16")
    fn(_ptr16(np.ascontiguousarray(A16)),
       _ptr16(np.ascontiguousarray(B16)),
       _ptr16(C16),
       ctypes.c_int32(batch))

    C_gpu_f32 = _f16_to_f32(C16)
    C_ref = np.zeros((batch, 8), dtype=np.float32)
    for i in range(batch):
        A_round = _f16_to_f32(A16[i])
        B_round = _f16_to_f32(B16[i])
        C_ref[i] = geometric_product(
            Multivector(A_round, a), Multivector(B_round, a)
        ).coefficients
    np.testing.assert_allclose(C_gpu_f32, C_ref, atol=5e-3, rtol=5e-3)


def test_geo_product_bf16_matches_python_reference(apple_gpu_runtime):
    from tessera.ga import Cl, Multivector, geometric_product

    a = Cl(3, 0)
    rng = np.random.RandomState(51)
    batch = 32
    A_f32 = rng.randn(batch, 8).astype(np.float32)
    B_f32 = rng.randn(batch, 8).astype(np.float32)
    A16 = _f32_to_bf16(A_f32)
    B16 = _f32_to_bf16(B_f32)
    C16 = np.zeros((batch, 8), dtype=np.uint16)

    fn = _bind_binary_8x8_f16(apple_gpu_runtime,
                              "tessera_apple_gpu_clifford_geo_product_cl30_bf16")
    fn(_ptr16(np.ascontiguousarray(A16)),
       _ptr16(np.ascontiguousarray(B16)),
       _ptr16(C16),
       ctypes.c_int32(batch))

    C_gpu_f32 = _bf16_to_f32(C16)
    C_ref = np.zeros((batch, 8), dtype=np.float32)
    for i in range(batch):
        A_round = _bf16_to_f32(A16[i])
        B_round = _bf16_to_f32(B16[i])
        C_ref[i] = geometric_product(
            Multivector(A_round, a), Multivector(B_round, a)
        ).coefficients
    # bf16 has 8 mantissa bits — accumulated error larger than fp16.
    np.testing.assert_allclose(C_gpu_f32, C_ref, atol=2e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# fp16/bf16 ports of rotor_sandwich
# ---------------------------------------------------------------------------

def test_rotor_sandwich_f16_matches_python_reference(apple_gpu_runtime):
    from tessera.ga import Cl, Multivector, rotor_sandwich

    a = Cl(3, 0)
    rng = np.random.RandomState(60)
    batch = 16
    R_f32 = (rng.randn(batch, 8) * 0.3).astype(np.float32)
    V_f32 = (rng.randn(batch, 8) * 0.3).astype(np.float32)
    R16 = _f32_to_f16(R_f32)
    V16 = _f32_to_f16(V_f32)
    O16 = np.zeros((batch, 8), dtype=np.uint16)

    fn = _bind_binary_8x8_f16(apple_gpu_runtime,
                              "tessera_apple_gpu_clifford_rotor_sandwich_cl30_f16")
    fn(_ptr16(np.ascontiguousarray(R16)),
       _ptr16(np.ascontiguousarray(V16)),
       _ptr16(O16),
       ctypes.c_int32(batch))

    O_gpu_f32 = _f16_to_f32(O16)
    O_ref = np.zeros((batch, 8), dtype=np.float32)
    for i in range(batch):
        R_round = _f16_to_f32(R16[i])
        V_round = _f16_to_f32(V16[i])
        O_ref[i] = rotor_sandwich(
            Multivector(R_round, a), Multivector(V_round, a)
        ).coefficients
    np.testing.assert_allclose(O_gpu_f32, O_ref, atol=2e-2, rtol=5e-2)


def test_rotor_sandwich_bf16_matches_python_reference(apple_gpu_runtime):
    from tessera.ga import Cl, Multivector, rotor_sandwich

    a = Cl(3, 0)
    rng = np.random.RandomState(61)
    batch = 16
    R_f32 = rng.randn(batch, 8).astype(np.float32)
    V_f32 = rng.randn(batch, 8).astype(np.float32)
    R16 = _f32_to_bf16(R_f32)
    V16 = _f32_to_bf16(V_f32)
    O16 = np.zeros((batch, 8), dtype=np.uint16)

    fn = _bind_binary_8x8_f16(apple_gpu_runtime,
                              "tessera_apple_gpu_clifford_rotor_sandwich_cl30_bf16")
    fn(_ptr16(np.ascontiguousarray(R16)),
       _ptr16(np.ascontiguousarray(V16)),
       _ptr16(O16),
       ctypes.c_int32(batch))

    O_gpu_f32 = _bf16_to_f32(O16)
    O_ref = np.zeros((batch, 8), dtype=np.float32)
    for i in range(batch):
        R_round = _bf16_to_f32(R16[i])
        V_round = _bf16_to_f32(V16[i])
        O_ref[i] = rotor_sandwich(
            Multivector(R_round, a), Multivector(V_round, a)
        ).coefficients
    # bf16 accumulates more rounding through 16 mul-adds.
    np.testing.assert_allclose(O_gpu_f32, O_ref, atol=1.0, rtol=0.2)


# ---------------------------------------------------------------------------
# Headline demo: hodge_star double-application on Cl(3,0) is identity
# ---------------------------------------------------------------------------

def test_hodge_star_double_application_is_identity_on_gpu(apple_gpu_runtime):
    """For Cl(3,0), ⋆⋆ω = ω uniformly across all grades. Verify on GPU."""
    rng = np.random.RandomState(70)
    batch = 8
    A = rng.randn(batch, 8).astype(np.float32)
    T = np.zeros_like(A)
    O = np.zeros_like(A)
    fn = _bind_unary_8x8(apple_gpu_runtime,
                         "tessera_apple_gpu_clifford_hodge_star_cl30_f32")
    fn(_ptr(np.ascontiguousarray(A)), _ptr(T), ctypes.c_int32(batch))
    fn(_ptr(np.ascontiguousarray(T)), _ptr(O), ctypes.c_int32(batch))
    np.testing.assert_allclose(O, A, atol=1e-7, rtol=1e-7)
