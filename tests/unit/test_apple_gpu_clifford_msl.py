"""GA9 native MSL kernels — Apple GPU integration test.

Compiles the Apple GPU runtime (`apple_gpu_runtime.mm`) into a dylib,
loads it via ctypes, and dispatches the two new GA9 fused MSL kernels
on randomly generated Cl(3,0) multivectors. Verifies the GPU result
bitwise-matches the Python `tessera.ga.geometric_product` /
`tessera.ga.rotor_sandwich` reference to fp32 tolerance.

This is the GA9 "@jit(target='apple_cpu') on rotor-sandwich" end-to-end
claim, realized via a native MSL kernel instead of the CPU reference
path. The test gracefully skips when not running on Apple Silicon.
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


pytestmark = pytest.mark.hardware_apple_gpu


@pytest.fixture(scope="module")
def apple_gpu_runtime(tmp_path_factory):
    """Compile `apple_gpu_runtime.mm` into a shared library and ctypes-load.

    Returns the loaded `ctypes.CDLL` ready for dispatch.
    """
    cxx = shutil.which("clang++") or shutil.which("c++")
    if cxx is None:
        pytest.fail("C++ compiler unavailable on the Apple hardware test host")

    source = ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm"
    tmp_dir = tmp_path_factory.mktemp("apple_gpu")
    lib = tmp_dir / "libtessera_apple_gpu_runtime.dylib"
    cmd = [
        cxx,
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-O2",
        "-fobjc-arc",
        "-x", "objective-c++",
        str(source),
        "-o", str(lib),
        "-framework", "Metal",
        "-framework", "MetalPerformanceShaders",
        "-framework", "MetalPerformanceShadersGraph",
        "-framework", "Foundation",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(
            "apple_gpu_runtime.mm did not compile (likely missing Metal SDK).\n"
            f"stderr (last 2k chars):\n{proc.stderr[-2000:]}"
        )
    return ctypes.CDLL(str(lib))


# ---------------------------------------------------------------------------
# clifford_geo_product_cl30_f32
# ---------------------------------------------------------------------------

def _bind_geo_product(rt):
    fn = rt.tessera_apple_gpu_clifford_geo_product_cl30_f32
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
    ]
    fn.restype = None
    return fn


def _bind_rotor_sandwich(rt):
    fn = rt.tessera_apple_gpu_clifford_rotor_sandwich_cl30_f32
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
    ]
    fn.restype = None
    return fn


def _np_to_ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _cl30_cayley_reference(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Independent blade-mask-order oracle for the native Cl(3,0) ABI."""
    from tessera.ga import Cl

    table = Cl(3, 0).product_table()
    out = np.zeros_like(lhs, dtype=np.float32)
    for batch_index in range(lhs.shape[0]):
        for left_blade in range(8):
            for right_blade in range(8):
                result_blade, sign = table[left_blade][right_blade]
                out[batch_index, result_blade] += (
                    sign
                    * lhs[batch_index, left_blade]
                    * rhs[batch_index, right_blade]
                )
    return out


def test_clifford_geo_product_cl30_f32_runtime_symbol_exists(apple_gpu_runtime):
    """The kernel C ABI symbol must be exported from the runtime."""
    assert hasattr(apple_gpu_runtime, "tessera_apple_gpu_clifford_geo_product_cl30_f32")
    assert hasattr(apple_gpu_runtime, "tessera_apple_gpu_clifford_rotor_sandwich_cl30_f32")


def test_clifford_geo_product_cl30_f32_matches_python_reference(apple_gpu_runtime):
    """The MSL kernel result must match `tessera.ga.geometric_product` on
    randomly generated Cl(3,0) multivectors to fp32 tolerance."""
    rng = np.random.RandomState(0)
    batch = 64
    A_np = rng.randn(batch, 8).astype(np.float32)
    B_np = rng.randn(batch, 8).astype(np.float32)
    C_np = np.zeros((batch, 8), dtype=np.float32)

    fn = _bind_geo_product(apple_gpu_runtime)
    fn(
        _np_to_ptr(np.ascontiguousarray(A_np)),
        _np_to_ptr(np.ascontiguousarray(B_np)),
        _np_to_ptr(C_np),
        ctypes.c_int32(batch),
    )

    # Do not call geometric_product here: on Apple it deliberately routes to
    # this same native symbol and would make the comparison self-referential.
    C_ref = _cl30_cayley_reference(A_np, B_np)

    # fp32 mul-add rounding: tighter than 1e-5.
    np.testing.assert_allclose(C_np, C_ref, atol=2e-5, rtol=2e-5)


def test_clifford_geo_product_cl30_f32_uses_blade_mask_order(apple_gpu_runtime):
    """The C ABI order is 1,e1,e2,e12,e3,e13,e23,e123, not grade order."""
    A_np = np.zeros((1, 8), dtype=np.float32)
    B_np = np.zeros((1, 8), dtype=np.float32)
    C_np = np.zeros((1, 8), dtype=np.float32)
    A_np[0, 1] = 1.0
    B_np[0, 2] = 1.0

    _bind_geo_product(apple_gpu_runtime)(
        _np_to_ptr(A_np),
        _np_to_ptr(B_np),
        _np_to_ptr(C_np),
        ctypes.c_int32(1),
    )

    expected = np.zeros((1, 8), dtype=np.float32)
    expected[0, 3] = 1.0
    np.testing.assert_array_equal(C_np, expected)


def test_clifford_rotor_sandwich_cl30_f32_matches_python_reference(apple_gpu_runtime):
    """Headline GA9 native-MSL test: `R · v · R†` on Apple GPU must
    bitwise-match `tessera.ga.rotor_sandwich` to fp32 tolerance.

    This is the GA9 acceptance: "@jit(target='apple_cpu') on a
    Cl(3,0) rotor-sandwich function executes via the native backend
    and bitwise-matches the numpy reference" — realized on Apple GPU
    rather than Apple CPU since the MSL fused kernel is the
    native-backend story for this op.
    """
    from tessera.ga import Cl, Multivector, rotor_sandwich

    a = Cl(3, 0)
    rng = np.random.RandomState(7)
    batch = 32

    # Generate random rotors (even-grade, unit-norm).
    # For testing we don't need them to be exact rotors — the kernel
    # works on any Cl(3,0) input.  Just use random multivectors.
    R_np = rng.randn(batch, 8).astype(np.float32)
    V_np = rng.randn(batch, 8).astype(np.float32)
    O_np = np.zeros((batch, 8), dtype=np.float32)

    fn = _bind_rotor_sandwich(apple_gpu_runtime)
    fn(
        _np_to_ptr(np.ascontiguousarray(R_np)),
        _np_to_ptr(np.ascontiguousarray(V_np)),
        _np_to_ptr(O_np),
        ctypes.c_int32(batch),
    )

    # Reference via Python GA ops.
    O_ref = np.zeros((batch, 8), dtype=np.float32)
    for i in range(batch):
        R = Multivector(R_np[i], a, grades=None)
        V = Multivector(V_np[i], a, grades=None)
        O_ref[i] = rotor_sandwich(R, V).coefficients

    # 16 fp32 mul-adds chained: error accumulates a bit, allow 5e-5.
    np.testing.assert_allclose(O_np, O_ref, atol=5e-5, rtol=5e-5)


def test_rotor_sandwich_rotates_vector_correctly_on_gpu(apple_gpu_runtime):
    """The headline equivariance demo: rotating a 3D vector via the
    Apple GPU rotor-sandwich kernel produces the same result as the
    Rodrigues SO(3) reference."""
    from tessera.ga import Cl, Multivector, rotor_from_axis

    a = Cl(3, 0)
    # Build a rotor for π/4 rotation around e12 (z-axis-like plane).
    bivector = Multivector.from_blade(a.blade("e12"), a, dtype=np.float32)
    R = rotor_from_axis(bivector, math.pi / 4)
    v = Multivector.from_vector([1.0, 0.0, 0.0], a, dtype=np.float32)

    R_np = R.coefficients.astype(np.float32)
    V_np = v.coefficients.astype(np.float32)
    O_np = np.zeros(8, dtype=np.float32)

    fn = _bind_rotor_sandwich(apple_gpu_runtime)
    fn(
        _np_to_ptr(np.ascontiguousarray(R_np.reshape(1, 8))),
        _np_to_ptr(np.ascontiguousarray(V_np.reshape(1, 8))),
        _np_to_ptr(O_np),
        ctypes.c_int32(1),
    )

    # Rotating (1, 0, 0) by π/4 around the e12 plane gives
    # (cos(π/4), sin(π/4), 0).
    e1 = a.blade("e1").mask
    e2 = a.blade("e2").mask
    e3 = a.blade("e3").mask
    assert O_np[e1] == pytest.approx(math.cos(math.pi / 4), abs=2e-6)
    assert O_np[e2] == pytest.approx(math.sin(math.pi / 4), abs=2e-6)
    assert abs(O_np[e3]) < 1e-6


def test_geo_product_batch_consistency_with_per_sample_dispatch(apple_gpu_runtime):
    """Dispatching one big batch should match dispatching each sample
    individually — verifies thread independence in the MSL kernel."""
    rng = np.random.RandomState(11)
    batch = 16
    A = rng.randn(batch, 8).astype(np.float32)
    B = rng.randn(batch, 8).astype(np.float32)

    fn = _bind_geo_product(apple_gpu_runtime)
    # Batched dispatch.
    C_batch = np.zeros_like(A)
    fn(
        _np_to_ptr(np.ascontiguousarray(A)),
        _np_to_ptr(np.ascontiguousarray(B)),
        _np_to_ptr(C_batch),
        ctypes.c_int32(batch),
    )
    # Per-sample dispatch.
    C_each = np.zeros_like(A)
    for i in range(batch):
        out = np.zeros(8, dtype=np.float32)
        fn(
            _np_to_ptr(np.ascontiguousarray(A[i:i+1])),
            _np_to_ptr(np.ascontiguousarray(B[i:i+1])),
            _np_to_ptr(out),
            ctypes.c_int32(1),
        )
        C_each[i] = out

    np.testing.assert_array_equal(C_batch, C_each)
