"""GA11 conformance — final 6 GA primitives on Apple GPU.

Covers the two trig-MSL pointwise ops and four field-signature ops:
  exp_mv, log_mv                          (2 unary 8→8, pure-bivector path)
  ext_deriv, vec_deriv, codiff            (3 field 8→8, 3D grid)
  integral                                (1 field+weights → 8 reduction)

Each test compiles the runtime once (module-scoped fixture), ctypes-loads
the dylib, dispatches the kernel, and matches against the Python
`tessera.ga` reference path (interior cells only for finite-difference
field ops; the runtime zero-pads boundaries while np.gradient uses
one-sided 2nd-order, so the comparison is restricted to interior).

Total: 6 new C ABI symbols verified.
"""

from __future__ import annotations

import ctypes
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
    tmp_dir = tmp_path_factory.mktemp("apple_gpu_ga11")
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


# ---------------------------------------------------------------------------
# Expected symbol surface
# ---------------------------------------------------------------------------

GA11_SYMBOLS = [
    "tessera_apple_gpu_clifford_exp_cl30_f32",
    "tessera_apple_gpu_clifford_log_cl30_f32",
    "tessera_apple_gpu_clifford_ext_deriv_cl30_f32",
    "tessera_apple_gpu_clifford_vec_deriv_cl30_f32",
    "tessera_apple_gpu_clifford_codiff_cl30_f32",
    "tessera_apple_gpu_clifford_integral_cl30_f32",
]


@pytest.mark.parametrize("symbol", GA11_SYMBOLS)
def test_ga11_symbols_exported(apple_gpu_runtime, symbol):
    assert hasattr(apple_gpu_runtime, symbol), f"runtime missing {symbol}"


# ---------------------------------------------------------------------------
# Pointwise trig-MSL ops: exp_mv, log_mv (closed-form for Cl(3,0) bivectors)
# ---------------------------------------------------------------------------

def _bind_unary_8x8(rt, name):
    fn = getattr(rt, name)
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32]
    fn.restype = None
    return fn


def _pure_bivector_batch(rng, batch):
    """Generate a batch of pure-bivector multivectors in Cl(3,0).

    Bivector basis blade masks in Cl(3,0): e12=3, e13=5, e23=6.
    """
    A = np.zeros((batch, 8), dtype=np.float32)
    A[:, 3] = rng.randn(batch).astype(np.float32)
    A[:, 5] = rng.randn(batch).astype(np.float32)
    A[:, 6] = rng.randn(batch).astype(np.float32)
    return A


def test_exp_pure_bivector_matches_python_reference(apple_gpu_runtime):
    """exp(B) for pure bivectors uses the cos(|B|) + sin(|B|)/|B| * B closed form."""
    from tessera.ga import Cl, Multivector, exp_mv

    a = Cl(3, 0)
    rng = np.random.RandomState(200)
    batch = 32
    A = _pure_bivector_batch(rng, batch)
    C = np.zeros_like(A)
    fn = _bind_unary_8x8(apple_gpu_runtime,
                         "tessera_apple_gpu_clifford_exp_cl30_f32")
    fn(_ptr(np.ascontiguousarray(A)), _ptr(C), ctypes.c_int32(batch))

    C_ref = np.zeros_like(A)
    for i in range(batch):
        C_ref[i] = exp_mv(Multivector(A[i], a)).coefficients
    np.testing.assert_allclose(C, C_ref, atol=2e-5, rtol=2e-5)


def test_exp_zero_bivector_returns_identity(apple_gpu_runtime):
    """exp(0) = 1 (scalar part only). Closed-form path must handle |B|=0."""
    fn = _bind_unary_8x8(apple_gpu_runtime,
                         "tessera_apple_gpu_clifford_exp_cl30_f32")
    A = np.zeros((4, 8), dtype=np.float32)
    C = np.zeros_like(A)
    fn(_ptr(A), _ptr(C), ctypes.c_int32(4))
    expected = np.zeros_like(A)
    expected[:, 0] = 1.0
    np.testing.assert_allclose(C, expected, atol=1e-7, rtol=1e-7)


def test_log_unit_rotor_matches_python_reference(apple_gpu_runtime):
    """log(R) for unit rotors (scalar+bivector) returns pure bivector axis."""
    from tessera.ga import Cl, Multivector, log_mv

    a = Cl(3, 0)
    rng = np.random.RandomState(201)
    batch = 16
    # Build rotors as exp(B/2) where B is a random pure bivector.
    B = _pure_bivector_batch(rng, batch) * 0.5
    Bnorm = np.sqrt(B[:, 3] ** 2 + B[:, 5] ** 2 + B[:, 6] ** 2)
    safe = np.where(Bnorm > 1e-12, Bnorm, 1.0)
    R = np.zeros_like(B)
    R[:, 0] = np.cos(Bnorm)
    sin_scaled = (np.sin(Bnorm) / safe)[:, None]
    R[:, 3:8] = B[:, 3:8] * sin_scaled
    R = R.astype(np.float32)

    C = np.zeros_like(R)
    fn = _bind_unary_8x8(apple_gpu_runtime,
                         "tessera_apple_gpu_clifford_log_cl30_f32")
    fn(_ptr(np.ascontiguousarray(R)), _ptr(C), ctypes.c_int32(batch))

    C_ref = np.zeros_like(R)
    for i in range(batch):
        C_ref[i] = log_mv(Multivector(R[i], a)).coefficients
    np.testing.assert_allclose(C, C_ref, atol=5e-5, rtol=5e-5)


def test_log_exp_round_trip_on_bivector(apple_gpu_runtime):
    """log(exp(B)) ≈ B for pure bivectors (within rotor branch)."""
    rng = np.random.RandomState(202)
    batch = 16
    # Keep |B| < π/2 so log is in principal branch.
    B = _pure_bivector_batch(rng, batch) * 0.4
    expB = np.zeros_like(B)
    fn_exp = _bind_unary_8x8(apple_gpu_runtime,
                             "tessera_apple_gpu_clifford_exp_cl30_f32")
    fn_log = _bind_unary_8x8(apple_gpu_runtime,
                             "tessera_apple_gpu_clifford_log_cl30_f32")
    fn_exp(_ptr(np.ascontiguousarray(B)), _ptr(expB), ctypes.c_int32(batch))
    out = np.zeros_like(B)
    fn_log(_ptr(np.ascontiguousarray(expB)), _ptr(out), ctypes.c_int32(batch))
    np.testing.assert_allclose(out, B, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Field-signature ops on a 3D grid
# ---------------------------------------------------------------------------

def _bind_field_op(rt, name):
    fn = getattr(rt, name)
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                   ctypes.c_float, ctypes.c_float, ctypes.c_float]
    fn.restype = None
    return fn


def _interior_slice(D0, D1, D2):
    """Return a mask selecting interior cells (excludes 1-cell boundary)."""
    return np.s_[1:D0 - 1, 1:D1 - 1, 1:D2 - 1]


@pytest.mark.parametrize("symbol,py_op", [
    ("tessera_apple_gpu_clifford_ext_deriv_cl30_f32", "ext_deriv"),
    ("tessera_apple_gpu_clifford_vec_deriv_cl30_f32", "vec_deriv"),
])
def test_field_op_interior_matches_python_reference(
        apple_gpu_runtime, symbol, py_op):
    from tessera.ga import Cl
    from tessera.ga.calculus import MultivectorField
    import tessera.ga.calculus as cal

    a = Cl(3, 0)
    rng = np.random.RandomState(300 + len(py_op))
    D0, D1, D2 = 5, 6, 7
    F = rng.randn(D0, D1, D2, 8).astype(np.float32)
    Out = np.zeros_like(F)
    h0, h1, h2 = 0.1, 0.2, 0.25

    fn = _bind_field_op(apple_gpu_runtime, symbol)
    fn(_ptr(np.ascontiguousarray(F)),
       _ptr(Out),
       ctypes.c_int32(D0), ctypes.c_int32(D1), ctypes.c_int32(D2),
       ctypes.c_float(h0), ctypes.c_float(h1), ctypes.c_float(h2))

    field = MultivectorField(F.astype(np.float64), a, spacing=(h0, h1, h2))
    py_fn = getattr(cal, py_op)
    Out_ref = py_fn(field).values.astype(np.float32)

    sl = _interior_slice(D0, D1, D2)
    np.testing.assert_allclose(Out[sl], Out_ref[sl], atol=1e-4, rtol=1e-4)


def test_codiff_interior_matches_python_reference(apple_gpu_runtime):
    """codiff = ⋆d⋆ — sequential 3-stage MSL composition on the grid."""
    from tessera.ga import Cl
    from tessera.ga.calculus import MultivectorField, codiff

    a = Cl(3, 0)
    rng = np.random.RandomState(310)
    D0, D1, D2 = 5, 6, 7
    F = rng.randn(D0, D1, D2, 8).astype(np.float32)
    Out = np.zeros_like(F)
    h0, h1, h2 = 0.1, 0.15, 0.2

    fn = _bind_field_op(apple_gpu_runtime,
                        "tessera_apple_gpu_clifford_codiff_cl30_f32")
    fn(_ptr(np.ascontiguousarray(F)),
       _ptr(Out),
       ctypes.c_int32(D0), ctypes.c_int32(D1), ctypes.c_int32(D2),
       ctypes.c_float(h0), ctypes.c_float(h1), ctypes.c_float(h2))

    field = MultivectorField(F.astype(np.float64), a, spacing=(h0, h1, h2))
    Out_ref = codiff(field).values.astype(np.float32)

    sl = _interior_slice(D0, D1, D2)
    np.testing.assert_allclose(Out[sl], Out_ref[sl], atol=1e-4, rtol=1e-4)


def test_field_ops_zero_at_corner_cells(apple_gpu_runtime):
    """At corner cells (all 3 axes at boundary) ext_deriv contributes nothing.

    The kernel skips per-axis stencils whose axis index is at the grid
    boundary, so a corner cell — boundary on every axis — receives no
    contributions from any axis and stays at zero.
    """
    rng = np.random.RandomState(320)
    D0, D1, D2 = 4, 4, 4
    F = rng.randn(D0, D1, D2, 8).astype(np.float32)
    Out = np.zeros_like(F)
    fn = _bind_field_op(apple_gpu_runtime,
                        "tessera_apple_gpu_clifford_ext_deriv_cl30_f32")
    fn(_ptr(np.ascontiguousarray(F)), _ptr(Out),
       ctypes.c_int32(D0), ctypes.c_int32(D1), ctypes.c_int32(D2),
       ctypes.c_float(0.1), ctypes.c_float(0.1), ctypes.c_float(0.1))
    for i in (0, D0 - 1):
        for j in (0, D1 - 1):
            for k in (0, D2 - 1):
                assert np.all(Out[i, j, k] == 0), f"corner ({i},{j},{k}) nonzero"


# ---------------------------------------------------------------------------
# integral — weighted Riemann sum (n × 8 field + n weights → 8 output)
# ---------------------------------------------------------------------------

def test_integral_matches_weighted_sum(apple_gpu_runtime):
    rt = apple_gpu_runtime
    fn = rt.tessera_apple_gpu_clifford_integral_cl30_f32
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32]
    fn.restype = None

    rng = np.random.RandomState(400)
    n = 64
    field = rng.randn(n, 8).astype(np.float32)
    weights = (rng.randn(n) ** 2 + 0.1).astype(np.float32)  # positive-ish weights
    out = np.zeros(8, dtype=np.float32)
    fn(_ptr(np.ascontiguousarray(field)),
       _ptr(np.ascontiguousarray(weights)),
       _ptr(out),
       ctypes.c_int32(n))

    expected = (weights[:, None] * field).sum(axis=0)
    np.testing.assert_allclose(out, expected, atol=5e-4, rtol=5e-4)


def test_integral_with_euclidean_manifold_weights(apple_gpu_runtime):
    """Use the Python Euclidean manifold to derive cell-volume weights,
    then verify the GPU kernel reproduces the weighted-sum semantics
    that `tessera.ga.integral` uses internally."""
    from tessera.ga.manifold import Euclidean

    n = 16
    rng = np.random.RandomState(401)
    field = rng.randn(n, 8).astype(np.float32)

    manifold = Euclidean(bounds=[(0.0, 1.0)], resolution=n)
    weights = manifold.weights().astype(np.float32)
    assert weights.shape == (n,)

    rt = apple_gpu_runtime
    fn = rt.tessera_apple_gpu_clifford_integral_cl30_f32
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32]
    fn.restype = None
    out = np.zeros(8, dtype=np.float32)
    fn(_ptr(np.ascontiguousarray(field)),
       _ptr(np.ascontiguousarray(weights)),
       _ptr(out),
       ctypes.c_int32(n))
    expected = (weights[:, None] * field).sum(axis=0)
    np.testing.assert_allclose(out, expected, atol=5e-4, rtol=5e-4)
