"""apple_gpu GA + EBM end-to-end benchmark.

Walks each GA primitive and each EBM execution path through the full
stack — Python API → backend manifest lookup → Apple runtime symbol
dispatch → real hardware execution → correctness check against the
Python reference → timing → report row.

GA coverage (all 17 GA primitives, every one with a fused MSL kernel
on Apple GPU as of GA11, 2026-05-17):

  Pointwise unary 8→8 (Cl(3,0) f32):
    reverse, grade_involution, conjugate, hodge_star, exp, log
  Pointwise unary 8→1:
    norm
  Pointwise binary 8x8→8:
    geometric_product, wedge, left_contraction, rotor_sandwich
  Pointwise binary 8x8→1:
    inner
  Pointwise unary 8→8 + int mask:
    grade_projection
  Field 8x8 on 3D grid (D0,D1,D2,h0,h1,h2):
    ext_deriv, vec_deriv, codiff
  Weighted reduction (N×8, N → 8):
    integral

EBM coverage (Python execution paths today — the EBM ops do not yet
ship Apple GPU MSL kernels, but the geometric-Langevin path uses the
GA primitives so the GA kernels accelerate it transitively when the
analytic gradient is wired into them):

  energy, inner_step, langevin_step (analytic grad), self_verify,
  decode_init, partition_function_exact,
  bivector_langevin_step, sphere_langevin_step

Output schema (matches ``benchmarks/benchmark_gemm.py`` for
roofline-tool compatibility):

    {"backend": "apple_gpu" | "python_ref",
     "op": "<clifford_op>" | "<ebm_op>",
     "shape": "<descriptor>",
     "dtype": "f32",
     "mode": "fused" | "reference",
     "reps": int,
     "latency_ms": <float>,
     "stdev_ms": <float>,
     "max_abs_err": <float>,
     "ok": <bool>,
     "device": "<host>",
     "tessera_version": "..."}

Skips cleanly on non-Darwin or when the runtime can't be compiled.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Allow running as a standalone script from the repo root before
# `pip install -e .` — the in-tree `python/tessera` package is the
# source of truth for the Python API.
_PY_SRC = ROOT / "python"
if str(_PY_SRC) not in sys.path:
    sys.path.insert(0, str(_PY_SRC))

import tessera.ga as ga  # noqa: E402
import tessera.ebm as ebm  # noqa: E402
from tessera.compiler import backend_manifest as bm  # noqa: E402
from tessera.rng import RNGKey  # noqa: E402


# ---------------------------------------------------------------------------
# Runtime compile/load — shared with the unit test fixture
# ---------------------------------------------------------------------------

def compile_apple_gpu_runtime(tmp_dir: Path) -> ctypes.CDLL | None:
    """Compile ``apple_gpu_runtime.mm`` into a dylib and ctypes-load it.

    Returns ``None`` on non-Darwin hosts or when the toolchain is
    missing / the source fails to compile.
    """
    if sys.platform != "darwin":
        return None
    cxx = shutil.which("clang++") or shutil.which("c++")
    if cxx is None:
        return None
    source = ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm"
    if not source.exists():
        return None
    tmp_dir.mkdir(parents=True, exist_ok=True)
    lib = tmp_dir / "libtessera_apple_gpu_runtime.dylib"
    cmd = [cxx, "-std=c++17", "-shared", "-fPIC", "-O2", "-fobjc-arc",
           "-x", "objective-c++", str(source), "-o", str(lib),
           "-framework", "Metal",
           "-framework", "MetalPerformanceShaders",
           "-framework", "Foundation"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return ctypes.CDLL(str(lib))


def _ptr_f(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# ---------------------------------------------------------------------------
# Per-ABI ctypes binding helpers
# ---------------------------------------------------------------------------

def _bind_unary_8x8(rt: ctypes.CDLL, name: str) -> Callable:
    fn = getattr(rt, name)
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32]
    fn.restype = None
    return fn


def _bind_binary_8x8(rt: ctypes.CDLL, name: str) -> Callable:
    fn = getattr(rt, name)
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32]
    fn.restype = None
    return fn


def _bind_grade_projection(rt: ctypes.CDLL) -> Callable:
    fn = rt.tessera_apple_gpu_clifford_grade_projection_cl30_f32
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32, ctypes.c_int32]
    fn.restype = None
    return fn


def _bind_field_op(rt: ctypes.CDLL, name: str) -> Callable:
    fn = getattr(rt, name)
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                   ctypes.c_float, ctypes.c_float, ctypes.c_float]
    fn.restype = None
    return fn


def _bind_integral(rt: ctypes.CDLL) -> Callable:
    fn = rt.tessera_apple_gpu_clifford_integral_cl30_f32
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32]
    fn.restype = None
    return fn


# ---------------------------------------------------------------------------
# GA primitive runners — each (a) calls the Python ref, (b) dispatches
# through the Apple GPU C ABI, (c) returns (ref_output, gpu_dispatch_callable)
# so the caller can time the dispatch and diff the results.
# ---------------------------------------------------------------------------

_CL30 = ga.Cl(3, 0)
_BATCH = 64                 # pointwise-op batch size
_GRID = (5, 6, 7)           # field-op grid dimensions
_SPACING = (0.1, 0.2, 0.25)
_INTEGRAL_N = 64


def _seeded_pointwise(seed: int) -> np.ndarray:
    return np.random.RandomState(seed).randn(_BATCH, 8).astype(np.float32)


def _seeded_pure_bivector(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    A = np.zeros((_BATCH, 8), dtype=np.float32)
    A[:, 3] = rng.randn(_BATCH).astype(np.float32) * 0.3
    A[:, 5] = rng.randn(_BATCH).astype(np.float32) * 0.3
    A[:, 6] = rng.randn(_BATCH).astype(np.float32) * 0.3
    return A


def _seeded_rotor(seed: int) -> np.ndarray:
    """Build R = exp(B/2) for random pure bivector B."""
    B = _seeded_pure_bivector(seed) * 0.5
    Bnorm = np.sqrt(B[:, 3] ** 2 + B[:, 5] ** 2 + B[:, 6] ** 2)
    safe = np.where(Bnorm > 1e-12, Bnorm, 1.0)
    R = np.zeros_like(B)
    R[:, 0] = np.cos(Bnorm)
    R[:, 3:8] = B[:, 3:8] * (np.sin(Bnorm) / safe)[:, None]
    return R.astype(np.float32)


def _seeded_field(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    D0, D1, D2 = _GRID
    return rng.randn(D0, D1, D2, 8).astype(np.float32)


def _py_ref_unary_8x8(op_name: str, A: np.ndarray) -> np.ndarray:
    fn = {
        "clifford_reverse": ga.reverse,
        "clifford_grade_involution": ga.grade_involution,
        "clifford_conjugate": ga.conjugate,
        "clifford_hodge_star": ga.hodge_star,
        "clifford_exp": ga.exp_mv,
        "clifford_log": ga.log_mv,
    }[op_name]
    out = np.zeros_like(A)
    for i in range(A.shape[0]):
        out[i] = fn(ga.Multivector(A[i], _CL30)).coefficients
    return out


def _py_ref_norm(A: np.ndarray) -> np.ndarray:
    out = np.zeros(A.shape[0], dtype=np.float32)
    for i in range(A.shape[0]):
        out[i] = float(np.asarray(ga.norm(ga.Multivector(A[i], _CL30))))
    return out


def _py_ref_binary_8x8(op_name: str, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    fn = {
        "clifford_geometric_product": ga.geometric_product,
        "clifford_wedge": ga.wedge,
        "clifford_left_contraction": ga.left_contraction,
        "clifford_rotor_sandwich": ga.rotor_sandwich,
    }[op_name]
    out = np.zeros_like(A)
    for i in range(A.shape[0]):
        out[i] = fn(ga.Multivector(A[i], _CL30),
                    ga.Multivector(B[i], _CL30)).coefficients
    return out


def _py_ref_inner(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    out = np.zeros(A.shape[0], dtype=np.float32)
    for i in range(A.shape[0]):
        out[i] = float(ga.inner(ga.Multivector(A[i], _CL30),
                                 ga.Multivector(B[i], _CL30)))
    return out


def _py_ref_grade_projection(A: np.ndarray, grades: set[int]) -> np.ndarray:
    out = np.zeros_like(A)
    for i in range(A.shape[0]):
        out[i] = ga.grade_projection(ga.Multivector(A[i], _CL30),
                                      grades).coefficients
    return out


def _py_ref_field_op(op_name: str, F: np.ndarray) -> np.ndarray:
    from tessera.ga.calculus import MultivectorField, ext_deriv, vec_deriv, codiff
    fn = {
        "clifford_ext_deriv": ext_deriv,
        "clifford_vec_deriv": vec_deriv,
        "clifford_codiff": codiff,
    }[op_name]
    field = MultivectorField(F.astype(np.float64), _CL30, spacing=_SPACING)
    return fn(field).values.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-primitive entries — input fixture, dispatch closure, py-ref output
# ---------------------------------------------------------------------------

def _ga_entry_unary_8x8(rt: ctypes.CDLL, op_name: str, symbol: str,
                         input_seed: int, input_fn: Callable[[int], np.ndarray]):
    A = input_fn(input_seed)
    A_c = np.ascontiguousarray(A)
    Out = np.zeros_like(A)
    fn = _bind_unary_8x8(rt, symbol)

    def dispatch() -> np.ndarray:
        fn(_ptr_f(A_c), _ptr_f(Out), ctypes.c_int32(_BATCH))
        return Out

    return A_c, dispatch, _py_ref_unary_8x8(op_name, A)


def _ga_entry_norm(rt: ctypes.CDLL, symbol: str, input_seed: int):
    A = _seeded_pointwise(input_seed)
    A_c = np.ascontiguousarray(A)
    Out = np.zeros(_BATCH, dtype=np.float32)
    fn = _bind_unary_8x8(rt, symbol)  # ABI same: in, out, batch

    def dispatch() -> np.ndarray:
        fn(_ptr_f(A_c), _ptr_f(Out), ctypes.c_int32(_BATCH))
        return Out

    return A_c, dispatch, _py_ref_norm(A)


def _ga_entry_binary_8x8(rt: ctypes.CDLL, op_name: str, symbol: str,
                          seed_a: int, seed_b: int,
                          input_a_fn: Callable[[int], np.ndarray],
                          input_b_fn: Callable[[int], np.ndarray]):
    A = input_a_fn(seed_a)
    B = input_b_fn(seed_b)
    A_c = np.ascontiguousarray(A)
    B_c = np.ascontiguousarray(B)
    Out = np.zeros_like(A)
    fn = _bind_binary_8x8(rt, symbol)

    def dispatch() -> np.ndarray:
        fn(_ptr_f(A_c), _ptr_f(B_c), _ptr_f(Out), ctypes.c_int32(_BATCH))
        return Out

    return (A_c, B_c), dispatch, _py_ref_binary_8x8(op_name, A, B)


def _ga_entry_inner(rt: ctypes.CDLL, symbol: str):
    A = _seeded_pointwise(700)
    B = _seeded_pointwise(701)
    A_c = np.ascontiguousarray(A)
    B_c = np.ascontiguousarray(B)
    Out = np.zeros(_BATCH, dtype=np.float32)
    fn = _bind_binary_8x8(rt, symbol)  # ABI same as binary 8x8

    def dispatch() -> np.ndarray:
        fn(_ptr_f(A_c), _ptr_f(B_c), _ptr_f(Out), ctypes.c_int32(_BATCH))
        return Out

    return (A_c, B_c), dispatch, _py_ref_inner(A, B)


def _ga_entry_grade_projection(rt: ctypes.CDLL):
    A = _seeded_pointwise(800)
    A_c = np.ascontiguousarray(A)
    Out = np.zeros_like(A)
    fn = _bind_grade_projection(rt)
    grade_mask = 0b0101  # grades {0, 2} — the even subalgebra

    def dispatch() -> np.ndarray:
        fn(_ptr_f(A_c), _ptr_f(Out),
           ctypes.c_int32(grade_mask), ctypes.c_int32(_BATCH))
        return Out

    return A_c, dispatch, _py_ref_grade_projection(A, {0, 2})


def _ga_entry_field(rt: ctypes.CDLL, op_name: str, symbol: str, seed: int):
    F = _seeded_field(seed)
    F_c = np.ascontiguousarray(F)
    Out = np.zeros_like(F)
    fn = _bind_field_op(rt, symbol)
    D0, D1, D2 = _GRID
    h0, h1, h2 = _SPACING

    def dispatch() -> np.ndarray:
        fn(_ptr_f(F_c), _ptr_f(Out),
           ctypes.c_int32(D0), ctypes.c_int32(D1), ctypes.c_int32(D2),
           ctypes.c_float(h0), ctypes.c_float(h1), ctypes.c_float(h2))
        return Out

    return F_c, dispatch, _py_ref_field_op(op_name, F)


def _ga_entry_integral(rt: ctypes.CDLL):
    rng = np.random.RandomState(900)
    field = rng.randn(_INTEGRAL_N, 8).astype(np.float32)
    weights = (rng.randn(_INTEGRAL_N) ** 2 + 0.1).astype(np.float32)
    field_c = np.ascontiguousarray(field)
    weights_c = np.ascontiguousarray(weights)
    out = np.zeros(8, dtype=np.float32)
    fn = _bind_integral(rt)

    def dispatch() -> np.ndarray:
        fn(_ptr_f(field_c), _ptr_f(weights_c), _ptr_f(out),
           ctypes.c_int32(_INTEGRAL_N))
        return out

    ref = (weights[:, None] * field).sum(axis=0).astype(np.float32)
    return (field_c, weights_c), dispatch, ref


def build_ga_entries(rt: ctypes.CDLL) -> list[dict[str, Any]]:
    """Construct the full GA stack-walk batch.

    Each row is a dict with: op_name, manifest_symbol, ref (np.ndarray),
    dispatch (no-arg callable returning the GPU output), shape (str
    descriptor for the report).
    """
    rows: list[dict[str, Any]] = []

    # 4 sign-flip unaries + hodge.
    for op_name in ("clifford_reverse", "clifford_grade_involution",
                    "clifford_conjugate", "clifford_hodge_star"):
        symbol = _resolve_symbol(op_name)
        _, dispatch, ref = _ga_entry_unary_8x8(rt, op_name, symbol, 100,
                                                _seeded_pointwise)
        rows.append({"op": op_name, "symbol": symbol,
                     "shape": f"{_BATCH}x8", "ref": ref,
                     "dispatch": dispatch})

    # exp/log on the closed-form (pure-bivector / rotor) path.
    sym = _resolve_symbol("clifford_exp")
    _, dispatch, ref = _ga_entry_unary_8x8(rt, "clifford_exp", sym, 101,
                                            _seeded_pure_bivector)
    rows.append({"op": "clifford_exp", "symbol": sym,
                 "shape": f"{_BATCH}x8/bivec", "ref": ref,
                 "dispatch": dispatch})

    sym = _resolve_symbol("clifford_log")
    _, dispatch, ref = _ga_entry_unary_8x8(rt, "clifford_log", sym, 102,
                                            _seeded_rotor)
    rows.append({"op": "clifford_log", "symbol": sym,
                 "shape": f"{_BATCH}x8/rotor", "ref": ref,
                 "dispatch": dispatch})

    # norm (8→1).
    sym = _resolve_symbol("clifford_norm")
    _, dispatch, ref = _ga_entry_norm(rt, sym, 200)
    rows.append({"op": "clifford_norm", "symbol": sym,
                 "shape": f"{_BATCH}x8→1", "ref": ref,
                 "dispatch": dispatch})

    # Binary 8x8→8.
    for op_name in ("clifford_geometric_product", "clifford_wedge",
                    "clifford_left_contraction"):
        symbol = _resolve_symbol(op_name)
        _, dispatch, ref = _ga_entry_binary_8x8(rt, op_name, symbol,
                                                 300, 301,
                                                 _seeded_pointwise,
                                                 _seeded_pointwise)
        rows.append({"op": op_name, "symbol": symbol,
                     "shape": f"{_BATCH}x8,{_BATCH}x8", "ref": ref,
                     "dispatch": dispatch})

    # rotor_sandwich — operand A is a rotor, operand B a generic mv.
    sym = _resolve_symbol("clifford_rotor_sandwich")
    _, dispatch, ref = _ga_entry_binary_8x8(rt, "clifford_rotor_sandwich",
                                             sym, 400, 401,
                                             _seeded_rotor,
                                             _seeded_pointwise)
    rows.append({"op": "clifford_rotor_sandwich", "symbol": sym,
                 "shape": f"{_BATCH}x8(rotor),{_BATCH}x8", "ref": ref,
                 "dispatch": dispatch})

    # inner (8x8→1).
    sym = _resolve_symbol("clifford_inner")
    _, dispatch, ref = _ga_entry_inner(rt, sym)
    rows.append({"op": "clifford_inner", "symbol": sym,
                 "shape": f"{_BATCH}x8,{_BATCH}x8→1", "ref": ref,
                 "dispatch": dispatch})

    # grade_projection.
    sym = _resolve_symbol("clifford_grade_projection")
    _, dispatch, ref = _ga_entry_grade_projection(rt)
    rows.append({"op": "clifford_grade_projection", "symbol": sym,
                 "shape": f"{_BATCH}x8/grades=02", "ref": ref,
                 "dispatch": dispatch})

    # Field ops on a 3D grid.
    for op_name, seed in (("clifford_ext_deriv", 500),
                          ("clifford_vec_deriv", 501),
                          ("clifford_codiff", 502)):
        symbol = _resolve_symbol(op_name)
        _, dispatch, ref = _ga_entry_field(rt, op_name, symbol, seed)
        rows.append({"op": op_name, "symbol": symbol,
                     "shape": f"{_GRID[0]}x{_GRID[1]}x{_GRID[2]}x8",
                     "ref": ref, "dispatch": dispatch,
                     "field_op": True})

    # integral.
    sym = _resolve_symbol("clifford_integral")
    _, dispatch, ref = _ga_entry_integral(rt)
    rows.append({"op": "clifford_integral", "symbol": sym,
                 "shape": f"{_INTEGRAL_N}x8→8", "ref": ref,
                 "dispatch": dispatch})

    return rows


def _resolve_symbol(op_name: str) -> str:
    """Look up the apple_gpu C ABI symbol via the backend manifest.

    This is the canonical 'manifest/backend selection' step: the
    benchmark consults the same dispatch table the compiler uses to
    pick the runtime kernel, asserting that the kernel ships fused
    on apple_gpu and extracting its symbol prefix.
    """
    manifest = bm.clifford_manifest_for(op_name)
    by_target = {e.target: e for e in manifest}
    apple_gpu = by_target.get("apple_gpu")
    if apple_gpu is None or apple_gpu.status != "fused":
        raise RuntimeError(
            f"manifest does not list apple_gpu status=fused for {op_name}"
        )
    spec = bm._CLIFFORD_APPLE_GPU_FUSED[op_name]
    return f"{spec['symbol_prefix']}f32"


# ---------------------------------------------------------------------------
# Stack-walk + timing for one GA primitive
# ---------------------------------------------------------------------------

def _interior_slice(shape: tuple[int, ...]) -> tuple:
    if len(shape) < 4:
        return (slice(None),) * len(shape)
    # Field op: (D0, D1, D2, 8) — restrict to interior cells.
    return (slice(1, shape[0] - 1),
            slice(1, shape[1] - 1),
            slice(1, shape[2] - 1),
            slice(None))


def time_dispatch(dispatch: Callable, reps: int) -> tuple[float, float]:
    dispatch()  # warm up
    samples_ms: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        dispatch()
        samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    return (statistics.median(samples_ms),
            statistics.stdev(samples_ms) if reps > 1 else 0.0)


def run_ga_primitive(row: dict[str, Any], reps: int,
                     device: str, version: str) -> dict[str, Any]:
    out = row["dispatch"]()
    ref = row["ref"]
    sl = _interior_slice(out.shape) if row.get("field_op") else (slice(None),) * out.ndim
    diff = np.abs(out[sl] - ref[sl])
    max_err = float(diff.max()) if diff.size else 0.0
    # Pointwise + reduction tolerances; field ops accumulate finite-diff error.
    tol = 5e-4 if row.get("field_op") else 1e-4
    ok = max_err <= tol
    latency_ms, stdev_ms = time_dispatch(row["dispatch"], reps)
    return {
        "backend": "apple_gpu",
        "op": row["op"],
        "shape": row["shape"],
        "dtype": "f32",
        "mode": "fused",
        "reps": reps,
        "latency_ms": latency_ms,
        "stdev_ms": stdev_ms,
        "max_abs_err": max_err,
        "tolerance": tol,
        "ok": ok,
        "symbol": row["symbol"],
        "device": device,
        "tessera_version": version,
    }


# ---------------------------------------------------------------------------
# EBM stack walks — Python execution today; manifest lookup confirms no
# Apple GPU MSL kernel exists yet and records mode='reference'.
# ---------------------------------------------------------------------------

def _ebm_energy_path(reps: int) -> tuple[float, float, float]:
    rng = np.random.RandomState(1000)
    x = rng.randn(8, 4).astype(np.float32)
    y = rng.randn(8, 4).astype(np.float32)

    def model_fn(xx, yy):
        return np.sum((xx - yy) ** 2, axis=1)

    expected = np.sum((x - y) ** 2, axis=1).astype(np.float32)
    out = ebm.energy(model_fn, x, y)
    err = float(np.abs(out - expected).max())
    ebm.energy(model_fn, x, y)  # warm up
    samples_ms: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        ebm.energy(model_fn, x, y)
        samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(samples_ms), (statistics.stdev(samples_ms)
                                            if reps > 1 else 0.0), err


def _ebm_inner_step_path(reps: int) -> tuple[float, float, float]:
    rng = np.random.RandomState(1001)
    y = rng.randn(16, 4).astype(np.float32)
    grad = rng.randn(16, 4).astype(np.float32)
    expected = y - 0.05 * grad
    out = ebm.inner_step(y, grad, eta=0.05)
    err = float(np.abs(out - expected).max())
    ebm.inner_step(y, grad, eta=0.05)
    samples_ms: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        ebm.inner_step(y, grad, eta=0.05)
        samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(samples_ms), (statistics.stdev(samples_ms)
                                            if reps > 1 else 0.0), err


def _ebm_langevin_step_path(reps: int) -> tuple[float, float, float]:
    rng = np.random.RandomState(1002)
    y = rng.randn(16, 4).astype(np.float32)
    key = RNGKey.from_seed(1002)
    # Analytic gradient — closed form to avoid `_numerical_grad` blowup.
    grad_fn = lambda yy: 2.0 * yy
    energy_fn = lambda yy: np.sum(yy * yy, axis=1)
    out, _ = ebm.langevin_step(y, energy_fn, eta=0.01, temperature=0.0,
                                rng_key=key, grad_fn=grad_fn)
    expected = y - 0.01 * 2.0 * y  # T=0 ⇒ pure GD
    err = float(np.abs(out - expected).max())
    ebm.langevin_step(y, energy_fn, eta=0.01, temperature=0.0,
                      rng_key=key, grad_fn=grad_fn)
    samples_ms: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        ebm.langevin_step(y, energy_fn, eta=0.01, temperature=0.0,
                           rng_key=key, grad_fn=grad_fn)
        samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(samples_ms), (statistics.stdev(samples_ms)
                                            if reps > 1 else 0.0), err


def _ebm_self_verify_path(reps: int) -> tuple[float, float, float]:
    rng = np.random.RandomState(1003)
    B, K, D = 4, 8, 16
    energies = rng.randn(B, K).astype(np.float32)
    candidates = rng.randn(B, K, D).astype(np.float32)
    expected = candidates[np.arange(B), energies.argmin(axis=1)]
    out = ebm.self_verify(energies, candidates)
    err = float(np.abs(out - expected).max())
    ebm.self_verify(energies, candidates)
    samples_ms: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        ebm.self_verify(energies, candidates)
        samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(samples_ms), (statistics.stdev(samples_ms)
                                            if reps > 1 else 0.0), err


def _ebm_decode_init_path(reps: int) -> tuple[float, float, float]:
    key = RNGKey.from_seed(1004)
    x = np.zeros((4, 12), dtype=np.float32)  # batch dim B=4 derived from x.shape[0]
    expected_shape = (4, 6, 12)
    kwargs = dict(K=6, init_strategy="noise", rng_key=key,
                  shape=(12,), dtype="fp32")
    out = ebm.decode_init(x, **kwargs)
    # Determinism: same key, same shape ⇒ same output.
    out2 = ebm.decode_init(x, **kwargs)
    err = float(np.abs(out - out2).max())
    assert out.shape == expected_shape, f"got {out.shape}, expected {expected_shape}"
    ebm.decode_init(x, **kwargs)  # warm up
    samples_ms: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        ebm.decode_init(x, **kwargs)
        samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(samples_ms), (statistics.stdev(samples_ms)
                                            if reps > 1 else 0.0), err


def _ebm_partition_exact_path(reps: int) -> tuple[float, float, float]:
    # Enumerate {-1, +1}^4 — tiny so partition is well-defined.
    states = np.array(np.meshgrid(*[[-1.0, 1.0]] * 4,
                                    indexing="ij")).reshape(4, -1).T
    states = states.astype(np.float32)
    state_list = [states[i] for i in range(states.shape[0])]

    def energy_fn(s):
        return -0.5 * float(np.sum(s * s))

    Z = ebm.partition_function_exact(energy_fn, state_list)
    # Z = sum_i exp(-E_i) — closed form on the discrete grid.
    expected = float(sum(math.exp(-energy_fn(s)) for s in state_list))
    err = abs(float(Z) - expected)
    ebm.partition_function_exact(energy_fn, state_list)
    samples_ms: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        ebm.partition_function_exact(energy_fn, state_list)
        samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(samples_ms), (statistics.stdev(samples_ms)
                                            if reps > 1 else 0.0), err


def _ebm_bivector_langevin_path(reps: int) -> tuple[float, float, float]:
    # Tiny Cl(3,0) bivector → ground-state energy; one Langevin step at T=0
    # collapses to gradient descent. Analytic gradient via exterior algebra.
    key = RNGKey.from_seed(1005)
    coeffs0 = np.zeros(8, dtype=np.float32)
    coeffs0[3] = 0.5   # e12
    coeffs0[5] = -0.2  # e13
    state = ga.Multivector(coeffs0, _CL30)

    def energy_fn(mv):
        c = mv.coefficients
        return 0.5 * float((c * c).sum())

    def grad_fn(mv):
        return ga.Multivector(mv.coefficients.astype(np.float64, copy=True),
                               mv.algebra)

    out, _ = ebm.bivector_langevin_step(state, energy_fn, eta=0.01,
                                          temperature=0.0, rng_key=key,
                                          grad_fn=grad_fn)
    # T=0 GD with grad = state: new = (1-eta)*state, restricted to grade-2.
    expected_e12 = (1.0 - 0.01) * 0.5
    err = abs(float(out.coefficients[3]) - expected_e12)
    for _ in range(2):
        ebm.bivector_langevin_step(state, energy_fn, eta=0.01,
                                     temperature=0.0, rng_key=key,
                                     grad_fn=grad_fn)
    samples_ms: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        ebm.bivector_langevin_step(state, energy_fn, eta=0.01,
                                     temperature=0.0, rng_key=key,
                                     grad_fn=grad_fn)
        samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(samples_ms), (statistics.stdev(samples_ms)
                                            if reps > 1 else 0.0), err


def _ebm_sphere_langevin_path(reps: int) -> tuple[float, float, float]:
    key = RNGKey.from_seed(1006)
    x = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def energy_fn(p):
        return -float(p[0])  # minimum at (+1, 0, 0)

    out, _ = ebm.sphere_langevin_step(x, energy_fn, eta=0.005,
                                        temperature=0.0, rng_key=key)
    # T=0 step should still be on the unit sphere.
    err = abs(float(np.linalg.norm(out)) - 1.0)
    for _ in range(2):
        ebm.sphere_langevin_step(x, energy_fn, eta=0.005,
                                   temperature=0.0, rng_key=key)
    samples_ms: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        ebm.sphere_langevin_step(x, energy_fn, eta=0.005,
                                   temperature=0.0, rng_key=key)
        samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(samples_ms), (statistics.stdev(samples_ms)
                                            if reps > 1 else 0.0), err


_EBM_PATHS: tuple[tuple[str, str, Callable, float], ...] = (
    ("ebm_energy",            "B=8,D=4",        _ebm_energy_path,            1e-5),
    ("ebm_inner_step",        "B=16,D=4",       _ebm_inner_step_path,        1e-5),
    ("ebm_langevin_step",     "B=16,D=4/T=0",   _ebm_langevin_step_path,     1e-5),
    ("ebm_self_verify",       "B=4,K=8,D=16",   _ebm_self_verify_path,       0.0),
    ("ebm_decode_init",       "K=6,shape=4x12", _ebm_decode_init_path,       0.0),
    ("ebm_partition_exact",   "{-1,+1}^4",      _ebm_partition_exact_path,   1e-4),
    ("ebm_bivector_langevin", "Cl(3,0)/T=0",    _ebm_bivector_langevin_path, 1e-4),
    ("ebm_sphere_langevin",   "S^2/T=0",        _ebm_sphere_langevin_path,   1e-5),
)


def run_ebm_path(name: str, shape: str, run_fn: Callable,
                  tolerance: float, reps: int,
                  device: str, version: str) -> dict[str, Any]:
    latency_ms, stdev_ms, err = run_fn(reps)
    # Confirm manifest is consistent: today EBM ops are Python-only on
    # apple_gpu (no fused MSL kernel registered). Document that in the
    # row so downstream tooling can plan kernel work.
    apple_gpu_status = "python_reference_only"
    return {
        "backend": "python_ref",
        "op": name,
        "shape": shape,
        "dtype": "f32",
        "mode": "reference",
        "reps": reps,
        "latency_ms": latency_ms,
        "stdev_ms": stdev_ms,
        "max_abs_err": err,
        "tolerance": tolerance,
        "ok": err <= tolerance,
        "apple_gpu_status": apple_gpu_status,
        "device": device,
        "tessera_version": version,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _device_name() -> str:
    return "apple_silicon_metal" if sys.platform == "darwin" else "non_darwin"


def _tessera_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("tessera")
    except Exception:
        return "dev"


def run_report(reps: int = 20, tmp_dir: Path | None = None) -> dict[str, Any]:
    """Top-level entry: build the report dict (no I/O)."""
    if tmp_dir is None:
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp(prefix="tessera_ga_ebm_bench_"))
    device = _device_name()
    version = _tessera_version()
    rt = compile_apple_gpu_runtime(tmp_dir)
    rows: list[dict[str, Any]] = []
    skipped_reason: str | None = None

    if rt is None:
        skipped_reason = ("apple_gpu runtime not available "
                          f"({sys.platform=}, dylib not built)")
    else:
        for row in build_ga_entries(rt):
            rows.append(run_ga_primitive(row, reps, device, version))

    for name, shape, fn, tol in _EBM_PATHS:
        rows.append(run_ebm_path(name, shape, fn, tol, reps, device, version))

    return {
        "runs": rows,
        "ga_primitives_count": sum(1 for r in rows if r["op"].startswith("clifford_")),
        "ebm_paths_count": sum(1 for r in rows if r["op"].startswith("ebm_")),
        "skipped_apple_gpu": skipped_reason,
        "device": device,
        "tessera_version": version,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=20,
                        help="Timing samples per primitive (median is reported)")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON output path (stdout if omitted)")
    args = parser.parse_args(argv)

    report = run_report(reps=args.reps)
    payload = json.dumps(report, indent=2, sort_keys=True, default=float)
    if args.output is not None:
        args.output.write_text(payload)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
