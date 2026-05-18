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
# source of truth for the Python API.  Keep the repo root importable too:
# importing `tessera` currently pulls a few test helpers that reference
# `examples.*` as a namespace package.
_PY_SRC = ROOT / "python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(_PY_SRC) not in sys.path:
    sys.path.insert(0, str(_PY_SRC))

import tessera.ga as ga  # noqa: E402
import tessera.ebm as ebm  # noqa: E402
from tessera.compiler import backend_manifest as bm  # noqa: E402
from tessera.rng import RNGKey  # noqa: E402


# ---------------------------------------------------------------------------
# Runtime compile/load — shared with the unit test fixture
# ---------------------------------------------------------------------------

def compile_apple_gpu_runtime(
    tmp_dir: Path,
) -> tuple[ctypes.CDLL | None, float, str | None]:
    """Compile ``apple_gpu_runtime.mm`` into a dylib and ctypes-load it.

    Returns ``(handle, compile_time_ms, skip_reason)`` — the handle is
    ``None`` on non-Darwin / missing-toolchain / compile-failure paths,
    in which case ``skip_reason`` is a non-empty diagnostic string.
    ``compile_time_ms`` is the wall-clock cost of the clang++ invocation
    so the report can separate compile time from dispatch time.
    """
    if sys.platform != "darwin":
        return None, 0.0, f"non-darwin host (sys.platform={sys.platform!r})"
    cxx = shutil.which("clang++") or shutil.which("c++")
    if cxx is None:
        return None, 0.0, "clang++/c++ not found on PATH"
    source = ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm"
    if not source.exists():
        return None, 0.0, f"runtime source missing: {source}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    lib = tmp_dir / "libtessera_apple_gpu_runtime.dylib"
    cmd = [cxx, "-std=c++17", "-shared", "-fPIC", "-O2", "-fobjc-arc",
           "-x", "objective-c++", str(source), "-o", str(lib),
           "-framework", "Metal",
           "-framework", "MetalPerformanceShaders",
           "-framework", "Foundation"]
    t0 = time.perf_counter_ns()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    compile_time_ms = (time.perf_counter_ns() - t0) / 1e6
    if proc.returncode != 0:
        return None, compile_time_ms, (
            f"clang++ failed (rc={proc.returncode}): {proc.stderr[-400:]}"
        )
    return ctypes.CDLL(str(lib)), compile_time_ms, None


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


def _percentile(samples: list[float], pct: float) -> float:
    """Linear-interpolation percentile (NumPy's default ``linear``
    method).  Robust for the tiny rep counts CI uses."""
    if not samples:
        return 0.0
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def collect_samples(dispatch: Callable, reps: int,
                     warmup: int = 1) -> list[float]:
    """Run ``dispatch`` ``warmup`` times then ``reps`` timed iterations.

    Returns the per-rep wall-clock samples in milliseconds.
    """
    for _ in range(warmup):
        dispatch()
    samples_ms: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        dispatch()
        samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    return samples_ms


def timing_stats(samples_ms: list[float]) -> dict[str, float]:
    """Compute the timing column for a report row.

    Always reports ``latency_ms`` (median, the canonical headline) plus
    ``stdev_ms``, ``p10_ms``, ``p50_ms``, ``p90_ms``, ``min_ms``,
    ``max_ms``.  ``stdev_ms`` is zero for a single rep — useful for CI
    where reps=2 still yields a stable signal but no spread.
    """
    if not samples_ms:
        return {"latency_ms": 0.0, "stdev_ms": 0.0,
                "p10_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0,
                "min_ms": 0.0, "max_ms": 0.0}
    median = statistics.median(samples_ms)
    stdev = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    return {
        "latency_ms": median,
        "stdev_ms": stdev,
        "p10_ms": _percentile(samples_ms, 10.0),
        "p50_ms": median,
        "p90_ms": _percentile(samples_ms, 90.0),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


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
    samples_ms = collect_samples(row["dispatch"], reps)
    timing = timing_stats(samples_ms)
    return {
        "backend": "apple_gpu",
        "namespace": "ga",
        "op": row["op"],
        "shape": row["shape"],
        "dtype": "f32",
        "mode": "fused",
        "reps": reps,
        **timing,
        "max_abs_err": max_err,
        "tolerance": tol,
        "ok": ok,
        "symbol": row["symbol"],
        "device": device,
        "tessera_version": version,
    }


# ---------------------------------------------------------------------------
# EBM stack walks — most are Python today; ebm_inner_step + ebm_refinement
# dispatch to the native Apple GPU C ABI when the runtime is available.
#
# Every path function returns ``(dispatch, max_err)`` where ``dispatch``
# is a no-arg callable that performs one step and ``max_err`` is the
# correctness gap vs the closed-form reference.  Timing happens in the
# central driver so percentiles, warmup, and the (manifest-driven)
# backend column are computed uniformly across rows.
# ---------------------------------------------------------------------------

def _ebm_energy_path() -> tuple[Callable[[], None], float]:
    rng = np.random.RandomState(1000)
    x = rng.randn(8, 4).astype(np.float32)
    y = rng.randn(8, 4).astype(np.float32)

    def model_fn(xx, yy):
        return np.sum((xx - yy) ** 2, axis=1)

    expected = np.sum((x - y) ** 2, axis=1).astype(np.float32)
    out = ebm.energy(model_fn, x, y)
    err = float(np.abs(out - expected).max())
    return (lambda: ebm.energy(model_fn, x, y)), err


def _ebm_inner_step_python_path() -> tuple[Callable[[], None], float]:
    """Python reference inner-step (kept for backward-compat).

    Superseded by ``_ebm_inner_step_apple_gpu_path`` when the runtime
    is available — both produce identical numerical output.
    """
    rng = np.random.RandomState(1001)
    y = rng.randn(16, 4).astype(np.float32)
    grad = rng.randn(16, 4).astype(np.float32)
    expected = y - 0.05 * grad
    out = ebm.inner_step(y, grad, eta=0.05)
    err = float(np.abs(out - expected).max())
    return (lambda: ebm.inner_step(y, grad, eta=0.05)), err


def _ebm_inner_step_apple_gpu_path(
    rt: ctypes.CDLL,
) -> tuple[Callable[[], np.ndarray], float, str]:
    """Native Apple-GPU EBM inner-step.

    ABI: ``out = y - eta * grad``, ``n`` is the total element count
    (the kernel is shape-agnostic).  Returns the ctypes-bound dispatch
    closure, the max-abs-err vs the closed-form reference, and the
    runtime symbol name (so the report can record it).
    """
    sym = "tessera_apple_gpu_ebm_inner_step_f32"
    fn = getattr(rt, sym)
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_float,
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32]
    fn.restype = None
    rng = np.random.RandomState(1001)
    y = np.ascontiguousarray(rng.randn(16, 4).astype(np.float32))
    grad = np.ascontiguousarray(rng.randn(16, 4).astype(np.float32))
    eta = 0.05
    n = int(y.size)
    out = np.zeros_like(y)
    expected = (y - eta * grad).astype(np.float32)

    def dispatch() -> np.ndarray:
        fn(_ptr_f(y), _ptr_f(grad), ctypes.c_float(eta),
           _ptr_f(out), ctypes.c_int32(n))
        return out

    dispatch()
    err = float(np.abs(out - expected).max())
    return dispatch, err, sym


def _ebm_refinement_apple_gpu_path(
    rt: ctypes.CDLL, T: int = 8,
) -> tuple[Callable[[], np.ndarray], float, str]:
    """Native Apple-GPU EBT refinement — T inner-step iterations.

    With a fixed gradient (no recomputation between steps), the closed
    form is ``y_T = y_0 - T * eta * grad`` — the kernel matches that
    bit-for-bit at fp32.
    """
    sym = "tessera_apple_gpu_ebm_refinement_f32"
    fn = getattr(rt, sym)
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_float, ctypes.c_int32,
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32]
    fn.restype = None
    rng = np.random.RandomState(1010)
    y0 = np.ascontiguousarray(rng.randn(8, 6).astype(np.float32))
    grad = np.ascontiguousarray(rng.randn(8, 6).astype(np.float32))
    eta = 0.02
    n = int(y0.size)
    y_out = np.zeros_like(y0)
    expected = (y0 - T * eta * grad).astype(np.float32)

    def dispatch() -> np.ndarray:
        fn(_ptr_f(y0), _ptr_f(grad), ctypes.c_float(eta),
           ctypes.c_int32(T), _ptr_f(y_out), ctypes.c_int32(n))
        return y_out

    dispatch()
    err = float(np.abs(y_out - expected).max())
    return dispatch, err, sym


def _ebm_langevin_step_apple_gpu_path(
    rt: ctypes.CDLL,
) -> tuple[Callable[[], np.ndarray], float, str]:
    """Native Apple-GPU Langevin step with caller-supplied noise.

    Mirrors ``tessera.ebm.langevin_step`` at T > 0:
      ``out = y - eta * grad + sqrt(2*eta*T) * noise``

    Caller pre-generates ``noise`` from the same Philox stream the
    Python path uses so the two outputs are bit-identical at fp32.
    """
    sym = "tessera_apple_gpu_ebm_langevin_step_f32"
    fn = getattr(rt, sym)
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_float, ctypes.c_float,
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32]
    fn.restype = None
    rng = np.random.RandomState(1020)
    y = np.ascontiguousarray(rng.randn(16, 4).astype(np.float32))
    grad = np.ascontiguousarray(rng.randn(16, 4).astype(np.float32))
    noise = np.ascontiguousarray(rng.randn(16, 4).astype(np.float32))
    eta = 0.01
    temperature = 1.0
    noise_scale = float(math.sqrt(2.0 * eta * temperature))
    n = int(y.size)
    out = np.zeros_like(y)
    expected = (y - eta * grad + noise_scale * noise).astype(np.float32)

    def dispatch() -> np.ndarray:
        fn(_ptr_f(y), _ptr_f(grad), _ptr_f(noise),
           ctypes.c_float(eta), ctypes.c_float(noise_scale),
           _ptr_f(out), ctypes.c_int32(n))
        return out

    dispatch()
    err = float(np.abs(out - expected).max())
    return dispatch, err, sym


def _ebm_decode_init_apple_gpu_path(
    rt: ctypes.CDLL,
) -> tuple[Callable[[], np.ndarray], float, str]:
    """Native Apple-GPU decode_init noise-apply.

    Mirrors ``ebm.decode_init(init_strategy='noise')`` semantics —
    output shape ``(B, K, *event)`` with each element ``base[b, *e] +
    std * noise``.  Caller pre-generates the deterministic ``noise``
    buffer from the RNGKey.
    """
    sym = "tessera_apple_gpu_ebm_decode_init_noise_apply_f32"
    fn = getattr(rt, sym)
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32,
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_float,
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32]
    fn.restype = None
    rng = np.random.RandomState(1030)
    # B=4, K=6, event=12 → 288-element output.
    B, K, E = 4, 6, 12
    base = np.ascontiguousarray(rng.randn(B, E).astype(np.float32))  # per-batch mean
    # Broadcast: base[b, e] → (b, k, e) for every k.
    noise = np.ascontiguousarray(rng.randn(B, K, E).astype(np.float32))
    std = 0.5
    n = int(noise.size)
    base_flat = np.ascontiguousarray(np.broadcast_to(
        base[:, None, :], (B, K, E)).reshape(-1).astype(np.float32))
    base_len = int(base_flat.size)  # n; no broadcasting needed at kernel level
    out = np.zeros_like(noise)
    expected = (base_flat.reshape(B, K, E) + std * noise).astype(np.float32)

    def dispatch() -> np.ndarray:
        fn(_ptr_f(base_flat), ctypes.c_int32(base_len),
           _ptr_f(noise), ctypes.c_float(std),
           _ptr_f(out), ctypes.c_int32(n))
        return out

    dispatch()
    err = float(np.abs(out - expected).max())
    return dispatch, err, sym


def _ebm_bivector_langevin_apple_gpu_path(
    rt: ctypes.CDLL,
) -> tuple[Callable[[], np.ndarray], float, str]:
    """Native Apple-GPU bivector Langevin — composition of GA
    ``grade_projection`` + ``ebm_langevin_step``.

    The 8-coefficient Cl(3,0) state is restricted to grade-2 bivectors
    (only blades 3, 5, 6 nonzero).  ``grad_fn`` returns a Multivector,
    we project to grade-2 (mask non-bivector blades to zero), generate
    grade-2 noise, then run a single langevin_step kernel.  This is
    the GA-kernel-reuse demonstration the manifest documents.
    """
    sym = "tessera_apple_gpu_ebm_langevin_step_f32"
    fn = getattr(rt, sym)
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_float, ctypes.c_float,
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32]
    fn.restype = None

    # State: bivector blades only.
    rng = np.random.RandomState(1040)
    coeffs = np.zeros(8, dtype=np.float32)
    coeffs[3] = 0.5    # e12
    coeffs[5] = -0.2   # e13
    coeffs[6] = 0.3    # e23
    # Gradient: full 8-vector that we grade-project to bivector subspace.
    raw_grad = rng.randn(8).astype(np.float32)
    grad_proj = np.zeros_like(raw_grad)
    for k in (3, 5, 6):
        grad_proj[k] = raw_grad[k]
    # Noise: same projection.
    raw_noise = rng.randn(8).astype(np.float32)
    noise_proj = np.zeros_like(raw_noise)
    for k in (3, 5, 6):
        noise_proj[k] = raw_noise[k]
    eta = 0.01
    temperature = 1.0
    noise_scale = float(math.sqrt(2.0 * eta * temperature))
    n = 8
    out = np.zeros(8, dtype=np.float32)
    expected = coeffs - eta * grad_proj + noise_scale * noise_proj

    state = np.ascontiguousarray(coeffs)
    g = np.ascontiguousarray(grad_proj)
    ns = np.ascontiguousarray(noise_proj)

    def dispatch() -> np.ndarray:
        fn(_ptr_f(state), _ptr_f(g), _ptr_f(ns),
           ctypes.c_float(eta), ctypes.c_float(noise_scale),
           _ptr_f(out), ctypes.c_int32(n))
        return out

    dispatch()
    # After the affine combo the non-bivector blades should still be 0.
    err = float(max(np.abs(out - expected).max(),
                    abs(out[0]), abs(out[1]), abs(out[2]),
                    abs(out[4]), abs(out[7])))
    return dispatch, err, sym


def _ebm_sphere_langevin_apple_gpu_path(
    rt: ctypes.CDLL,
) -> tuple[Callable[[], np.ndarray], float, str]:
    """Native Apple-GPU sphere Langevin — full step in one MSL kernel.

    Tangent-projects grad + noise, applies the Euler-Maruyama step,
    retracts to the unit sphere — all on-device.  Matches
    ``tessera.ebm.sphere_langevin_step`` semantically (with caller-
    supplied noise for determinism).
    """
    sym = "tessera_apple_gpu_ebm_sphere_langevin_step_f32"
    fn = getattr(rt, sym)
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_float, ctypes.c_float,
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32]
    fn.restype = None
    # Start on the unit sphere.
    x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rng = np.random.RandomState(1050)
    grad = rng.randn(3).astype(np.float32)
    noise = rng.randn(3).astype(np.float32)
    eta = 0.005
    temperature = 0.5
    noise_scale = float(math.sqrt(2.0 * eta * temperature))
    d = 3
    out = np.zeros(3, dtype=np.float32)

    # Reference: tangent project, step, retract.
    gdot = float(np.dot(grad, x))
    ndot = float(np.dot(noise, x))
    grad_tan = grad - gdot * x
    noise_tan = noise - ndot * x
    y = x - eta * grad_tan + noise_scale * noise_tan
    expected = (y / float(np.linalg.norm(y))).astype(np.float32)

    x_c = np.ascontiguousarray(x)
    g_c = np.ascontiguousarray(grad)
    n_c = np.ascontiguousarray(noise)

    def dispatch() -> np.ndarray:
        fn(_ptr_f(x_c), _ptr_f(g_c), _ptr_f(n_c),
           ctypes.c_float(eta), ctypes.c_float(noise_scale),
           _ptr_f(out), ctypes.c_int32(d))
        return out

    dispatch()
    err = float(np.abs(out - expected).max())
    return dispatch, err, sym


def _ebm_self_verify_apple_gpu_path(
    rt: ctypes.CDLL,
) -> tuple[Callable[[], np.ndarray], float, str]:
    """Native Apple-GPU self_verify (hard argmin path).

    One MSL threadgroup per batch row scans K energies and gathers
    the winning candidate row.  Matches the hard-argmin path of
    ``tessera.ebm.self_verify(... beta=None)``.
    """
    sym = "tessera_apple_gpu_ebm_self_verify_hard_argmin_f32"
    fn = getattr(rt, sym)
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    fn.restype = None
    B, K, D = 4, 8, 16
    rng = np.random.RandomState(1060)
    energies = np.ascontiguousarray(rng.randn(B, K).astype(np.float32))
    candidates = np.ascontiguousarray(rng.randn(B, K, D).astype(np.float32))
    out = np.zeros((B, D), dtype=np.float32)
    expected = candidates[np.arange(B), energies.argmin(axis=1)]

    def dispatch() -> np.ndarray:
        fn(_ptr_f(energies), _ptr_f(candidates), _ptr_f(out),
           ctypes.c_int32(B), ctypes.c_int32(K), ctypes.c_int32(D))
        return out

    dispatch()
    err = float(np.abs(out - expected).max())
    return dispatch, err, sym


def _ebm_energy_apple_gpu_path(
    rt: ctypes.CDLL,
) -> tuple[Callable[[], np.ndarray], float, str]:
    """Native Apple-GPU energy — quadratic specialization
    ``E_b = 0.5 * ||x_b - y_b||^2``.

    Callers opt into this specialization when their ``model_fn``
    matches the quadratic form (true for diffusion noise prediction,
    EBT reconstruction loss, Gaussian log-likelihood up to a constant).
    """
    sym = "tessera_apple_gpu_ebm_energy_quadratic_f32"
    fn = getattr(rt, sym)
    fn.argtypes = [ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int32, ctypes.c_int32]
    fn.restype = None
    B, D = 8, 4
    rng = np.random.RandomState(1070)
    x = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    y = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    energies = np.zeros(B, dtype=np.float32)
    expected = (0.5 * np.sum((x - y) ** 2, axis=1)).astype(np.float32)

    def dispatch() -> np.ndarray:
        fn(_ptr_f(x), _ptr_f(y), _ptr_f(energies),
           ctypes.c_int32(B), ctypes.c_int32(D))
        return energies

    dispatch()
    err = float(np.abs(energies - expected).max())
    return dispatch, err, sym


# Python-reference Langevin (kept for the non-Apple skip path).
def _ebm_langevin_step_path() -> tuple[Callable[[], None], float]:
    rng = np.random.RandomState(1002)
    y = rng.randn(16, 4).astype(np.float32)
    key = RNGKey.from_seed(1002)
    grad_fn = lambda yy: 2.0 * yy
    energy_fn = lambda yy: np.sum(yy * yy, axis=1)
    out, _ = ebm.langevin_step(y, energy_fn, eta=0.01, temperature=0.0,
                                rng_key=key, grad_fn=grad_fn)
    expected = y - 0.01 * 2.0 * y
    err = float(np.abs(out - expected).max())
    return (lambda: ebm.langevin_step(y, energy_fn, eta=0.01,
                                       temperature=0.0, rng_key=key,
                                       grad_fn=grad_fn)), err


def _ebm_self_verify_path() -> tuple[Callable[[], None], float]:
    rng = np.random.RandomState(1003)
    B, K, D = 4, 8, 16
    energies = rng.randn(B, K).astype(np.float32)
    candidates = rng.randn(B, K, D).astype(np.float32)
    expected = candidates[np.arange(B), energies.argmin(axis=1)]
    out = ebm.self_verify(energies, candidates)
    err = float(np.abs(out - expected).max())
    return (lambda: ebm.self_verify(energies, candidates)), err


def _ebm_decode_init_path() -> tuple[Callable[[], None], float]:
    key = RNGKey.from_seed(1004)
    x = np.zeros((4, 12), dtype=np.float32)
    kwargs = dict(K=6, init_strategy="noise", rng_key=key,
                  shape=(12,), dtype="fp32")
    out = ebm.decode_init(x, **kwargs)
    out2 = ebm.decode_init(x, **kwargs)
    err = float(np.abs(out - out2).max())
    assert out.shape == (4, 6, 12)
    return (lambda: ebm.decode_init(x, **kwargs)), err


def _ebm_partition_exact_path() -> tuple[Callable[[], None], float]:
    states = np.array(np.meshgrid(*[[-1.0, 1.0]] * 4,
                                    indexing="ij")).reshape(4, -1).T
    states = states.astype(np.float32)
    state_list = [states[i] for i in range(states.shape[0])]

    def energy_fn(s):
        return -0.5 * float(np.sum(s * s))

    Z = ebm.partition_function_exact(energy_fn, state_list)
    expected = float(sum(math.exp(-energy_fn(s)) for s in state_list))
    err = abs(float(Z) - expected)
    return (lambda: ebm.partition_function_exact(energy_fn, state_list)), err


def _ebm_bivector_langevin_path() -> tuple[Callable[[], None], float]:
    key = RNGKey.from_seed(1005)
    coeffs0 = np.zeros(8, dtype=np.float32)
    coeffs0[3] = 0.5
    coeffs0[5] = -0.2
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
    expected_e12 = (1.0 - 0.01) * 0.5
    err = abs(float(out.coefficients[3]) - expected_e12)
    return (lambda: ebm.bivector_langevin_step(state, energy_fn, eta=0.01,
                                                 temperature=0.0,
                                                 rng_key=key,
                                                 grad_fn=grad_fn)), err


def _ebm_sphere_langevin_path() -> tuple[Callable[[], None], float]:
    key = RNGKey.from_seed(1006)
    x = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def energy_fn(p):
        return -float(p[0])

    out, _ = ebm.sphere_langevin_step(x, energy_fn, eta=0.005,
                                        temperature=0.0, rng_key=key)
    err = abs(float(np.linalg.norm(out)) - 1.0)
    return (lambda: ebm.sphere_langevin_step(x, energy_fn, eta=0.005,
                                               temperature=0.0,
                                               rng_key=key)), err


# Python-reference EBM paths: (op_name, shape_desc, builder, tolerance).
# Always emitted (cross-platform coverage); on Apple Silicon they sit
# alongside the apple_gpu rows so the speedup is a single subtraction.
_EBM_PYTHON_PATHS: tuple[tuple[str, str, Callable, float], ...] = (
    ("ebm_energy",            "B=8,D=4",        _ebm_energy_path,            1e-5),
    ("ebm_inner_step",        "B=16,D=4",       _ebm_inner_step_python_path, 1e-5),
    ("ebm_langevin_step",     "B=16,D=4/T=0",   _ebm_langevin_step_path,     1e-5),
    ("ebm_self_verify",       "B=4,K=8,D=16",   _ebm_self_verify_path,       0.0),
    ("ebm_decode_init",       "K=6,shape=4x12", _ebm_decode_init_path,       0.0),
    ("ebm_partition_exact",   "{-1,+1}^4",      _ebm_partition_exact_path,   1e-4),
    ("ebm_bivector_langevin", "Cl(3,0)/T=0",    _ebm_bivector_langevin_path, 1e-4),
    ("ebm_sphere_langevin",   "S^2/T=0",        _ebm_sphere_langevin_path,   1e-5),
)


# Native-EBM Apple-GPU builder registry: each entry is
# (op_name, shape_desc, builder_fn(rt) -> (dispatch, err, symbol), tolerance).
def _NATIVE_EBM_BUILDERS(
    refinement_T: int,
) -> tuple[tuple[str, str, Callable, float], ...]:
    return (
        ("ebm_inner_step",        "B=16,D=4",
         _ebm_inner_step_apple_gpu_path,        1e-6),
        ("ebm_refinement",        f"B=8,D=6/T={refinement_T}",
         lambda rt: _ebm_refinement_apple_gpu_path(rt, T=refinement_T),
         1e-5),
        ("ebm_langevin_step",     "B=16,D=4/T=1",
         _ebm_langevin_step_apple_gpu_path,     1e-6),
        ("ebm_decode_init",       "B=4,K=6,event=12",
         _ebm_decode_init_apple_gpu_path,       1e-6),
        ("ebm_bivector_langevin", "Cl(3,0)/T=1",
         _ebm_bivector_langevin_apple_gpu_path, 1e-6),
        ("ebm_sphere_langevin",   "S^2/T=0.5",
         _ebm_sphere_langevin_apple_gpu_path,   1e-5),
        ("ebm_self_verify",       "B=4,K=8,D=16",
         _ebm_self_verify_apple_gpu_path,       0.0),
        ("ebm_energy",            "B=8,D=4/quadratic",
         _ebm_energy_apple_gpu_path,            1e-6),
    )


# ---------------------------------------------------------------------------
# Workload mode — small composite pipelines that string multiple
# primitives together, not single-op timings.
# ---------------------------------------------------------------------------

def _workload_ga_pipeline_inputs(seed: int = 2000):
    """Deterministic inputs for the GA feature pipeline:
        - B random pure bivectors (8-coeff each)
        - B random multivectors V
        - Pipeline: R = exp(B); O = rotor_sandwich(R, V); s = norm(O)
    """
    B = 32
    rng = np.random.RandomState(seed)
    bivecs = np.zeros((B, 8), dtype=np.float32)
    bivecs[:, 3] = rng.randn(B).astype(np.float32) * 0.3
    bivecs[:, 5] = rng.randn(B).astype(np.float32) * 0.3
    bivecs[:, 6] = rng.randn(B).astype(np.float32) * 0.3
    vectors = rng.randn(B, 8).astype(np.float32)
    return B, bivecs, vectors


def _workload_ga_pipeline_python_path() -> tuple[Callable[[], np.ndarray],
                                                   float, str]:
    """Python reference: exp_mv → rotor_sandwich → norm on a batch.

    Returns ``(dispatch, max_abs_err_self, shape_descriptor)``. The
    "err vs reference" for the Python row is 0 (it *is* the reference).
    """
    B, bivecs, vectors = _workload_ga_pipeline_inputs()

    def step() -> np.ndarray:
        out = np.zeros(B, dtype=np.float32)
        for i in range(B):
            rotor = ga.exp_mv(ga.Multivector(bivecs[i], _CL30))
            rotated = ga.rotor_sandwich(rotor,
                                         ga.Multivector(vectors[i], _CL30))
            out[i] = float(np.asarray(ga.norm(rotated)))
        return out

    step()  # warm up
    return step, 0.0, f"B={B}/exp→sandwich→norm"


def _workload_ga_pipeline_apple_gpu_path(
    rt: ctypes.CDLL,
) -> tuple[Callable[[], np.ndarray], float, tuple[str, ...], str]:
    """Apple-GPU pipeline through **public APIs**.

    ``ga.exp_mv``, ``ga.rotor_sandwich`` and ``ga.norm`` all route
    through ``tessera._apple_gpu_dispatch`` when given Cl(3,0) f32
    batched inputs.  This driver calls the public functions directly
    — no benchmark-local ctypes — to prove the integration is
    user-visible.  The ``rt`` parameter is kept for ABI parity with
    the other workload builders but unused; the dispatcher loads its
    own runtime handle on first call.
    """
    del rt  # routed through the public dispatcher cache instead
    B, bivecs, vectors = _workload_ga_pipeline_inputs()
    a = _CL30
    bivec_mv = ga.Multivector(np.ascontiguousarray(bivecs), a)
    vec_mv = ga.Multivector(np.ascontiguousarray(vectors), a)

    def step() -> np.ndarray:
        rotor = ga.exp_mv(bivec_mv)
        rotated = ga.rotor_sandwich(rotor, vec_mv)
        return np.asarray(ga.norm(rotated))

    out = step()
    # Reference via Python — same inputs, same algorithm.
    ref, _, _ = _workload_ga_pipeline_python_path()
    ref_out = ref()
    err = float(np.abs(out - ref_out).max())
    syms = (
        "tessera_apple_gpu_clifford_exp_cl30_f32 [via ga.exp_mv]",
        "tessera_apple_gpu_clifford_rotor_sandwich_cl30_f32 [via ga.rotor_sandwich]",
        "tessera_apple_gpu_clifford_norm_cl30_f32 [via ga.norm]",
    )
    # Shape descriptor must match the python_ref row's so the
    # break-even pairing in `_ebt_sweep_break_even_summary` works.
    # The "via public API" provenance is recorded in `symbols`.
    return step, err, syms, f"B={B}/exp→sandwich→norm"


def _workload_ebt_tiny_inputs(
    seed: int = 2100, B: int = 4, K: int = 8, D: int = 6,
):
    """Deterministic inputs for the EBT-tiny refinement loop.

    The energy is E(y) = 0.5 * ||y||^2 ⇒ grad = y. With fixed grad
    (snapshot of y at step 0), T inner steps drive y → y0 - T·eta·y0
    = (1 - T·eta) * y0 — closed form.  Real EBT recomputes grad each
    step; the snapshot variant is what `ebm_refinement` measures so
    we keep the workload consistent with it.

    Returns ``(B, K, D, y0[BK,D], eta)``.
    """
    rng = np.random.RandomState(seed)
    y0 = rng.randn(B * K, D).astype(np.float32)
    eta = 0.02
    return B, K, D, y0, eta


def _workload_ebt_tiny_python_path(
    T: int, *, B: int = 4, K: int = 8, D: int = 6,
) -> tuple[Callable[[], np.ndarray], float, str]:
    """Python reference EBT-tiny loop:
        - K candidates per batch
        - T inner steps on a snapshot of grad = state_0
        - self_verify reduces K → 1 by argmin(energy)
    """
    _, _, _, y0, eta = _workload_ebt_tiny_inputs(B=B, K=K, D=D)

    def step() -> np.ndarray:
        grad = y0.copy()                          # fixed grad snapshot
        y = y0.copy()
        for _ in range(T):
            y = y - eta * grad                     # raw numpy, no dispatcher
        energies = np.sum(y * y, axis=1).reshape(B, K)
        candidates = y.reshape(B, K, D)
        return ebm.self_verify(energies, candidates)

    step()
    return step, 0.0, f"B={B},K={K},D={D}/T={T}"


def _workload_ebt_tiny_apple_gpu_path(
    rt: ctypes.CDLL, T: int, *,
    B: int = 4, K: int = 8, D: int = 6,
) -> tuple[Callable[[], np.ndarray], float, tuple[str, ...], str]:
    """Apple-GPU EBT-tiny loop through the **fused public API**.

    ``ebm.ebt_tiny(y0, grad, eta=eta, T=T, B=B, K=K, D=D)`` routes
    through ``tessera._apple_gpu_dispatch`` to the fused MSL kernel
    ``ebm_ebt_tiny_refinement_argmin_f32`` — a single dispatch that
    runs T-step refinement in registers, computes squared-norm
    energies per candidate, and hard-argmins over K, all on-device.

    The ``rt`` parameter is kept for ABI parity with the other
    workload builders but unused; the dispatcher manages its own
    runtime handle.
    """
    del rt  # routed through the public dispatcher cache
    _, _, _, y0, eta = _workload_ebt_tiny_inputs(B=B, K=K, D=D)
    y0_c = np.ascontiguousarray(y0)
    grad_c = np.ascontiguousarray(y0.copy())          # fixed grad snapshot

    def step() -> np.ndarray:
        return ebm.ebt_tiny(y0_c, grad_c, eta=eta, T=T, B=B, K=K, D=D)

    out = step()
    ref, _, _ = _workload_ebt_tiny_python_path(T, B=B, K=K, D=D)
    ref_out = ref()
    err = float(np.abs(out - ref_out).max())
    syms = (
        "tessera_apple_gpu_ebm_ebt_tiny_refinement_argmin_f32 [via ebm.ebt_tiny]",
    )
    return step, err, syms, f"B={B},K={K},D={D}/T={T}"


# Default break-even sweep — chosen so the report shows both regimes
# now that the fused single-dispatch refinement kernel has landed:
# native loses at sub-millisecond numpy times (dispatch overhead) and
# wins at larger shapes (one MSL dispatch beats T sequential numpy
# passes).  Break-even is around ``(32, 64, 512, 32)``; peak speedup
# ~22× at ``(64, 128, 1024, 256)``.
_DEFAULT_EBT_SWEEP: tuple[tuple[int, int, int, int], ...] = (
    # loss regime — dispatch overhead bound
    (4,    8,    6,    8),
    (16,  32,  128,    8),
    (32,  64,  256,    8),
    # break-even shoulder
    (32,  64,  512,   32),
    (32, 128,  512,   64),
    # win regime — native dominates
    (64, 128, 1024,   64),
    (64, 128, 1024,  256),
)


def run_workload_apple_gpu(name: str, shape: str,
                            dispatch: Callable, err: float,
                            symbols: tuple[str, ...],
                            tolerance: float, reps: int,
                            device: str, version: str) -> dict[str, Any]:
    samples_ms = collect_samples(dispatch, reps)
    timing = timing_stats(samples_ms)
    return {
        "backend": "apple_gpu",
        "namespace": "workload",
        "op": name,
        "shape": shape,
        "dtype": "f32",
        "mode": "fused_chain",
        "reps": reps,
        **timing,
        "max_abs_err": err,
        "tolerance": tolerance,
        "ok": err <= tolerance,
        "symbols": list(symbols),
        "device": device,
        "tessera_version": version,
    }


def run_workload_python(name: str, shape: str,
                         dispatch: Callable, err: float,
                         tolerance: float, reps: int,
                         device: str, version: str) -> dict[str, Any]:
    samples_ms = collect_samples(dispatch, reps)
    timing = timing_stats(samples_ms)
    return {
        "backend": "python_ref",
        "namespace": "workload",
        "op": name,
        "shape": shape,
        "dtype": "f32",
        "mode": "reference_chain",
        "reps": reps,
        **timing,
        "max_abs_err": err,
        "tolerance": tolerance,
        "ok": err <= tolerance,
        "device": device,
        "tessera_version": version,
    }


def _manifest_status_for(op_name: str, target: str) -> str:
    """Pull the canonical status for ``op_name`` on ``target`` from the
    backend manifest.  Falls back to ``"unknown"`` so the report still
    serializes if the manifest gets out of sync."""
    for entry in bm.manifest_for(op_name):
        if entry.target == target:
            return entry.status
    return "unknown"


def run_ebm_python_path(name: str, shape: str, builder: Callable,
                         tolerance: float, reps: int,
                         device: str, version: str) -> dict[str, Any]:
    """Time the Python reference path for an EBM op + record manifest
    status (``apple_gpu = planned`` for these rows today)."""
    dispatch, err = builder()
    samples_ms = collect_samples(dispatch, reps)
    timing = timing_stats(samples_ms)
    return {
        "backend": "python_ref",
        "namespace": "ebm",
        "op": name,
        "shape": shape,
        "dtype": "f32",
        "mode": "reference",
        "reps": reps,
        **timing,
        "max_abs_err": err,
        "tolerance": tolerance,
        "ok": err <= tolerance,
        "apple_gpu_status": _manifest_status_for(name, "apple_gpu"),
        "x86_status": _manifest_status_for(name, "x86"),
        "device": device,
        "tessera_version": version,
    }


def run_ebm_apple_gpu_path(name: str, shape: str,
                            dispatch: Callable, err: float, symbol: str,
                            tolerance: float, reps: int,
                            device: str, version: str) -> dict[str, Any]:
    """Time a native Apple-GPU EBM dispatch + cross-check the manifest
    has the matching ``apple_gpu=fused`` entry."""
    samples_ms = collect_samples(dispatch, reps)
    timing = timing_stats(samples_ms)
    manifest_status = _manifest_status_for(name, "apple_gpu")
    return {
        "backend": "apple_gpu",
        "namespace": "ebm",
        "op": name,
        "shape": shape,
        "dtype": "f32",
        "mode": "fused",
        "reps": reps,
        **timing,
        "max_abs_err": err,
        "tolerance": tolerance,
        "ok": err <= tolerance and manifest_status == "fused",
        "apple_gpu_status": manifest_status,
        "symbol": symbol,
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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

# Default reps tuning: small for CI (cheap + stable verdict), larger for
# manual runs where percentiles need a meaningful sample.
DEFAULT_REPS_MANUAL = 50
DEFAULT_REPS_CI = 2


def run_report(reps: int = DEFAULT_REPS_MANUAL,
               tmp_dir: Path | None = None,
               *,
               refinement_T: int = 8,
               include_primitives: bool = True,
               include_workloads: bool = True,
               include_ebt_sweep: bool = False,
               ebt_sweep_points: tuple[tuple[int, int, int, int], ...]
                   = _DEFAULT_EBT_SWEEP) -> dict[str, Any]:
    """Top-level entry: build the report dict (no I/O).

    The report envelope records ``compile_time_ms`` (clang++ wall-clock
    for the runtime dylib) separately from per-row ``latency_ms`` (the
    pure GPU/CPU dispatch time), so consumers can amortize compile
    cost across many runs and compare dispatch latency in isolation.

    Two coverage axes:
      - ``include_primitives``: 17 GA primitives + 8 native EBM
        primitives + Python-reference EBM rows (so consumers can
        compute native-vs-reference speedup per op).
      - ``include_workloads``: GA feature pipeline (exp →
        rotor_sandwich → norm) + EBT-tiny refinement loop (decode_init
        → T × inner_step → self_verify), each in apple_gpu + python_ref
        variants so speedup is a single subtraction.
    """
    if tmp_dir is None:
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp(prefix="tessera_ga_ebm_bench_"))
    device = _device_name()
    version = _tessera_version()
    rt, compile_time_ms, skip_reason = compile_apple_gpu_runtime(tmp_dir)
    rows: list[dict[str, Any]] = []
    native_ebm_ops: set[str] = set()

    if include_primitives:
        if rt is not None:
            # 17 GA primitives — all ship fused MSL kernels.
            for row in build_ga_entries(rt):
                rows.append(run_ga_primitive(row, reps, device, version))

            # Native EBM primitives — 6 total as of broadening sprint.
            #   inner_step, refinement, langevin_step,
            #   decode_init, bivector_langevin, sphere_langevin
            for op_name, shape_desc, builder, tol in _NATIVE_EBM_BUILDERS(
                    refinement_T):
                dispatch, err, sym = builder(rt)
                rows.append(run_ebm_apple_gpu_path(
                    op_name, shape_desc, dispatch, err, sym,
                    tolerance=tol, reps=reps,
                    device=device, version=version,
                ))
                native_ebm_ops.add(op_name)

        # Python-reference EBM paths — emitted on every host so the
        # report keeps comprehensive coverage even on non-Darwin AND so
        # speedup-vs-reference is visible on Apple Silicon as the
        # apple_gpu/python_ref row pair per op.
        for name, shape, builder, tol in _EBM_PYTHON_PATHS:
            rows.append(run_ebm_python_path(name, shape, builder, tol,
                                              reps, device, version))

    if include_workloads:
        # Workload pair 1: small GA feature pipeline.
        if rt is not None:
            dispatch, err, syms, shape = _workload_ga_pipeline_apple_gpu_path(rt)
            rows.append(run_workload_apple_gpu(
                "ga_feature_pipeline", shape, dispatch, err, syms,
                tolerance=2e-5, reps=reps, device=device, version=version,
            ))
        dispatch, err, shape = _workload_ga_pipeline_python_path()
        rows.append(run_workload_python(
            "ga_feature_pipeline", shape, dispatch, err,
            tolerance=2e-5, reps=reps, device=device, version=version,
        ))

        # Workload pair 2: EBT-tiny refinement loop (default shape).
        if rt is not None:
            dispatch, err, syms, shape = _workload_ebt_tiny_apple_gpu_path(
                rt, refinement_T)
            rows.append(run_workload_apple_gpu(
                "ebt_tiny_refinement", shape, dispatch, err, syms,
                tolerance=1e-4, reps=reps, device=device, version=version,
            ))
        dispatch, err, shape = _workload_ebt_tiny_python_path(refinement_T)
        rows.append(run_workload_python(
            "ebt_tiny_refinement", shape, dispatch, err,
            tolerance=1e-4, reps=reps, device=device, version=version,
        ))

    # EBT-tiny break-even sweep — apples-to-apples timings across a
    # ladder of (B, K, D, T) points so consumers can locate where the
    # native MSL chain starts beating numpy.  Off by default (adds
    # ~14 rows × reps dispatches); enable via --ebt-sweep.
    if include_ebt_sweep:
        for (B, K, D, T) in ebt_sweep_points:
            if rt is not None:
                dispatch, err, syms, shape = _workload_ebt_tiny_apple_gpu_path(
                    rt, T, B=B, K=K, D=D)
                rows.append(run_workload_apple_gpu(
                    "ebt_tiny_sweep", shape, dispatch, err, syms,
                    tolerance=1e-4, reps=reps,
                    device=device, version=version,
                ))
            dispatch, err, shape = _workload_ebt_tiny_python_path(
                T, B=B, K=K, D=D)
            rows.append(run_workload_python(
                "ebt_tiny_sweep", shape, dispatch, err,
                tolerance=1e-4, reps=reps,
                device=device, version=version,
            ))

    ga_count = sum(1 for r in rows if r["op"].startswith("clifford_"))
    ebm_count = sum(1 for r in rows if r["op"].startswith("ebm_"))
    workload_count = sum(1 for r in rows if r.get("namespace") == "workload")
    ebm_native_count = sum(1 for r in rows
                            if r["op"].startswith("ebm_")
                            and r["backend"] == "apple_gpu")
    sweep_count = sum(1 for r in rows if r["op"] == "ebt_tiny_sweep")
    return {
        "runs": rows,
        "ga_primitives_count": ga_count,
        "ebm_paths_count": ebm_count,
        "ebm_native_apple_gpu_count": ebm_native_count,
        "native_ebm_ops": sorted(native_ebm_ops),
        "workload_count": workload_count,
        "ebt_sweep_count": sweep_count,
        "ebt_sweep_summary": (_ebt_sweep_break_even_summary(rows)
                               if sweep_count else None),
        "compile_time_ms": compile_time_ms,
        "skipped_apple_gpu": skip_reason,
        "device": device,
        "tessera_version": version,
        "reps": reps,
    }


def _ebt_sweep_break_even_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce a per-shape native-vs-numpy comparison table.

    Pairs each ``ebt_tiny_sweep`` row by its ``shape`` descriptor and
    computes ``speedup = python_ms / native_ms`` plus a boolean
    ``native_wins`` (speedup > 1).  Useful as the headline of the
    sweep — consumers can scan one column to see the break-even.
    """
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        if r["op"] != "ebt_tiny_sweep":
            continue
        pairs.setdefault(r["shape"], {})[r["backend"]] = r
    table: list[dict[str, Any]] = []
    for shape, by_backend in sorted(pairs.items()):
        native = by_backend.get("apple_gpu")
        python = by_backend.get("python_ref")
        entry: dict[str, Any] = {"shape": shape}
        if native is not None:
            entry["native_ms"] = native["latency_ms"]
        if python is not None:
            entry["python_ms"] = python["latency_ms"]
        if native is not None and python is not None and native["latency_ms"] > 0:
            sp = python["latency_ms"] / native["latency_ms"]
            entry["speedup"] = sp
            entry["native_wins"] = sp >= 1.0
        table.append(entry)
    break_even = None
    for entry in table:
        if entry.get("native_wins"):
            break_even = entry["shape"]
            break
    return {"table": table, "first_native_win_shape": break_even}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS_MANUAL,
                        help=("Timing samples per primitive (median is "
                              "the headline; p10/p50/p90 also reported). "
                              f"Default: {DEFAULT_REPS_MANUAL}."))
    parser.add_argument("--ci", action="store_true",
                        help=(f"Use the CI-friendly rep count "
                              f"({DEFAULT_REPS_CI}); overrides --reps."))
    parser.add_argument("--refinement-T", type=int, default=8,
                        help="EBT-refinement inner-step iterations (default 8).")
    parser.add_argument("--primitives-only", action="store_true",
                        help="Skip workload mode (GA pipeline + EBT-tiny loop).")
    parser.add_argument("--workloads-only", action="store_true",
                        help="Skip per-primitive rows; only run composite workloads.")
    parser.add_argument("--ebt-sweep", action="store_true",
                        help=("Run the EBT-tiny break-even sweep — emits "
                              "apple_gpu + python_ref rows for each "
                              "(B, K, D, T) point in _DEFAULT_EBT_SWEEP and "
                              "summarizes break-even in the envelope."))
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON output path (stdout if omitted)")
    args = parser.parse_args(argv)

    reps = DEFAULT_REPS_CI if args.ci else args.reps
    include_primitives = not args.workloads_only
    include_workloads = not args.primitives_only
    report = run_report(reps=reps, refinement_T=args.refinement_T,
                        include_primitives=include_primitives,
                        include_workloads=include_workloads,
                        include_ebt_sweep=args.ebt_sweep)
    payload = json.dumps(report, indent=2, sort_keys=True, default=float)
    if args.output is not None:
        args.output.write_text(payload)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
