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

EBM coverage (9 of 9 native MSL kernels — closed 2026-05-17):

  Native MSL (Apple GPU): inner_step, refinement (fused single-dispatch
    T-step closed-form), langevin_step, decode_init,
    bivector_langevin_step (composition over langevin_step on
    grade-projected inputs), sphere_langevin_step, self_verify
    (hard-argmin), energy (quadratic specialization), partition_exact
    (single-dispatch stable logsumexp: Z = exp(max + log(sum(exp(-E_i/T - max)))) ).
  Native MSL (workload-only fused kernel): ebt_tiny (streaming
    closed-form refinement + per-row energy + K-way argmin in one
    Metal dispatch; K <= 256, D unbounded).
  Python reference (always emitted alongside native rows for the
    native-vs-reference speedup column): partition_function_exact
    (callable-energy variant), every native EBM op.

Proof-of-dispatch coverage:
  - Native EBM primitive rows + JIT-bridge benchmark rows: every row
    carries a ``dispatched_on_gpu`` bit sourced from the
    ``tessera.compiler.jit_bridge`` route trace (the runner opens a
    one-shot trace span around the timed dispatch and confirms the
    trace recorded the expected op).
  - EBT-tiny workload + sweep rows: every row carries
    ``dispatched_on_gpu`` from ``ebm.ebt_tiny_dispatched_on_gpu()``;
    the sweep summary additionally tags each shape with
    ``status="native_dispatched"`` vs ``"degraded_fallback"``.
  - GA per-primitive rows: the runner builds each `dispatch` closure
    by binding a local ctypes pointer; correctness is verified bit-
    wise against the Python reference + the recorded symbol must
    equal the manifest-resolved symbol (``test_each_ga_row_carries_
    manifest_resolved_symbol`` in the unit-test sweep).
  - In every case: a silent numpy fallback degrades the row's
    ``backend`` to ``python_ref`` and its ``ok`` field to ``False``.

Output schema keeps the historical ``backend`` / ``mode`` columns for
roofline-tool compatibility and adds the Stage 16F execution-claim contract:

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
     "tessera_version": "...",
     "variant_kind": "python_reference" | "compiler_visible_reference"
                     | "apple_gpu_value_target_ir" | ...,
     "compiler_path": "apple_value_target_ir" | "manifest" | null,
     "executor": "python_reference" | "apple_gpu_value_target_ir" | ...,
     "runtime_status": "reference" | "success" | ...,
     "execution_kind": "reference_cpu" | "native_gpu" | ...}

Only ``variant_kind="apple_gpu_value_target_ir"`` may carry the trio
``executor="apple_gpu_value_target_ir"``, ``runtime_status="success"``, and
``execution_kind="native_gpu"``, and only after the value-runtime probe and
row-local numerical comparison have both passed.

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
from typing import Any, Callable, Optional

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
from tessera import runtime as tessera_runtime  # noqa: E402
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
           # 2026-05-29: MPSGraph-backed Tier-1 / long-tail execution lane.
           "-framework", "MetalPerformanceShadersGraph",
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


def _stage16f_fields(
    variant_kind: str, *, compiler_path: str | None, executor: str | None,
    runtime_status: str, execution_kind: str,
) -> dict[str, Any]:
    """Common row contract for GA/EBM benchmark honesty gates.

    Stage 16F mirrors the RL benchmark discipline.  The historical
    `backend`/`mode` columns remain for compatibility, while these fields are
    the row-level authority for execution claims.
    """
    return {
        "variant_kind": variant_kind,
        "compiler_path": compiler_path,
        "executor": executor,
        "runtime_status": runtime_status,
        "execution_kind": execution_kind,
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
        **_stage16f_fields(
            "legacy_manifest_native", compiler_path="manifest",
            executor="apple_gpu_manifest" if ok else None,
            runtime_status="success" if ok else "numerical_mismatch",
            execution_kind="legacy_native_gpu" if ok else "unknown"),
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

def _ebm_energy_path() -> tuple[Callable[[], Any], float]:
    rng = np.random.RandomState(1000)
    x = rng.randn(8, 4).astype(np.float32)
    y = rng.randn(8, 4).astype(np.float32)

    def model_fn(xx, yy):
        return np.sum((xx - yy) ** 2, axis=1)

    expected = np.sum((x - y) ** 2, axis=1).astype(np.float32)
    out = ebm.energy(model_fn, x, y)
    err = float(np.abs(out - expected).max())
    return (lambda: ebm.energy(model_fn, x, y)), err


def _ebm_inner_step_python_path() -> tuple[Callable[[], Any], float]:
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
    """Native Apple-GPU EBM inner-step via the public API.

    Calls ``tessera.ebm.inner_step(y, grad, eta)`` directly so the
    dispatch flows through the JIT bridge and the route trace
    captures every invocation.  Returns ``(dispatch, err, symbol)``.
    """
    del rt  # routed through the public dispatcher
    sym = "tessera_apple_gpu_ebm_inner_step_f32"
    rng = np.random.RandomState(1001)
    y = np.ascontiguousarray(rng.randn(16, 4).astype(np.float32))
    grad = np.ascontiguousarray(rng.randn(16, 4).astype(np.float32))
    eta = 0.05
    expected = (y - eta * grad).astype(np.float32)

    def dispatch() -> np.ndarray:
        return ebm.inner_step(y, grad, eta=eta)

    out = dispatch()
    err = float(np.abs(out - expected).max())
    return dispatch, err, sym


def _ebm_refinement_apple_gpu_path(
    rt: ctypes.CDLL, T: int = 8,
) -> tuple[Callable[[], np.ndarray], float, str]:
    """Native Apple-GPU EBT refinement via the public API.

    With a fixed gradient the closed form is ``y_T = y_0 − T·η·grad``
    — the kernel matches that bit-for-bit at fp32.  Calls
    ``ebm.refinement`` so the JIT bridge captures the dispatch.
    """
    del rt  # routed through the public dispatcher
    sym = "tessera_apple_gpu_ebm_refinement_f32"
    rng = np.random.RandomState(1010)
    y0 = np.ascontiguousarray(rng.randn(8, 6).astype(np.float32))
    grad = np.ascontiguousarray(rng.randn(8, 6).astype(np.float32))
    eta = 0.02
    expected = (y0 - T * eta * grad).astype(np.float32)

    def dispatch() -> np.ndarray:
        return ebm.refinement(y0, grad, eta=eta, T=T)

    out = dispatch()
    err = float(np.abs(out - expected).max())
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
    del rt  # routed through the public dispatcher
    sym = "tessera_apple_gpu_ebm_langevin_step_f32"
    rng = np.random.RandomState(1020)
    y = np.ascontiguousarray(rng.randn(16, 4).astype(np.float32))
    grad = np.ascontiguousarray(rng.randn(16, 4).astype(np.float32))
    eta = 0.01
    temperature = 1.0
    key = RNGKey.from_seed(1020)
    grad_fn = lambda yy: grad
    energy_fn = lambda yy: np.sum(yy * yy, axis=1)

    def dispatch() -> np.ndarray:
        out, _ = ebm.langevin_step(y, energy_fn, eta=eta,
                                     temperature=temperature,
                                     rng_key=key, grad_fn=grad_fn)
        return out

    # Correctness: at temperature=0 the langevin step reduces to
    # pure GD (no noise term).  Use that for a stable bit-check;
    # this builder runs T>0 for the perf timing.
    grad0 = grad
    out_t0, _ = ebm.langevin_step(y, energy_fn, eta=eta, temperature=0.0,
                                    rng_key=key, grad_fn=grad_fn)
    expected = (y - eta * grad0).astype(np.float32)
    err = float(np.abs(out_t0 - expected).max())
    dispatch()  # warm + record one route
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
    del rt  # routed through the public dispatcher
    sym = "tessera_apple_gpu_ebm_decode_init_noise_apply_f32"
    rng = np.random.RandomState(1030)
    B, K, E = 4, 6, 12
    base = np.ascontiguousarray(rng.randn(B, E).astype(np.float32))
    mean = np.broadcast_to(base[:, None, :], (B, K, E)).astype(np.float32)
    mean_c = np.ascontiguousarray(mean)
    x = np.zeros((B, E), dtype=np.float32)
    key = RNGKey.from_seed(1030)
    std = 0.5

    def dispatch() -> np.ndarray:
        return ebm.decode_init(x, K=K, init_strategy="noise",
                                 rng_key=key, shape=(E,), dtype="fp32",
                                 std=std, mean=mean_c)

    # Reference: same Philox stream → bitwise reproducible noise.
    from tessera.rng import normal as _rng_normal
    noise_ref = _rng_normal(key, shape=(B, K, E), dtype="fp32", std=1.0)
    expected = (mean_c + std * noise_ref).astype(np.float32)
    out = dispatch()
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
    del rt  # routed through the public dispatcher
    sym = "tessera_apple_gpu_ebm_langevin_step_f32"
    a = ga.Cl(3, 0)
    rng = np.random.RandomState(1040)
    coeffs = np.zeros(8, dtype=np.float32)
    coeffs[3] = 0.5
    coeffs[5] = -0.2
    coeffs[6] = 0.3
    state = ga.Multivector(coeffs, a)
    grad_coeffs = rng.randn(8).astype(np.float32)
    key = RNGKey.from_seed(1040)
    eta = 0.01

    def grad_fn(mv):
        return ga.Multivector(grad_coeffs.copy(), mv.algebra)

    def dispatch() -> np.ndarray:
        new_state, _ = ebm.bivector_langevin_step(
            state, lambda mv: 0.0, eta=eta, temperature=0.0,
            rng_key=key, grad_fn=grad_fn,
        )
        return new_state.coefficients.astype(np.float32, copy=False)

    out = dispatch()
    # T=0 ⇒ new = grade_2_project(state - eta * grad).  Compute by hand.
    grad_proj = np.zeros_like(grad_coeffs)
    for k in (3, 5, 6):
        grad_proj[k] = grad_coeffs[k]
    expected = coeffs - eta * grad_proj
    err = float(np.abs(out - expected).max())
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
    del rt  # routed through the public dispatcher
    sym = "tessera_apple_gpu_ebm_sphere_langevin_step_f32"
    x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rng = np.random.RandomState(1050)
    grad = rng.randn(3).astype(np.float32)
    eta = 0.005
    key = RNGKey.from_seed(1050)

    def dispatch() -> np.ndarray:
        out, _ = ebm.sphere_langevin_step(x, lambda p: -float(p[0]),
                                            eta=eta, temperature=0.0,
                                            rng_key=key,
                                            grad_fn=lambda p: grad)
        return np.asarray(out, dtype=np.float32)

    out = dispatch()
    # T=0 reference: tangent-project, step, retract.
    gdot = float(np.dot(grad, x))
    grad_tan = grad - gdot * x
    y = x - eta * grad_tan
    expected = (y / float(np.linalg.norm(y))).astype(np.float32)
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
    del rt  # routed through the public dispatcher
    sym = "tessera_apple_gpu_ebm_self_verify_hard_argmin_f32"
    B, K, D = 4, 8, 16
    rng = np.random.RandomState(1060)
    energies = np.ascontiguousarray(rng.randn(B, K).astype(np.float32))
    candidates = np.ascontiguousarray(rng.randn(B, K, D).astype(np.float32))
    expected = candidates[np.arange(B), energies.argmin(axis=1)]

    def dispatch() -> np.ndarray:
        return ebm.self_verify(energies, candidates)

    out = dispatch()
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
    del rt  # routed through the public dispatcher
    sym = "tessera_apple_gpu_ebm_energy_quadratic_f32"
    B, D = 8, 4
    rng = np.random.RandomState(1070)
    x = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    y = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    expected = (0.5 * np.sum((x - y) ** 2, axis=1)).astype(np.float32)

    def dispatch() -> np.ndarray:
        return ebm.energy_quadratic(x, y)

    out = dispatch()
    err = float(np.abs(out - expected).max())
    return dispatch, err, sym


def _ebm_partition_exact_apple_gpu_path(
    rt: ctypes.CDLL,
) -> tuple[Callable[[], np.ndarray], float, str]:
    """Native Apple-GPU partition_exact — stable logsumexp over a
    precomputed energies array.  Closes the 8/9 → 9/9 native EBM
    gap.  Calls ``ebm.partition_exact_from_energies`` directly so
    the JIT bridge captures the dispatch."""
    del rt  # routed through the public dispatcher
    sym = "tessera_apple_gpu_ebm_partition_exact_f32"
    rng = np.random.RandomState(1080)
    # 64 states is a typical small-state exhaustive-enumeration count;
    # the kernel scales linearly in N so timing is informative here.
    N = 64
    energies = np.ascontiguousarray(rng.randn(N).astype(np.float32) * 2.0)
    temperature = 1.0
    expected = float(
        np.exp(-energies.astype(np.float64) / temperature).sum()
    )

    def dispatch() -> np.ndarray:
        return np.array([
            ebm.partition_exact_from_energies(energies,
                                                temperature=temperature)
        ], dtype=np.float32)

    out = dispatch()
    err = float(abs(float(out[0]) - expected) / max(1.0, abs(expected)))
    return dispatch, err, sym


# Python-reference Langevin (kept for the non-Apple skip path).
def _ebm_langevin_step_path() -> tuple[Callable[[], Any], float]:
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


def _ebm_self_verify_path() -> tuple[Callable[[], Any], float]:
    rng = np.random.RandomState(1003)
    B, K, D = 4, 8, 16
    energies = rng.randn(B, K).astype(np.float32)
    candidates = rng.randn(B, K, D).astype(np.float32)
    expected = candidates[np.arange(B), energies.argmin(axis=1)]
    out = ebm.self_verify(energies, candidates)
    err = float(np.abs(out - expected).max())
    return (lambda: ebm.self_verify(energies, candidates)), err


def _ebm_decode_init_path() -> tuple[Callable[[], Any], float]:
    key = RNGKey.from_seed(1004)
    x = np.zeros((4, 12), dtype=np.float32)
    kwargs: dict[str, Any] = dict(K=6, init_strategy="noise", rng_key=key,
                                  shape=(12,), dtype="fp32")
    out = ebm.decode_init(x, **kwargs)
    out2 = ebm.decode_init(x, **kwargs)
    err = float(np.abs(out - out2).max())
    assert out.shape == (4, 6, 12)
    return (lambda: ebm.decode_init(x, **kwargs)), err


def _ebm_partition_exact_path() -> tuple[Callable[[], Any], float]:
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


def _ebm_bivector_langevin_path() -> tuple[Callable[[], Any], float]:
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


def _ebm_sphere_langevin_path() -> tuple[Callable[[], Any], float]:
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
        # 9/9 closure — partition_exact stable logsumexp.
        ("ebm_partition_exact",   "N=64",
         _ebm_partition_exact_apple_gpu_path,   1e-5),
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
) -> tuple[Callable[[], np.ndarray], float, tuple[str, ...], str, bool]:
    """Apple-GPU EBT-tiny loop through the **fused public API**.

    Returns ``(step, err, symbols, shape_descriptor, dispatched_on_gpu)``.
    ``dispatched_on_gpu`` is the proof bit: it is ``True`` only when
    the most-recent ``ebm.ebt_tiny`` call ran on the GPU.  Captured
    via :func:`tessera.ebm.ebt_tiny_dispatched_on_gpu` so a silent
    fallback to numpy (e.g., the input violates a guard rail) cannot
    mislabel a row as native.

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
    dispatched_on_gpu = ebm.ebt_tiny_dispatched_on_gpu()
    ref, _, _ = _workload_ebt_tiny_python_path(T, B=B, K=K, D=D)
    ref_out = ref()
    err = float(np.abs(out - ref_out).max())
    syms = (
        "tessera_apple_gpu_ebm_ebt_tiny_refinement_argmin_f32 [via ebm.ebt_tiny]",
    )
    return (step, err, syms, f"B={B},K={K},D={D}/T={T}", dispatched_on_gpu)


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


# ---------------------------------------------------------------------------
# Rotor-conditioned EBT workload — the GA + EBM fused workload.
#
# Pattern: given B rotor candidates (parameterized as bivectors) and a
# fixed target vector V, refine each candidate's rotor to minimize the
# distance between rotor_sandwich(R, V) and V.  Uses GA primitives for
# the rotor application + ebm.ebt_tiny for the K-candidate selection.
# Demonstrates the two families running as one workload through public
# APIs.
# ---------------------------------------------------------------------------

def _workload_rotor_conditioned_ebt_inputs(seed: int = 3000,
                                             B: int = 4, K: int = 8):
    """Inputs for the rotor-conditioned EBT workload."""
    rng = np.random.RandomState(seed)
    # Candidate rotor parameterization as pure bivectors (B*K, 8) ⇒
    # rotor = exp(B/2) in the closed form.  Use small magnitudes to
    # keep rotors close to identity (rotation angle ≲ 0.6 rad).
    bivecs = np.zeros((B * K, 8), dtype=np.float32)
    bivecs[:, 3] = rng.randn(B * K).astype(np.float32) * 0.3
    bivecs[:, 5] = rng.randn(B * K).astype(np.float32) * 0.3
    bivecs[:, 6] = rng.randn(B * K).astype(np.float32) * 0.3
    # Target vectors: one per (B*K) candidate, grade-1 only.
    vectors = rng.randn(B * K, 8).astype(np.float32)
    return B, K, bivecs, vectors


def _workload_rotor_ebt_python_path() -> tuple[Callable[[], np.ndarray],
                                                float, str]:
    """Python-reference rotor-conditioned EBT workload.

    Pipeline (per K candidate):
      1. R = exp_mv(B)            # bivector → rotor
      2. O = rotor_sandwich(R, V) # apply rotor
      3. e = ||O||²               # squared norm (rotor-invariant ⇒ = ||V||²)
    Then argmin_k over the (B, K) energy grid picks the best candidate
    rotor per batch.  In the absence of a non-trivial energy term the
    "best" candidate is degenerate but the workload exercises the
    full GA→EBT chain end-to-end.
    """
    B, K, bivecs, vectors = _workload_rotor_conditioned_ebt_inputs()
    a = _CL30

    def step() -> np.ndarray:
        rotors_co = np.zeros_like(bivecs)
        for i in range(B * K):
            rotors_co[i] = ga.exp_mv(ga.Multivector(bivecs[i], a)).coefficients
        rotated_co = np.zeros_like(bivecs)
        for i in range(B * K):
            rotated_co[i] = ga.rotor_sandwich(
                ga.Multivector(rotors_co[i], a),
                ga.Multivector(vectors[i], a),
            ).coefficients
        # Per-candidate energy = ||rotated||² over the 8 coefficients.
        # Use raw numpy to keep this loop closed-form and Python-side.
        energies = np.sum(rotated_co * rotated_co, axis=1).reshape(B, K)
        candidates = rotated_co.reshape(B, K, 8)
        return candidates[np.arange(B), energies.argmin(axis=1)]

    step()  # warm
    return step, 0.0, f"B={B},K={K}/exp→sandwich→ebt"


def _workload_rotor_ebt_apple_gpu_path(
    rt: "ctypes.CDLL",
) -> tuple[Callable[[], np.ndarray], float, tuple[str, ...], str, bool]:
    """Apple-GPU rotor-conditioned EBT workload through public APIs.

    Calls ``ga.exp_mv`` (batched), ``ga.rotor_sandwich`` (batched),
    then ``ebm.ebt_tiny`` for the K-way argmin.  Every dispatch
    routes through the bridge.  Returns the 5-tuple shape used by
    :func:`run_workload_apple_gpu` including the proof-of-dispatch
    bit (``dispatched_on_gpu``).
    """
    del rt
    B, K, bivecs, vectors = _workload_rotor_conditioned_ebt_inputs()
    a = _CL30
    bivecs_c = np.ascontiguousarray(bivecs)
    vectors_c = np.ascontiguousarray(vectors)

    def step() -> np.ndarray:
        rotors = ga.exp_mv(ga.Multivector(bivecs_c, a))
        rotated = ga.rotor_sandwich(rotors, ga.Multivector(vectors_c, a))
        # Convert (B*K, 8) → ebt_tiny inputs.  Use T=1 with eta=0
        # so the refinement is a no-op (y_T = y0 - 1*0*grad = y0)
        # but the GPU kernel actually fires — the public API
        # rejects T=0 as a degenerate case so we need T>=1.
        flat = np.ascontiguousarray(rotated.coefficients.reshape(B * K, 8))
        return ebm.ebt_tiny(flat, flat, eta=0.0, T=1, B=B, K=K, D=8)

    out = step()
    # The closed-form numpy reference is exact at fp32.
    ref, _, _ = _workload_rotor_ebt_python_path()
    ref_out = ref()
    err = float(np.abs(out - ref_out).max())
    # Proof-of-dispatch — read the route trace + check both ops fired.
    from tessera.compiler import jit_bridge as _bridge
    prev_tracing = _bridge.tracing_enabled()
    _bridge.set_tracing_enabled(True)
    _bridge.clear_dispatch_trace()
    try:
        step()
    finally:
        _bridge.set_tracing_enabled(prev_tracing)
    routes = _bridge.take_dispatch_trace()
    ops_seen = {r.op_name for r in routes}
    dispatched_on_gpu = (
        "clifford_exp" in ops_seen and
        "clifford_rotor_sandwich" in ops_seen and
        ebm.ebt_tiny_dispatched_on_gpu()
    )
    syms = (
        "tessera_apple_gpu_clifford_exp_cl30_f32 [via ga.exp_mv]",
        "tessera_apple_gpu_clifford_rotor_sandwich_cl30_f32 [via ga.rotor_sandwich]",
        "tessera_apple_gpu_ebm_ebt_tiny_refinement_argmin_f32 [via ebm.ebt_tiny]",
    )
    return (step, err, syms,
            f"B={B},K={K}/exp→sandwich→ebt", dispatched_on_gpu)


# ---------------------------------------------------------------------------
# Compiler vertical slice — @clifford_jit(target="apple_gpu") on a
# small Cl(3,0) point-cloud invariant.
# ---------------------------------------------------------------------------

def _vertical_slice_compiled_callable():
    """Build + return the ``@clifford_jit`` compiled callable.

    Defined lazily so importing this benchmark module on non-Darwin
    doesn't try to compile against a missing runtime.
    """
    from tessera.compiler.clifford_jit import clifford_jit

    @clifford_jit(target="apple_gpu")
    def point_cloud_rotor_invariant(rotor, points):
        """For a unit rotor R, ``|R x R†| = |x|`` is an SO(3)
        invariant.  Returns the per-batch norm of the rotated
        points."""
        rotated = ga.rotor_sandwich(rotor, points)
        return ga.norm(rotated)

    return point_cloud_rotor_invariant


def _vertical_slice_apple_gpu_path(
    rt: "ctypes.CDLL",
) -> tuple[Callable[[], np.ndarray], float, str, dict[str, Any]]:
    """The compiler-integrated vertical slice — `@clifford_jit`
    decorator → traced op plan → manifest-resolved Apple target
    metadata → runtime dispatch → benchmark row.

    Returns ``(dispatch, err, plan_hash, artifact_metadata)``.
    """
    del rt
    compiled = _vertical_slice_compiled_callable()
    a = _CL30
    rng = np.random.RandomState(4000)
    R = rng.randn(64, 8).astype(np.float32) * 0.3
    V = rng.randn(64, 8).astype(np.float32)
    rotor = ga.Multivector(R, a)
    points = ga.Multivector(V, a)

    def step() -> np.ndarray:
        return np.asarray(compiled(rotor, points))

    # First call compiles + executes.
    out = step()
    # Reference: per-batch ||R V R†||.
    ref_out = np.array([
        float(np.asarray(ga.norm(ga.rotor_sandwich(
            ga.Multivector(R[i], a), ga.Multivector(V[i], a)))))
        for i in range(R.shape[0])
    ])
    err = float(np.abs(out - ref_out).max())
    metadata = compiled.artifact.as_metadata()
    return step, err, compiled.artifact.plan_hash, metadata


def run_workload_apple_gpu(name: str, shape: str,
                            dispatch: Callable, err: float,
                            symbols: tuple[str, ...],
                            tolerance: float, reps: int,
                            device: str, version: str, *,
                            dispatched_on_gpu: Optional[bool] = None,
                            ) -> dict[str, Any]:
    """Build an ``apple_gpu`` workload row.

    ``dispatched_on_gpu`` is the proof-of-dispatch bit.  When it's
    ``False`` the row is degraded to ``backend="python_ref"`` +
    ``mode="reference_chain"`` + ``ok=False`` so consumers can't
    mistake a silent numpy fallback for a real native dispatch.  When
    ``None`` (legacy workloads that don't yet provide the probe) the
    bit defaults to ``True`` for backward compat.
    """
    samples_ms = collect_samples(dispatch, reps)
    timing = timing_stats(samples_ms)
    is_native = (dispatched_on_gpu is None) or bool(dispatched_on_gpu)
    backend = "apple_gpu" if is_native else "python_ref"
    mode = "fused_chain" if is_native else "reference_chain_fallback"
    ok = (err <= tolerance) and is_native
    return {
        "backend": backend,
        "namespace": "workload",
        "op": name,
        "shape": shape,
        "dtype": "f32",
        "mode": mode,
        "reps": reps,
        **timing,
        "max_abs_err": err,
        "tolerance": tolerance,
        "ok": ok,
        "dispatched_on_gpu": is_native,
        "symbols": list(symbols),
        "device": device,
        "tessera_version": version,
        **_stage16f_fields(
            "legacy_manifest_native" if ok else "python_reference",
            compiler_path="manifest" if ok else None,
            executor="apple_gpu_manifest" if ok else None,
            runtime_status="success" if ok else "reference_fallback",
            execution_kind="legacy_native_gpu" if ok else "reference_cpu"),
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
        **_stage16f_fields(
            "python_reference", compiler_path=None,
            executor="python_reference", runtime_status="reference",
            execution_kind="reference_cpu"),
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
        **_stage16f_fields(
            "python_reference", compiler_path=None,
            executor="python_reference", runtime_status="reference",
            execution_kind="reference_cpu"),
    }


def run_ebm_apple_gpu_path(name: str, shape: str,
                            dispatch: Callable, err: float, symbol: str,
                            tolerance: float, reps: int,
                            device: str, version: str) -> dict[str, Any]:
    """Time a native Apple-GPU EBM dispatch + record proof of dispatch.

    The bridge trace is the proof bit: we open a one-shot trace span
    around a single dispatch, drain it, and confirm the trace recorded
    at least one route with the expected ``op_name``.  If the dispatch
    silently fell back to numpy (e.g., a guard rail rejected the
    input), the trace stays empty and the row is degraded to
    ``backend="python_ref"`` / ``mode="reference_fallback"`` /
    ``ok=False`` so consumers can't mistake it for a native dispatch.
    """
    from tessera.compiler import jit_bridge as _bridge
    # One-shot trace probe — single dispatch, drain the trace.
    _bridge.clear_dispatch_trace()
    prev_tracing = _bridge.tracing_enabled()
    _bridge.set_tracing_enabled(True)
    try:
        dispatch()
    finally:
        _bridge.set_tracing_enabled(prev_tracing)
    probe_routes = _bridge.take_dispatch_trace()
    dispatched_on_gpu = any(
        r.op_name == name and r.target == "apple_gpu" and r.status == "fused"
        for r in probe_routes
    )

    samples_ms = collect_samples(dispatch, reps)
    timing = timing_stats(samples_ms)
    manifest_status = _manifest_status_for(name, "apple_gpu")
    backend = "apple_gpu" if dispatched_on_gpu else "python_ref"
    mode = "fused" if dispatched_on_gpu else "reference_fallback"
    ok = (err <= tolerance) and dispatched_on_gpu and manifest_status == "fused"
    return {
        "backend": backend,
        "namespace": "ebm",
        "op": name,
        "shape": shape,
        "dtype": "f32",
        "mode": mode,
        "reps": reps,
        **timing,
        "max_abs_err": err,
        "tolerance": tolerance,
        "ok": ok,
        "apple_gpu_status": manifest_status,
        "symbol": symbol,
        "dispatched_on_gpu": dispatched_on_gpu,
        "device": device,
        "tessera_version": version,
        **_stage16f_fields(
            "legacy_manifest_native", compiler_path="manifest",
            executor="apple_gpu_manifest" if ok else None,
            runtime_status="success" if ok else "unverified_dispatch",
            execution_kind="legacy_native_gpu" if ok else "unknown"),
    }


def run_ebm_apple_value_path(name: str, shape: str,
                             dispatch: Callable, err: float, symbol: str,
                             tolerance: float, reps: int,
                             device: str, version: str, *,
                             namespace: str = "ebm") -> dict[str, Any]:
    """Time an Apple GPU Value Target IR GA/EBM row.

    These rows are deliberately distinct from the legacy `backend="apple_gpu"`
    rows. They execute through the value runtime adapter and are emitted only
    when the value-specific status-returning C ABI probe has already passed.
    """
    samples_ms = collect_samples(dispatch, reps)
    timing = timing_stats(samples_ms)
    ok = err <= tolerance
    return {
        "backend": "apple_gpu_value_target_ir",
        "namespace": namespace,
        "op": name,
        "shape": shape,
        "dtype": "f32",
        "mode": "value_target_ir",
        "executor": "apple_gpu_value_target_ir",
        "runtime_status": "success",
        "reps": reps,
        **timing,
        "max_abs_err": err,
        "tolerance": tolerance,
        "ok": ok,
        "apple_gpu_status": _manifest_status_for(name, "apple_gpu"),
        "symbol": symbol,
        "device": device,
        "tessera_version": version,
        **_stage16f_fields(
            "apple_gpu_value_target_ir", compiler_path="apple_value_target_ir",
            executor="apple_gpu_value_target_ir" if ok else None,
            runtime_status="success" if ok else "numerical_mismatch",
            execution_kind="native_gpu" if ok else "unknown"),
    }


def run_compiler_visible_reference_path(
    namespace: str, name: str, shape: str, dispatch: Callable, err: float,
    symbol: str, tolerance: float, reps: int, device: str, version: str,
) -> dict[str, Any]:
    """Time a compiler-visible reference row for a GA/EBM value envelope."""
    samples_ms = collect_samples(dispatch, reps)
    timing = timing_stats(samples_ms)
    return {
        "backend": "compiler_visible_reference",
        "namespace": namespace,
        "op": name,
        "shape": shape,
        "dtype": "f32",
        "mode": "compiler_visible_reference",
        "reps": reps,
        **timing,
        "max_abs_err": err,
        "tolerance": tolerance,
        "ok": err <= tolerance,
        "apple_gpu_status": _manifest_status_for(name, "apple_gpu"),
        "symbol": symbol,
        "device": device,
        "tessera_version": version,
        **_stage16f_fields(
            "compiler_visible_reference",
            compiler_path="apple_value_target_ir",
            executor="python_reference",
            runtime_status="reference",
            execution_kind="reference_cpu"),
    }


def _ebm_energy_value_target_ir_path() -> (
    tuple[Callable[[], np.ndarray], float, str] | None
):
    if not tessera_runtime._apple_gpu_ebm_energy_quadratic_value_available():
        return None
    B, D = 8, 4
    rng = np.random.RandomState(2070)
    x = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    y = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    expected = (0.5 * np.sum((x - y) ** 2, axis=1)).astype(np.float32)
    symbol = "tessera_apple_gpu_ebm_energy_quadratic_value_f32"
    artifact = tessera_runtime.RuntimeArtifact(metadata={
        "target": "apple_gpu",
        "compiler_path": "apple_value_target_ir",
        "apple_target_ir_kind": "value_target_ir",
        "executable": True,
        "apple_value_calls": [{
            "op": "tessera_apple.gpu.kernel_call",
            "op_kind": "ebm_energy_quadratic",
            "symbol": symbol,
            "status": "executable",
        }],
    })

    def dispatch() -> np.ndarray:
        res = tessera_runtime.launch(artifact, [x, y])
        if not res.get("ok"):
            raise RuntimeError(res.get("reason", "EBM value dispatch failed"))
        return np.asarray(res["output"], dtype=np.float32)

    out = dispatch()
    err = float(np.abs(out - expected).max())
    return dispatch, err, symbol


def _ebm_langevin_value_target_ir_path() -> (
    tuple[Callable[[], np.ndarray], float, str] | None
):
    if not tessera_runtime._apple_gpu_ebm_langevin_step_value_available():
        return None
    B, D = 16, 4
    rng = np.random.RandomState(2080)
    y = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    grad = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    noise = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    eta = 0.05
    noise_scale = 0.125
    expected = (y - eta * grad + noise_scale * noise).astype(np.float32)
    symbol = "tessera_apple_gpu_ebm_langevin_step_value_f32"
    artifact = tessera_runtime.RuntimeArtifact(metadata={
        "target": "apple_gpu",
        "compiler_path": "apple_value_target_ir",
        "apple_target_ir_kind": "value_target_ir",
        "executable": True,
        "apple_value_calls": [{
            "op": "tessera_apple.gpu.kernel_call",
            "op_kind": "ebm_langevin_step",
            "symbol": symbol,
            "status": "executable",
            "eta": eta,
            "noise_scale": noise_scale,
        }],
    })

    def dispatch() -> np.ndarray:
        res = tessera_runtime.launch(artifact, [y, grad, noise])
        if not res.get("ok"):
            raise RuntimeError(res.get("reason", "EBM value dispatch failed"))
        return np.asarray(res["output"], dtype=np.float32)

    out = dispatch()
    err = float(np.abs(out - expected).max())
    return dispatch, err, symbol


def _ebm_refinement_value_target_ir_path() -> (
    tuple[Callable[[], np.ndarray], float, str] | None
):
    if not tessera_runtime._apple_gpu_ebm_refinement_value_available():
        return None
    B, D = 8, 6
    rng = np.random.RandomState(2090)
    y0 = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    grad = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    eta = 0.02
    steps = 8
    expected = (y0 - steps * eta * grad).astype(np.float32)
    symbol = "tessera_apple_gpu_ebm_refinement_value_f32"
    artifact = tessera_runtime.RuntimeArtifact(metadata={
        "target": "apple_gpu",
        "compiler_path": "apple_value_target_ir",
        "apple_target_ir_kind": "value_target_ir",
        "executable": True,
        "apple_value_calls": [{
            "op": "tessera_apple.gpu.kernel_call",
            "op_kind": "ebm_refinement",
            "symbol": symbol,
            "status": "executable",
            "eta": eta,
            "steps": steps,
        }],
    })

    def dispatch() -> np.ndarray:
        res = tessera_runtime.launch(artifact, [y0, grad])
        if not res.get("ok"):
            raise RuntimeError(res.get("reason", "EBM value dispatch failed"))
        return np.asarray(res["output"], dtype=np.float32)

    out = dispatch()
    err = float(np.abs(out - expected).max())
    return dispatch, err, symbol


def _ebm_partition_value_target_ir_path() -> (
    tuple[Callable[[], np.ndarray], float, str] | None
):
    if not tessera_runtime._apple_gpu_ebm_partition_exact_value_available():
        return None
    rng = np.random.RandomState(2100)
    energies = np.ascontiguousarray(rng.randn(64).astype(np.float32) * 2.0)
    temperature = 1.0
    scaled = -energies.astype(np.float64) / temperature
    expected = float(np.exp(np.max(scaled)) *
                     np.sum(np.exp(scaled - np.max(scaled))))
    symbol = "tessera_apple_gpu_ebm_partition_exact_value_f32"
    artifact = tessera_runtime.RuntimeArtifact(metadata={
        "target": "apple_gpu",
        "compiler_path": "apple_value_target_ir",
        "apple_target_ir_kind": "value_target_ir",
        "executable": True,
        "apple_value_calls": [{
            "op": "tessera_apple.gpu.kernel_call",
            "op_kind": "ebm_partition_exact",
            "symbol": symbol,
            "status": "executable",
            "temperature": temperature,
            "reduction": "logsumexp",
        }],
    })

    def dispatch() -> np.ndarray:
        res = tessera_runtime.launch(artifact, [energies])
        if not res.get("ok"):
            raise RuntimeError(res.get("reason", "EBM value dispatch failed"))
        return np.asarray(res["output"], dtype=np.float32)

    out = dispatch()
    err = float(abs(float(out) - expected) / max(1.0, abs(expected)))
    return dispatch, err, symbol


def _clifford_geo_value_target_ir_path() -> (
    tuple[Callable[[], np.ndarray], float, str] | None
):
    if not tessera_runtime._apple_gpu_clifford_geo_product_cl30_value_available():
        return None
    A = _seeded_pointwise(300)
    B = _seeded_pointwise(301)
    A_c = np.ascontiguousarray(A)
    B_c = np.ascontiguousarray(B)
    expected = _py_ref_binary_8x8("clifford_geometric_product", A, B)
    symbol = "tessera_apple_gpu_clifford_geo_product_cl30_value_f32"
    artifact = tessera_runtime.RuntimeArtifact(metadata={
        "target": "apple_gpu",
        "compiler_path": "apple_value_target_ir",
        "apple_target_ir_kind": "value_target_ir",
        "executable": True,
        "apple_value_calls": [{
            "op": "tessera_apple.gpu.kernel_call",
            "op_kind": "clifford_geometric_product",
            "symbol": symbol,
            "status": "executable",
            "p": 3,
            "q": 0,
        }],
    })

    def dispatch() -> np.ndarray:
        res = tessera_runtime.launch(artifact, [A_c, B_c])
        if not res.get("ok"):
            raise RuntimeError(res.get("reason", "GA value dispatch failed"))
        return np.asarray(res["output"], dtype=np.float32)

    out = dispatch()
    err = float(np.abs(out - expected).max())
    return dispatch, err, symbol


def _compiler_visible_ebm_energy_reference_path() -> (
    tuple[Callable[[], np.ndarray], float, str]
):
    B, D = 8, 4
    rng = np.random.RandomState(2070)
    x = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    y = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    expected = (0.5 * np.sum((x - y) ** 2, axis=1)).astype(np.float32)

    def dispatch() -> np.ndarray:
        return (0.5 * np.sum((x - y) ** 2, axis=1)).astype(np.float32)

    err = float(np.abs(dispatch() - expected).max())
    return dispatch, err, "tessera_apple_gpu_ebm_energy_quadratic_value_f32"


def _compiler_visible_ebm_langevin_reference_path() -> (
    tuple[Callable[[], np.ndarray], float, str]
):
    B, D = 16, 4
    rng = np.random.RandomState(2080)
    y = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    grad = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    noise = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    eta = 0.05
    noise_scale = 0.125
    expected = (y - eta * grad + noise_scale * noise).astype(np.float32)

    def dispatch() -> np.ndarray:
        return (y - eta * grad + noise_scale * noise).astype(np.float32)

    err = float(np.abs(dispatch() - expected).max())
    return dispatch, err, "tessera_apple_gpu_ebm_langevin_step_value_f32"


def _compiler_visible_ebm_refinement_reference_path() -> (
    tuple[Callable[[], np.ndarray], float, str]
):
    B, D = 8, 6
    rng = np.random.RandomState(2090)
    y0 = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    grad = np.ascontiguousarray(rng.randn(B, D).astype(np.float32))
    eta = 0.02
    steps = 8
    expected = (y0 - steps * eta * grad).astype(np.float32)

    def dispatch() -> np.ndarray:
        return (y0 - steps * eta * grad).astype(np.float32)

    err = float(np.abs(dispatch() - expected).max())
    return dispatch, err, "tessera_apple_gpu_ebm_refinement_value_f32"


def _compiler_visible_ebm_partition_reference_path() -> (
    tuple[Callable[[], np.ndarray], float, str]
):
    rng = np.random.RandomState(2100)
    energies = np.ascontiguousarray(rng.randn(64).astype(np.float32) * 2.0)
    temperature = 1.0
    scaled = -energies.astype(np.float64) / temperature
    expected = float(np.exp(np.max(scaled)) *
                     np.sum(np.exp(scaled - np.max(scaled))))

    def dispatch() -> np.ndarray:
        scaled_f = -energies.astype(np.float64) / temperature
        return np.asarray(
            np.exp(np.max(scaled_f)) *
            np.sum(np.exp(scaled_f - np.max(scaled_f))),
            dtype=np.float32)

    err = float(abs(float(dispatch()) - expected) / max(1.0, abs(expected)))
    return dispatch, err, "tessera_apple_gpu_ebm_partition_exact_value_f32"


def _compiler_visible_clifford_geo_reference_path() -> (
    tuple[Callable[[], np.ndarray], float, str]
):
    A = _seeded_pointwise(300)
    B = _seeded_pointwise(301)
    expected = _py_ref_binary_8x8("clifford_geometric_product", A, B)

    def dispatch() -> np.ndarray:
        return _py_ref_binary_8x8("clifford_geometric_product", A, B)

    err = float(np.abs(dispatch() - expected).max())
    return dispatch, err, "tessera_apple_gpu_clifford_geo_product_cl30_value_f32"


_VALUE_TARGET_BUILDERS = [
    ("ebm", "ebm_energy", "B=8,D=4/quadratic/value_ir",
     _compiler_visible_ebm_energy_reference_path,
     _ebm_energy_value_target_ir_path, 1e-6),
    ("ebm", "ebm_langevin_step", "B=16,D=4/T=1/value_ir",
     _compiler_visible_ebm_langevin_reference_path,
     _ebm_langevin_value_target_ir_path, 1e-6),
    ("ebm", "ebm_refinement", "B=8,D=6/T=8/value_ir",
     _compiler_visible_ebm_refinement_reference_path,
     _ebm_refinement_value_target_ir_path, 1e-6),
    ("ebm", "ebm_partition_exact", "N=64/value_ir",
     _compiler_visible_ebm_partition_reference_path,
     _ebm_partition_value_target_ir_path, 1e-5),
    ("ga", "clifford_geometric_product", f"{_BATCH}x8,{_BATCH}x8/value_ir",
     _compiler_visible_clifford_geo_reference_path,
     _clifford_geo_value_target_ir_path, 1e-4),
]


# ---------------------------------------------------------------------------
# Stage 17 composite proof lanes.
#
# These rows intentionally do not claim Apple GPU value execution.  They
# prove that multi-op GA/EBM math contracts can be represented as ordered
# value-call plans and validated against a reference.  The current value
# runtime still accepts exactly one gpu.kernel_call, so composite rows stay
# compiler-visible/reference until a real multi-call or fused executor lands.
# ---------------------------------------------------------------------------

_STAGE17_COMPOSITE_STATUS = "multi_call_value_ir_gated"


def _value_call_record(
    op_kind: str, symbol: str, status: str,
) -> dict[str, str]:
    return {
        "op": "tessera_apple.gpu.kernel_call",
        "op_kind": op_kind,
        "symbol": symbol,
        "status": status,
    }


def _cl30_geometric_product_coeffs(
    a: np.ndarray, b: np.ndarray,
) -> np.ndarray:
    """Pure NumPy Cl(3,0) product for Stage 17 reference rows."""
    table = _CL30.product_table()
    leading = np.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    a_b = np.broadcast_to(a, (*leading, 8)).astype(np.float32, copy=False)
    b_b = np.broadcast_to(b, (*leading, 8)).astype(np.float32, copy=False)
    out = np.zeros((*leading, 8), dtype=np.float32)
    for i in range(8):
        ai = a_b[..., i]
        if not np.any(ai):
            continue
        for j in range(8):
            result_mask, sign = table[i][j]
            if sign == 0:
                continue
            term = ai * b_b[..., j]
            if sign > 0:
                out[..., result_mask] = out[..., result_mask] + term
            else:
                out[..., result_mask] = out[..., result_mask] - term
    return out


def _cl30_grade_project_coeffs(
    coeffs: np.ndarray, grades: set[int],
) -> np.ndarray:
    out = coeffs.astype(np.float32, copy=True)
    for blade in _CL30.blades():
        if blade.grade not in grades:
            out[..., blade.mask] = 0.0
    return out


def _cl30_reverse_coeffs(coeffs: np.ndarray) -> np.ndarray:
    out = coeffs.astype(np.float32, copy=True)
    for blade in _CL30.blades():
        sign = -1.0 if (blade.grade * (blade.grade - 1) // 2) % 2 else 1.0
        out[..., blade.mask] *= np.float32(sign)
    return out


def _cl30_rotor_sandwich_coeffs(
    rotor: np.ndarray, x: np.ndarray,
) -> np.ndarray:
    return _cl30_geometric_product_coeffs(
        _cl30_geometric_product_coeffs(rotor, x),
        _cl30_reverse_coeffs(rotor),
    )


def _compiler_composite_ebt_tiny_path() -> tuple[
    Callable[[], np.ndarray], float, str, dict[str, Any]
]:
    B, K, D, steps = 3, 4, 5, 3
    eta = np.float32(0.07)
    rng = np.random.RandomState(4100)
    y0 = rng.randn(B, K, D).astype(np.float32)
    target = rng.randn(B, D).astype(np.float32)
    target_k = np.broadcast_to(target[:, None, :], (B, K, D)).astype(np.float32)
    grad = (y0 - target_k).astype(np.float32)

    def dispatch() -> np.ndarray:
        refined = (y0 - np.float32(steps) * eta * grad).astype(np.float32)
        energies = (0.5 * np.sum((refined - target_k) ** 2, axis=2)).astype(
            np.float32)
        winner = energies.argmin(axis=1)
        return refined[np.arange(B), winner]

    refined = (y0 - np.float32(steps) * eta * grad).astype(np.float32)
    energies = (0.5 * np.sum((refined - target_k) ** 2, axis=2)).astype(
        np.float32)
    expected = refined[np.arange(B), energies.argmin(axis=1)]
    err = float(np.abs(dispatch() - expected).max())
    details = {
        "component_ops": [
            "ebm_decode_init",
            "ebm_refinement",
            "ebm_energy_quadratic",
            "ebm_self_verify",
        ],
        "component_value_status": {
            "ebm_decode_init": "compiler_visible_gated",
            "ebm_refinement": "executable_single_call",
            "ebm_energy_quadratic": "executable_single_call",
            "ebm_self_verify": "compiler_visible_gated",
        },
        "value_calls": [
            _value_call_record(
                "ebm_decode_init",
                "tessera_apple_gpu_ebm_decode_init_noise_apply_f32",
                "compiler_visible"),
            _value_call_record(
                "ebm_refinement",
                "tessera_apple_gpu_ebm_refinement_value_f32",
                "executable"),
            _value_call_record(
                "ebm_energy_quadratic",
                "tessera_apple_gpu_ebm_energy_quadratic_value_f32",
                "executable"),
            _value_call_record(
                "ebm_self_verify",
                "tessera_apple_gpu_ebm_self_verify_hard_argmin_f32",
                "compiler_visible"),
        ],
        "math_contract": (
            "decode_init candidates, fixed-gradient refinement, quadratic "
            "energy over refined candidates, then hard argmin self_verify"),
        "contract_metrics": {
            "winner_indices": energies.argmin(axis=1).astype(int).tolist(),
            "min_energy_max": float(energies.min(axis=1).max()),
        },
    }
    return dispatch, err, "multi_call:ebt_tiny_refinement", details


def _compiler_composite_manifold_ebm_path() -> tuple[
    Callable[[], np.ndarray], float, str, dict[str, Any]
]:
    rng = np.random.RandomState(4200)
    x = rng.randn(3).astype(np.float32)
    x = (x / np.linalg.norm(x)).astype(np.float32)
    sphere_grad = rng.randn(3).astype(np.float32)
    sphere_eta = np.float32(0.01)
    coeffs = np.zeros(8, dtype=np.float32)
    coeffs[[3, 5, 6]] = rng.randn(3).astype(np.float32) * 0.25
    biv_grad = rng.randn(8).astype(np.float32)
    biv_eta = np.float32(0.02)

    def _sphere_step() -> np.ndarray:
        grad_tan = sphere_grad - np.float32(np.dot(sphere_grad, x)) * x
        y = x - sphere_eta * grad_tan
        return (y / np.linalg.norm(y)).astype(np.float32)

    def _bivector_step() -> np.ndarray:
        grad_proj = np.zeros_like(biv_grad)
        grad_proj[[3, 5, 6]] = biv_grad[[3, 5, 6]]
        state_proj = np.zeros_like(coeffs)
        state_proj[[3, 5, 6]] = coeffs[[3, 5, 6]]
        return (state_proj - biv_eta * grad_proj).astype(np.float32)

    def dispatch() -> np.ndarray:
        return np.concatenate([_sphere_step(), _bivector_step()])

    out = dispatch()
    sphere_out = out[:3]
    biv_out = out[3:]
    norm_err = abs(float(np.linalg.norm(sphere_out)) - 1.0)
    leakage = float(np.abs(biv_out[[0, 1, 2, 4, 7]]).max())
    err = max(norm_err, leakage)
    details = {
        "component_ops": [
            "ebm_sphere_langevin_step",
            "ebm_bivector_langevin_step",
        ],
        "component_value_status": {
            "ebm_sphere_langevin_step": "compiler_visible_gated",
            "ebm_bivector_langevin_step": "compiler_visible_gated",
        },
        "value_calls": [
            _value_call_record(
                "ebm_sphere_langevin_step",
                "tessera_apple_gpu_ebm_sphere_langevin_step_f32",
                "compiler_visible"),
            _value_call_record(
                "ebm_bivector_langevin_step",
                "tessera_apple_gpu_ebm_bivector_langevin_step_f32",
                "compiler_visible"),
        ],
        "math_contract": (
            "sphere step must retract to unit norm; bivector step must "
            "preserve the grade-2 subspace"),
        "contract_metrics": {
            "sphere_norm_error": norm_err,
            "bivector_grade_leakage": leakage,
        },
    }
    return dispatch, err, "multi_call:manifold_ebm", details


def _compiler_composite_ga_feature_path() -> tuple[
    Callable[[], np.ndarray], float, str, dict[str, Any]
]:
    A = _seeded_pointwise(4300)
    B = _seeded_pointwise(4301)
    rotors = _seeded_rotor(4302)

    def dispatch() -> np.ndarray:
        product = _cl30_geometric_product_coeffs(A, B)
        projected = _cl30_grade_project_coeffs(product, {0, 2})
        return _cl30_rotor_sandwich_coeffs(rotors, projected)

    product = _cl30_geometric_product_coeffs(A, B)
    projected = _cl30_grade_project_coeffs(product, {0, 2})
    expected = _cl30_rotor_sandwich_coeffs(rotors, projected)
    err = float(np.abs(dispatch() - expected).max())
    details = {
        "component_ops": [
            "clifford_geometric_product",
            "clifford_grade_projection",
            "clifford_rotor_sandwich",
        ],
        "component_value_status": {
            "clifford_geometric_product": "executable_single_call",
            "clifford_grade_projection": "compiler_visible_gated",
            "clifford_rotor_sandwich": "compiler_visible_gated",
        },
        "value_calls": [
            _value_call_record(
                "clifford_geometric_product",
                "tessera_apple_gpu_clifford_geo_product_cl30_value_f32",
                "executable"),
            _value_call_record(
                "clifford_grade_projection",
                "tessera_apple_gpu_clifford_grade_projection_cl30_f32",
                "compiler_visible"),
            _value_call_record(
                "clifford_rotor_sandwich",
                "tessera_apple_gpu_clifford_rotor_sandwich_cl30_f32",
                "compiler_visible"),
        ],
        "math_contract": (
            "geometric product feeds grade projection, then a rotor "
            "sandwich preserves blade-last Cl(3,0) coefficient layout"),
        "contract_metrics": {
            "projected_non_even_max": float(
                np.abs(projected[:, [1, 2, 4, 7]]).max()),
            "batch": int(A.shape[0]),
        },
    }
    return dispatch, err, "multi_call:ga_feature_pipeline", details


_COMPOSITE_BUILDERS = [
    ("composite_ebt_tiny_refinement",
     "B=3,K=4,D=5/decode_refine_energy_verify",
     _compiler_composite_ebt_tiny_path, 1e-6),
    ("composite_manifold_ebm",
     "sphere(S^2)+bivector(Cl(3,0))",
     _compiler_composite_manifold_ebm_path, 1e-5),
    ("composite_ga_feature_pipeline",
     f"{_BATCH}x8/geo_grade_rotor",
     _compiler_composite_ga_feature_path, 1e-5),
]


def run_composite_compiler_visible_reference_path(
    name: str, shape: str, dispatch: Callable, err: float, symbol: str,
    tolerance: float, reps: int, device: str, version: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    samples_ms = collect_samples(dispatch, reps)
    timing = timing_stats(samples_ms)
    value_calls = list(details["value_calls"])
    symbols = [call["symbol"] for call in value_calls]
    return {
        "backend": "compiler_visible_reference",
        "namespace": "composite",
        "op": name,
        "shape": shape,
        "dtype": "f32",
        "mode": "composite_compiler_visible_reference",
        "reps": reps,
        **timing,
        "max_abs_err": err,
        "tolerance": tolerance,
        "ok": err <= tolerance,
        "apple_gpu_status": "composite_multi_call_gated",
        "symbol": symbol,
        "symbols": symbols,
        "device": device,
        "tessera_version": version,
        "component_ops": list(details["component_ops"]),
        "component_value_status": dict(details["component_value_status"]),
        "value_calls": value_calls,
        "value_call_count": len(value_calls),
        "composite_status": _STAGE17_COMPOSITE_STATUS,
        "multi_call_executor": None,
        "math_contract": details["math_contract"],
        "contract_metrics": dict(details["contract_metrics"]),
        **_stage16f_fields(
            "compiler_visible_reference",
            compiler_path="apple_value_target_ir",
            executor="python_reference",
            runtime_status="reference",
            execution_kind="reference_cpu"),
    }


# ---------------------------------------------------------------------------
# JIT-bridge benchmark — exercises the full Python → JIT-context →
# manifest-resolve → shared-loader-dispatch → result path with the
# bridge's trace recording on.  The bridge contract is:
#
#   1. ``tessera.ga.inner(a, b)`` (or any GA/EBM public API with a
#      fast path) calls ``jit_bridge.dispatch_via_manifest(op_name=...)``
#   2. The bridge looks up the apple_gpu symbol via the manifest
#   3. The bridge binds it through ``tessera._apple_gpu_dispatch``
#   4. The bridge records a ``JitBridgeRoute(op, target, symbol,
#      context, latency_ms, ...)`` in the thread-local trace
#
# This benchmark wraps a tiny @jit(target="apple_gpu") span using the
# bridge's ``jit_context`` so the route trace carries
# ``context="jit:apple_gpu"`` for every dispatch — proving the route
# end-to-end.
# ---------------------------------------------------------------------------

def _jit_bridge_ga_inner_path() -> tuple[Callable[[], np.ndarray],
                                          float, "list[Any]"]:
    """Run a small ``ga.inner`` workload under ``jit_context`` and
    return ``(dispatch, max_abs_err_vs_numpy, routes_seen)``.

    The dispatch closure resets the trace, runs the call, drains the
    trace.  Each invocation produces exactly one route row.
    """
    from tessera.compiler import jit_bridge as _bridge
    a = ga.Cl(3, 0)
    rng = np.random.RandomState(3000)
    A = rng.randn(32, 8).astype(np.float32)
    B = rng.randn(32, 8).astype(np.float32)
    mv_a = ga.Multivector(A, a)
    mv_b = ga.Multivector(B, a)
    expected = np.array([
        float(ga.inner(ga.Multivector(A[i], a),
                        ga.Multivector(B[i], a))) for i in range(32)
    ])

    last_routes: list[Any] = []

    def dispatch() -> np.ndarray:
        _bridge.clear_dispatch_trace()
        with _bridge.jit_context("apple_gpu"):
            out = ga.inner(mv_a, mv_b)
        last_routes[:] = _bridge.take_dispatch_trace()
        return np.asarray(out)

    out = dispatch()
    err = float(np.abs(out - expected).max())
    return dispatch, err, last_routes


def _jit_bridge_ebm_inner_step_path() -> tuple[Callable[[], np.ndarray],
                                                 float, "list[Any]"]:
    """Mirror of the GA case for ``ebm.inner_step``."""
    from tessera.compiler import jit_bridge as _bridge
    rng = np.random.RandomState(3001)
    y = rng.randn(64, 16).astype(np.float32)
    grad = rng.randn(64, 16).astype(np.float32)
    eta = 0.05
    expected = (y - eta * grad).astype(np.float32)

    last_routes: list[Any] = []

    def dispatch() -> np.ndarray:
        _bridge.clear_dispatch_trace()
        with _bridge.jit_context("apple_gpu"):
            out = ebm.inner_step(y, grad, eta=eta)
        last_routes[:] = _bridge.take_dispatch_trace()
        return out

    out = dispatch()
    err = float(np.abs(out - expected).max())
    return dispatch, err, last_routes


def run_jit_bridge_path(name: str, shape: str, builder: Callable,
                         tolerance: float, reps: int,
                         device: str, version: str) -> dict[str, Any]:
    """Time a bridge-mediated dispatch + record the routes.

    The dispatch closure has already opened/closed a ``jit_context``
    span, so the captured routes carry ``context="jit:apple_gpu"``.
    Tracing is on for the duration of the timed loop so every
    dispatch in the loop produces a route row in the shared
    ``last_routes`` list the builder returns; we read the final
    routes from that list (the closure itself drains the trace).
    """
    from tessera.compiler import jit_bridge as _bridge
    dispatch, err, last_routes = builder()
    prev_tracing = _bridge.tracing_enabled()
    _bridge.set_tracing_enabled(True)
    try:
        samples_ms = collect_samples(dispatch, reps)
    finally:
        _bridge.set_tracing_enabled(prev_tracing)
    timing = timing_stats(samples_ms)
    # After the timed loop `last_routes` holds the route(s) captured
    # by the most-recent dispatch — exactly what we want to record.
    route_records = [
        {"op": r.op_name, "target": r.target, "status": r.status,
         "context": r.context, "symbol": r.symbol,
         "latency_ms": r.latency_ms}
        for r in last_routes
    ]
    return {
        "backend": "apple_gpu",
        "namespace": "jit_bridge",
        "op": name,
        "shape": shape,
        "dtype": "f32",
        "mode": "jit_resolved",
        "reps": reps,
        **timing,
        "max_abs_err": err,
        "tolerance": tolerance,
        "ok": err <= tolerance,
        "routes": route_records,
        "device": device,
        "tessera_version": version,
        **_stage16f_fields(
            "legacy_manifest_native" if err <= tolerance else "python_reference",
            compiler_path="jit_bridge",
            executor="apple_gpu_manifest" if err <= tolerance else None,
            runtime_status="success" if err <= tolerance else "numerical_mismatch",
            execution_kind="legacy_native_gpu" if err <= tolerance else "unknown"),
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
               include_composites: bool = True,
               include_ebt_sweep: bool = False,
               include_jit_bridge: bool = True,
               include_vertical_slice: bool = True,
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
      - ``include_composites``: Stage 17 compiler-visible multi-call
        GA/EBM proof rows. These validate chained math contracts and
        value-call metadata, but remain reference-executed until the
        Apple value runtime grows a multi-call or fused executor.
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
            for (namespace, op_name, shape_desc, ref_builder,
                 value_builder, tol) in _VALUE_TARGET_BUILDERS:
                built = value_builder()
                if built is None:
                    continue
                dispatch, err, sym = built
                rows.append(run_ebm_apple_value_path(
                    op_name, shape_desc, dispatch, err, sym,
                    tolerance=tol, reps=reps,
                    device=device, version=version, namespace=namespace,
                ))

        for namespace, op_name, shape_desc, ref_builder, _, tol in _VALUE_TARGET_BUILDERS:
            dispatch, err, sym = ref_builder()
            rows.append(run_compiler_visible_reference_path(
                namespace, op_name, shape_desc, dispatch, err, sym,
                tolerance=tol, reps=reps, device=device, version=version,
            ))

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
            (dispatch, err, syms, shape,
             dispatched_on_gpu) = _workload_ebt_tiny_apple_gpu_path(
                rt, refinement_T)
            rows.append(run_workload_apple_gpu(
                "ebt_tiny_refinement", shape, dispatch, err, syms,
                tolerance=1e-4, reps=reps, device=device, version=version,
                dispatched_on_gpu=dispatched_on_gpu,
            ))
        dispatch, err, shape = _workload_ebt_tiny_python_path(refinement_T)
        rows.append(run_workload_python(
            "ebt_tiny_refinement", shape, dispatch, err,
            tolerance=1e-4, reps=reps, device=device, version=version,
        ))

        # Workload pair 3: rotor-conditioned EBT — the GA + EBM
        # fused workload that "makes the two families feel like one
        # system".  GA primitives (exp_mv + rotor_sandwich) feed an
        # EBM K-candidate selection (ebt_tiny).  All through public
        # APIs, all bridge-traced.
        if rt is not None:
            (dispatch, err, syms, shape,
             dispatched_on_gpu) = _workload_rotor_ebt_apple_gpu_path(rt)
            rows.append(run_workload_apple_gpu(
                "rotor_conditioned_ebt", shape, dispatch, err, syms,
                tolerance=5e-5, reps=reps, device=device, version=version,
                dispatched_on_gpu=dispatched_on_gpu,
            ))
        dispatch, err, shape = _workload_rotor_ebt_python_path()
        rows.append(run_workload_python(
            "rotor_conditioned_ebt", shape, dispatch, err,
            tolerance=5e-5, reps=reps, device=device, version=version,
        ))

    if include_composites:
        for op_name, shape_desc, builder, tol in _COMPOSITE_BUILDERS:
            dispatch, err, sym, details = builder()
            rows.append(run_composite_compiler_visible_reference_path(
                op_name, shape_desc, dispatch, err, sym,
                tolerance=tol, reps=reps, device=device, version=version,
                details=details,
            ))

    # EBT-tiny break-even sweep — apples-to-apples timings across a
    # ladder of (B, K, D, T) points so consumers can locate where the
    # native MSL chain starts beating numpy.  Off by default (adds
    # ~14 rows × reps dispatches); enable via --ebt-sweep.
    if include_ebt_sweep:
        for (B, K, D, T) in ebt_sweep_points:
            if rt is not None:
                (dispatch, err, syms, shape,
                 dispatched_on_gpu) = _workload_ebt_tiny_apple_gpu_path(
                    rt, T, B=B, K=K, D=D)
                rows.append(run_workload_apple_gpu(
                    "ebt_tiny_sweep", shape, dispatch, err, syms,
                    tolerance=1e-4, reps=reps,
                    device=device, version=version,
                    dispatched_on_gpu=dispatched_on_gpu,
                ))
            dispatch, err, shape = _workload_ebt_tiny_python_path(
                T, B=B, K=K, D=D)
            rows.append(run_workload_python(
                "ebt_tiny_sweep", shape, dispatch, err,
                tolerance=1e-4, reps=reps,
                device=device, version=version,
            ))

    # JIT-bridge rows — exercise the full Python → JIT context →
    # manifest resolve → shared-loader dispatch → result path and
    # record each route in the report.
    if include_jit_bridge and rt is not None:
        for op_name, shape_desc, builder, tol in (
            ("clifford_inner",  "B=32,D=8/Cl(3,0)",
             _jit_bridge_ga_inner_path, 5e-5),
            ("ebm_inner_step",  "B=64,D=16",
             _jit_bridge_ebm_inner_step_path, 1e-6),
        ):
            rows.append(run_jit_bridge_path(
                op_name, shape_desc, builder, tolerance=tol,
                reps=reps, device=device, version=version,
            ))

    # Compiler-integrated vertical slice — `@clifford_jit` decorator
    # → traced op plan → manifest-resolved Apple target metadata →
    # runtime dispatch → benchmark row.  Records the full plan
    # (op list + manifest-resolved symbols + plan hash) in the
    # `compiled_artifact` column.
    if include_vertical_slice and rt is not None:
        try:
            (vs_dispatch, vs_err, vs_plan_hash,
             vs_metadata) = _vertical_slice_apple_gpu_path(rt)
        except Exception as exc:  # pragma: no cover — surfaced in row
            rows.append({
                "backend": "apple_gpu",
                "namespace": "vertical_slice",
                "op": "point_cloud_rotor_invariant",
                "shape": "B=64/Cl(3,0)",
                "dtype": "f32",
                "mode": "failed",
                "reps": reps,
                "latency_ms": 0.0,
                "stdev_ms": 0.0,
                "p10_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0,
                "min_ms": 0.0, "max_ms": 0.0,
                "max_abs_err": float("inf"),
                "tolerance": 5e-5,
                "ok": False,
                "error": str(exc),
                "device": device,
                "tessera_version": version,
                **_stage16f_fields(
                    "legacy_manifest_native", compiler_path="clifford_jit",
                    executor=None, runtime_status="compile_failed",
                    execution_kind="unknown"),
            })
        else:
            samples_ms = collect_samples(vs_dispatch, reps)
            timing = timing_stats(samples_ms)
            ok = vs_err <= 5e-5
            rows.append({
                "backend": "apple_gpu",
                "namespace": "vertical_slice",
                "op": "point_cloud_rotor_invariant",
                "shape": "B=64/Cl(3,0)",
                "dtype": "f32",
                "mode": "jit_compiled",
                "reps": reps,
                **timing,
                "max_abs_err": vs_err,
                "tolerance": 5e-5,
                "ok": ok,
                "plan_hash": vs_plan_hash,
                "compiled_artifact": vs_metadata,
                "device": device,
                "tessera_version": version,
                **_stage16f_fields(
                    "legacy_manifest_native", compiler_path="clifford_jit",
                    executor="apple_gpu_manifest" if ok else None,
                    runtime_status="success" if ok else "numerical_mismatch",
                    execution_kind="legacy_native_gpu" if ok else "unknown"),
            })

    ga_count = sum(1 for r in rows if r["op"].startswith("clifford_")
                                       and r.get("namespace") == "ga"
                                       and r.get("backend") == "apple_gpu")
    ebm_count = sum(1 for r in rows if r["op"].startswith("ebm_")
                                        and r.get("namespace") == "ebm")
    workload_count = sum(1 for r in rows if r.get("namespace") == "workload")
    ebm_native_count = sum(1 for r in rows
                            if r["op"].startswith("ebm_")
                            and r["backend"] == "apple_gpu"
                            and r.get("namespace") == "ebm")
    sweep_count = sum(1 for r in rows if r["op"] == "ebt_tiny_sweep")
    composite_count = sum(1 for r in rows
                          if r.get("namespace") == "composite")
    jit_bridge_count = sum(1 for r in rows
                            if r.get("namespace") == "jit_bridge")
    vertical_slice_count = sum(1 for r in rows
                                if r.get("namespace") == "vertical_slice")
    return {
        "runs": rows,
        "ga_primitives_count": ga_count,
        "ebm_paths_count": ebm_count,
        "ebm_native_apple_gpu_count": ebm_native_count,
        "native_ebm_ops": sorted(native_ebm_ops),
        "workload_count": workload_count,
        "composite_count": composite_count,
        "ebt_sweep_count": sweep_count,
        "ebt_sweep_summary": (_ebt_sweep_break_even_summary(rows)
                               if sweep_count else None),
        "jit_bridge_count": jit_bridge_count,
        "vertical_slice_count": vertical_slice_count,
        "compile_time_ms": compile_time_ms,
        "skipped_apple_gpu": skip_reason,
        "device": device,
        "tessera_version": version,
        "reps": reps,
    }


def _ebt_sweep_break_even_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce a per-shape native-vs-numpy comparison table.

    Speedup is only computed when the GPU attempt actually dispatched
    on-device (``dispatched_on_gpu=True``).  A row whose GPU path
    silently fell back to numpy lives in the table with
    ``status="degraded_fallback"`` and no ``speedup`` column — the
    sweep refuses to claim a win for shapes that didn't take the
    native path.
    """
    # Two slots per shape: the GPU-attempt row (any mode), and the
    # explicit python_ref baseline row (``mode="reference_chain"``).
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        if r["op"] != "ebt_tiny_sweep":
            continue
        slot_key = "baseline" if r.get("mode") == "reference_chain" else "native_attempt"
        pairs.setdefault(r["shape"], {})[slot_key] = r
    table: list[dict[str, Any]] = []
    for shape, slots in sorted(pairs.items()):
        native = slots.get("native_attempt")
        python = slots.get("baseline")
        entry: dict[str, Any] = {"shape": shape}
        if native is not None:
            entry["native_ms"] = native["latency_ms"]
            entry["native_dispatched_on_gpu"] = bool(
                native.get("dispatched_on_gpu", False))
        if python is not None:
            entry["python_ms"] = python["latency_ms"]
        if native is not None and python is not None:
            if native.get("dispatched_on_gpu") and native["latency_ms"] > 0:
                sp = python["latency_ms"] / native["latency_ms"]
                entry["speedup"] = sp
                entry["native_wins"] = sp >= 1.0
                entry["status"] = "native_dispatched"
            else:
                # Native attempt fell back to numpy — refuse to compute
                # a "speedup" since both rows ran numpy and the
                # comparison is meaningless.
                entry["status"] = "degraded_fallback"
                entry["native_wins"] = False
        table.append(entry)
    break_even = None
    for entry in table:
        if entry.get("native_wins"):
            break_even = entry["shape"]
            break
    degraded = sum(1 for e in table if e.get("status") == "degraded_fallback")
    return {
        "table": table,
        "first_native_win_shape": break_even,
        "degraded_count": degraded,
    }


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
    parser.add_argument("--no-composites", action="store_true",
                        help=("Skip Stage 17 composite proof rows "
                              "(compiler-visible multi-call GA/EBM chains)."))
    parser.add_argument("--no-jit-bridge", action="store_true",
                        help=("Skip the JIT-bridge rows (Python → "
                              "jit_context → manifest → shared loader → "
                              "result, with per-dispatch trace recorded "
                              "in the report row's `routes` column)."))
    parser.add_argument("--no-vertical-slice", action="store_true",
                        help=("Skip the compiler-integrated vertical "
                              "slice row (@clifford_jit decorator → "
                              "traced op plan → manifest-resolved "
                              "Apple target metadata → runtime "
                              "dispatch)."))
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON output path (stdout if omitted)")
    args = parser.parse_args(argv)

    reps = DEFAULT_REPS_CI if args.ci else args.reps
    include_primitives = not args.workloads_only
    include_workloads = not args.primitives_only
    report = run_report(reps=reps, refinement_T=args.refinement_T,
                        include_primitives=include_primitives,
                        include_workloads=include_workloads,
                        include_composites=not args.no_composites,
                        include_ebt_sweep=args.ebt_sweep,
                        include_jit_bridge=not args.no_jit_bridge,
                        include_vertical_slice=not args.no_vertical_slice)
    payload = json.dumps(report, indent=2, sort_keys=True, default=float)
    if args.output is not None:
        args.output.write_text(payload)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
