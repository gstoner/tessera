"""Production-lane CPU JIT boundary (ctypes → ``libtessera_jit``).

EXPERIMENTAL production-lane plumbing — see ``docs/spec/RUNTIME_ABI_SPEC.md`` §12
and ``docs/spec/PRODUCTION_COMPILER_PLAN.md``. This is **not** "runtime v2".

The whole point of this lane: it executes the MLIR/LLVM compiled function and has
**no numpy fallback by construction**. Any compile/invoke failure raises
:class:`TesseraJitError`. Numerical equality alone does not prove the lane ran;
oracle tests also assert :func:`invocation_count` advanced.

Phase 0 was hardcoded to rank-2 f32 ``tessera_jit_add``. Phase 1 is generic:

* :func:`compile_module` accepts any MLIR text.
* :func:`invoke` dispatches any compiled function by name via the C ABI's
  :c:`tessera_jit_invoke` (which forwards to ``mlir::ExecutionEngine::invokePacked``).
* Memref descriptors are built dynamically per (rank, dtype) — Phase 1 covers
  rank-N f32; bf16 / other dtypes land later under the same machinery.

High-level helpers (:func:`jit_add`, :func:`jit_sub`, :func:`jit_mul`,
:func:`jit_matmul`) are thin wrappers over :func:`invoke`.
"""

from __future__ import annotations

import ctypes
import glob
import os
from typing import Any, Sequence

import numpy as np

LAST_EXECUTION_BACKEND = "mlir_llvm_jit"


class TesseraJitError(RuntimeError):
    """Raised when the production JIT lane cannot compile or invoke.

    Notably this is raised *instead of* silently falling back to numpy — the
    lane either executes compiled code or fails loudly.
    """


_LIB: ctypes.CDLL | None = None


def _repo_root() -> str:
    # python/tessera/_jit_boundary.py -> repo root is three levels up.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _find_dylib() -> str | None:
    if (env := os.environ.get("TESSERA_JIT_LIB")) and os.path.exists(env):
        return env
    repo = _repo_root()
    pats = [
        os.path.join(repo, "build", "tools", "tessera-jit", "libtessera_jit.*"),
        os.path.join(repo, "build*", "tools", "tessera-jit", "libtessera_jit.*"),
    ]
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def _load() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    path = _find_dylib()
    if path is None:
        raise TesseraJitError(
            "libtessera_jit not built; run "
            "`ninja -C build tessera_jit` (or set TESSERA_JIT_LIB)"
        )
    lib = ctypes.CDLL(path)
    lib.tessera_jit_compile.restype = ctypes.c_void_p
    lib.tessera_jit_compile.argtypes = [ctypes.c_char_p]
    lib.tessera_jit_invoke.restype = ctypes.c_int
    lib.tessera_jit_invoke.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.c_char_p,  # symbol name (e.g. "tessera_jit_add")
        ctypes.POINTER(ctypes.c_void_p),  # packed_args (void**)
        ctypes.c_int,  # nargs
    ]
    lib.tessera_jit_destroy.restype = None
    lib.tessera_jit_destroy.argtypes = [ctypes.c_void_p]
    lib.tessera_jit_last_error.restype = ctypes.c_char_p
    lib.tessera_jit_last_error.argtypes = []
    lib.tessera_jit_invocation_count.restype = ctypes.c_int64
    lib.tessera_jit_invocation_count.argtypes = []
    _LIB = lib
    return lib


def is_available() -> bool:
    """True when the production JIT dylib can be loaded."""
    try:
        _load()
        return True
    except TesseraJitError:
        return False


def invocation_count() -> int:
    """Successful JIT invocations since process start (proof-of-execution)."""
    return int(_load().tessera_jit_invocation_count())


# ── Descriptor packing ─────────────────────────────────────────────────────

# Tessera dtype name → (numpy/ml_dtypes dtype, ctypes elem type, MLIR spelling).
# RUNTIME_ABI_SPEC.md §12.5: bf16 uses ml_dtypes.bfloat16 on the Python side and
# RAW 16-bit storage at the boundary (ctypes.c_uint16) — the bits are passed
# through unreinterpreted; the MLIR `bf16` element type gives them meaning.
_DTYPE_TABLE: dict[str, tuple[Any, Any, str]] = {
    "f32": (np.float32, ctypes.c_float, "f32"),
}

try:  # bf16 is optional: present iff ml_dtypes is installed (soft dependency).
    import ml_dtypes as _ml_dtypes

    _BF16 = _ml_dtypes.bfloat16
    _DTYPE_TABLE["bf16"] = (_BF16, ctypes.c_uint16, "bf16")
except ImportError:  # pragma: no cover - exercised only on minimal envs
    _BF16 = None


def _dtype_entry(arr: np.ndarray):
    for tag, (np_dt, ct_dt, mlir) in _DTYPE_TABLE.items():
        if arr.dtype == np_dt:
            return tag, ct_dt, mlir
    raise TesseraJitError(
        f"Phase 1 boundary supports {sorted(_DTYPE_TABLE)} only (got {arr.dtype})"
    )


def _resolve_elem(arrays: Sequence[np.ndarray]) -> tuple[str, Any]:
    """Validate that all `arrays` share one supported dtype; return its MLIR
    element spelling + the numpy/ml_dtypes dtype (for output allocation).

    Mixed dtypes are rejected here rather than silently promoted — the boundary
    contract is explicit (RUNTIME_ABI_SPEC §12.5: convert on mismatch is the
    *caller's* responsibility).
    """
    dts = {a.dtype for a in arrays}
    if len(dts) != 1:
        raise TesseraJitError(f"operands must share one dtype; got {sorted(map(str, dts))}")
    dt = next(iter(dts))
    for _tag, (np_dt, _ct, mlir) in _DTYPE_TABLE.items():
        if dt == np_dt:
            return mlir, np_dt
    raise TesseraJitError(
        f"unsupported dtype {dt}; supported: {sorted(_DTYPE_TABLE)}"
    )


def _descriptor_struct(rank: int, elem_ct: Any) -> type:
    """Build the ctypes mirror of MLIR's standard memref descriptor for (rank, elem).

    Cached per (rank, ctype) because every new ctypes Structure type is a fresh
    class (and we'd otherwise leak class objects on every invoke).
    """
    key = (rank, elem_ct)
    cache = _descriptor_struct.__dict__.setdefault("_cache", {})
    if key in cache:
        return cache[key]
    elem_ptr = ctypes.POINTER(elem_ct)
    fields = [
        ("allocated", elem_ptr),
        ("aligned", elem_ptr),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * rank),
        ("strides", ctypes.c_int64 * rank),
    ]
    name = f"MemRef{rank}d_{elem_ct.__name__}"
    # No _pack_: every field is pointer/int64-sized, so natural arm64/x86_64
    # alignment already matches MLIR's emitted descriptor layout. _pack_ would
    # trigger a Python 3.19 layout-default DeprecationWarning for zero gain.
    cls = type(name, (ctypes.Structure,), {"_fields_": fields})
    cache[key] = cls
    return cls


def _row_major_strides(shape: Sequence[int]) -> list[int]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _make_descriptor(arr: np.ndarray, elem_ct: Any):
    """Build a memref descriptor for `arr`. RUNTIME_ABI_SPEC §12.2 / §12.4:
    identity layout, C-contiguous, offset 0.
    """
    if not arr.flags.c_contiguous:
        raise TesseraJitError(
            "boundary buffers must be C-contiguous (§12.4); "
            "caller must materialize a contiguous copy"
        )
    rank = arr.ndim
    Desc = _descriptor_struct(rank, elem_ct)
    ptr = arr.ctypes.data_as(ctypes.POINTER(elem_ct))
    d = Desc()
    d.allocated = ptr
    d.aligned = ptr
    d.offset = 0
    for i in range(rank):
        d.sizes[i] = int(arr.shape[i])
    for i, s in enumerate(_row_major_strides(arr.shape)):
        d.strides[i] = int(s)
    return d


def _build_packed_args(descriptors):
    """Build the ``void**`` array ``tessera_jit_invoke`` forwards to the c-iface.

    ABI (RUNTIME_ABI_SPEC §12.1): ``_mlir_ciface_<name>`` takes
    ``void(Desc*, Desc*, ..., Desc*)`` — descriptor pointers, in declaration
    order. So ``packed[i]`` is the descriptor pointer for the i-th argument —
    one level of indirection.

    Returns ``(packed, keepalive)`` where ``keepalive`` must outlive the call.
    """
    n = len(descriptors)
    packed = (ctypes.c_void_p * n)(
        *[ctypes.addressof(d) for d in descriptors]
    )
    keepalive: list[Any] = [descriptors, packed]
    return packed, keepalive


# ── Compile + invoke ───────────────────────────────────────────────────────


def compile_module(mlir_text: str) -> int:
    """Compile an arbitrary MLIR module through the production pipeline.

    Returns an opaque integer handle (truthy on success). Raises
    :class:`TesseraJitError` on parse/lowering/JIT failure.
    """
    lib = _load()
    handle = lib.tessera_jit_compile(mlir_text.encode("utf-8"))
    if not handle:
        raise TesseraJitError(lib.tessera_jit_last_error().decode("utf-8", "replace"))
    return handle


def destroy(handle: int) -> None:
    if handle:
        _load().tessera_jit_destroy(handle)


def invoke(
    handle: int, symbol: str, arrays: Sequence[np.ndarray], out: np.ndarray
) -> None:
    """Invoke a compiled function via the C ABI.

    ``arrays`` is the list of *input* arrays (c-iface order, all f32 in Phase 1).
    ``out`` is the caller-allocated destination (DPS, §12.3). All buffers must
    be C-contiguous identity-layout f32. Raises :class:`TesseraJitError` on any
    failure — never silently falls back.
    """
    lib = _load()
    _, elem_ct, _ = _dtype_entry(out)  # out determines dtype family
    descs = [_make_descriptor(a, elem_ct) for a in (*arrays, out)]
    packed, _keepalive = _build_packed_args(descs)
    rc = lib.tessera_jit_invoke(
        handle, symbol.encode("utf-8"), packed, len(descs)
    )
    if rc != 0:
        raise TesseraJitError(lib.tessera_jit_last_error().decode("utf-8", "replace"))


# ── High-level helpers ─────────────────────────────────────────────────────


def _f32_envelope_check(arrays: Sequence[np.ndarray]) -> None:
    for a in arrays:
        if a.dtype != np.float32:
            raise TesseraJitError(
                f"Phase 1 jit boundary is f32-only (got {a.dtype}); "
                "boundary must REJECT non-f32 rather than silently coerce"
            )


def _jit_binary(op_name: str, sym: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    b = np.asarray(b)
    elem, _npdt = _resolve_elem([a, b])  # f32 or bf16; rejects mixed/unsupported
    if a.shape != b.shape:
        raise TesseraJitError(
            f"elementwise requires equal shapes (got {a.shape}, {b.shape})"
        )
    if a.ndim < 1:
        raise TesseraJitError("Phase 1 boundary requires rank>=1")
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    t = "tensor<" + "x".join(str(s) for s in a.shape) + f"x{elem}>"
    mlir = (
        f"func.func @{sym}(%a: {t}, %b: {t}) -> {t} {{\n"
        f"  %0 = {op_name} %a, %b : ({t}, {t}) -> {t}\n"
        f"  return %0 : {t}\n"
        f"}}\n"
    )
    handle = compile_module(mlir)
    try:
        out = np.empty_like(a)
        invoke(handle, sym, [a, b], out)
        return out
    finally:
        destroy(handle)


def jit_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Production-lane elementwise add (rank>=1, f32 or bf16). No fallback."""
    return _jit_binary("tessera.add", "tessera_jit_add", a, b)


def jit_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Production-lane elementwise sub (rank>=1, f32 or bf16). No fallback."""
    return _jit_binary("tessera.sub", "tessera_jit_sub", a, b)


def jit_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Production-lane elementwise mul (rank>=1, f32 or bf16). No fallback."""
    return _jit_binary("tessera.mul", "tessera_jit_mul", a, b)


def jit_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Production-lane elementwise div (rank>=1 f32). No fallback."""
    return _jit_binary("tessera.div", "tessera_jit_div", a, b)


def jit_softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Production-lane numerically-stable softmax over `axis` (f32).

    Lowers to the stable decomposition ``exp(x - max) / sum(exp(x - max))``
    (the JIT lowering does the max-subtract for stability). Same shape as input,
    rank >= 1. No fallback.
    """
    a = np.asarray(a)
    _f32_envelope_check([a])
    if a.ndim < 1:
        raise TesseraJitError("Phase 1 jit_softmax requires rank>=1")
    ax = axis + a.ndim if axis < 0 else axis
    if ax < 0 or ax >= a.ndim:
        raise TesseraJitError(f"axis {axis} out of range for rank {a.ndim}")
    a = np.ascontiguousarray(a)
    t = "tensor<" + "x".join(str(s) for s in a.shape) + "xf32>"
    sym = "tessera_jit_softmax"
    mlir = (
        f"func.func @{sym}(%x: {t}) -> {t} {{\n"
        f"  %0 = tessera.softmax %x {{axis = {ax} : i64}} : ({t}) -> {t}\n"
        f"  return %0 : {t}\n"
        f"}}\n"
    )
    handle = compile_module(mlir)
    try:
        out = np.empty_like(a)
        invoke(handle, sym, [a], out)
        return out
    finally:
        destroy(handle)


def jit_matmul(
    a: np.ndarray,
    b: np.ndarray,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> np.ndarray:
    """Production-lane rank-2 matmul (f32, or bf16 storage with f32 accumulate).

    `transpose_a`/`transpose_b` select the attention-shaped variants: with
    `transpose_b=True`, `b` is stored (N, K) and the op computes `a @ bᵀ`
    (the `Q @ Kᵀ` pattern). For bf16 the result is bf16 (ABI §12.5: f32
    accumulate, truncate-on-store, in the JIT lowering). No fallback.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    elem, npdt = _resolve_elem([a, b])
    if a.ndim != 2 or b.ndim != 2:
        raise TesseraJitError(
            f"Phase 1 jit_matmul is rank-2 only (got {a.shape}, {b.shape})"
        )
    # Logical (M,K) for a and (K,N) for b after honoring the transpose flags.
    M, Ka = (a.shape[1], a.shape[0]) if transpose_a else (a.shape[0], a.shape[1])
    Kb, N = (b.shape[1], b.shape[0]) if transpose_b else (b.shape[0], b.shape[1])
    if Ka != Kb:
        raise TesseraJitError(
            f"matmul contracting mismatch: a={a.shape} (tA={transpose_a}) "
            f"b={b.shape} (tB={transpose_b})"
        )
    M, N = int(M), int(N)
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    ta = "tensor<" + "x".join(map(str, a.shape)) + f"x{elem}>"
    tb = "tensor<" + "x".join(map(str, b.shape)) + f"x{elem}>"
    tc = f"tensor<{M}x{N}x{elem}>"
    attrs = []
    if transpose_a:
        attrs.append("transposeA = true")
    if transpose_b:
        attrs.append("transposeB = true")
    attr_str = (" {" + ", ".join(attrs) + "}") if attrs else ""
    mlir = (
        f"func.func @tessera_jit_matmul(%a: {ta}, %b: {tb}) -> {tc} {{\n"
        f"  %0 = tessera.matmul %a, %b{attr_str} : ({ta}, {tb}) -> {tc}\n"
        f"  return %0 : {tc}\n"
        f"}}\n"
    )
    handle = compile_module(mlir)
    try:
        out = np.empty((M, N), dtype=npdt)
        invoke(handle, "tessera_jit_matmul", [a, b], out)
        return out
    finally:
        destroy(handle)


def jit_transpose(a: np.ndarray) -> np.ndarray:
    """Production-lane rank-2 transpose (f32 or bf16). No fallback."""
    a = np.asarray(a)
    elem, npdt = _resolve_elem([a])
    if a.ndim != 2:
        raise TesseraJitError(f"Phase 1 jit_transpose is rank-2 only (got {a.shape})")
    a = np.ascontiguousarray(a)
    M, N = int(a.shape[0]), int(a.shape[1])
    ti = f"tensor<{M}x{N}x{elem}>"
    to = f"tensor<{N}x{M}x{elem}>"
    sym = "tessera_jit_transpose"
    mlir = (
        f"func.func @{sym}(%x: {ti}) -> {to} {{\n"
        f"  %0 = tessera.transpose %x : ({ti}) -> {to}\n"
        f"  return %0 : {to}\n"
        f"}}\n"
    )
    handle = compile_module(mlir)
    try:
        out = np.empty((N, M), dtype=npdt)
        invoke(handle, sym, [a], out)
        return out
    finally:
        destroy(handle)


# ── Reductions (Phase 1 Sprint 1.2: first op with result rank != input rank) ─

_REDUCE_KINDS = frozenset({"sum", "max", "min", "mean"})


def jit_reduce(a: np.ndarray, axis: int, kind: str) -> np.ndarray:
    """Production-lane single-axis reduction (f32).

    ``kind`` ∈ {sum, max, min, mean}. Reduces `a` over `axis`, returning an
    array of rank ``a.ndim - 1``. Input must be rank >= 2 (so the result is
    rank >= 1 — the boundary has no rank-0 descriptor in Phase 1). No fallback.
    """
    a = np.asarray(a)
    _f32_envelope_check([a])
    if kind not in _REDUCE_KINDS:
        raise TesseraJitError(f"reduce kind must be one of {sorted(_REDUCE_KINDS)}")
    if a.ndim < 2:
        raise TesseraJitError(
            f"Phase 1 jit_reduce requires rank>=2 (got rank {a.ndim}); "
            "rank-0 results have no boundary descriptor yet"
        )
    ax = axis + a.ndim if axis < 0 else axis
    if ax < 0 or ax >= a.ndim:
        raise TesseraJitError(f"axis {axis} out of range for rank {a.ndim}")
    a = np.ascontiguousarray(a)

    in_shape = tuple(int(s) for s in a.shape)
    out_shape = tuple(s for i, s in enumerate(in_shape) if i != ax)
    ti = "tensor<" + "x".join(str(s) for s in in_shape) + "xf32>"
    to = "tensor<" + "x".join(str(s) for s in out_shape) + "xf32>"
    sym = "tessera_jit_reduce"
    mlir = (
        f"func.func @{sym}(%a: {ti}) -> {to} {{\n"
        f'  %0 = tessera.reduce %a {{kind = "{kind}", axis = {ax} : i64}} '
        f": ({ti}) -> {to}\n"
        f"  return %0 : {to}\n"
        f"}}\n"
    )
    handle = compile_module(mlir)
    try:
        out = np.empty(out_shape, dtype=np.float32)
        invoke(handle, sym, [a], out)
        return out
    finally:
        destroy(handle)


def jit_sum(a: np.ndarray, axis: int) -> np.ndarray:
    """Production-lane sum over `axis` (f32). No fallback."""
    return jit_reduce(a, axis, "sum")


def jit_amax(a: np.ndarray, axis: int) -> np.ndarray:
    """Production-lane max over `axis` (f32). No fallback."""
    return jit_reduce(a, axis, "max")


def jit_amin(a: np.ndarray, axis: int) -> np.ndarray:
    """Production-lane min over `axis` (f32). No fallback."""
    return jit_reduce(a, axis, "min")


def jit_mean(a: np.ndarray, axis: int) -> np.ndarray:
    """Production-lane mean over `axis` (f32). No fallback."""
    return jit_reduce(a, axis, "mean")


# ── Normalization (Phase 1 Sprint 1.4: mean-reduce + broadcast + sqrt) ───────


def _jit_norm(op_name: str, sym: str, a: np.ndarray, eps: float) -> np.ndarray:
    a = np.asarray(a)
    _f32_envelope_check([a])
    if a.ndim < 1:
        raise TesseraJitError("Phase 1 norm requires rank>=1")
    a = np.ascontiguousarray(a)
    t = "tensor<" + "x".join(str(s) for s in a.shape) + "xf32>"
    mlir = (
        f"func.func @{sym}(%x: {t}) -> {t} {{\n"
        f"  %0 = {op_name} %x {{eps = {eps:.9e} : f64}} : ({t}) -> {t}\n"
        f"  return %0 : {t}\n"
        f"}}\n"
    )
    handle = compile_module(mlir)
    try:
        out = np.empty_like(a)
        invoke(handle, sym, [a], out)
        return out
    finally:
        destroy(handle)


def jit_rmsnorm(a: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Production-lane unweighted RMSNorm over the innermost axis (f32).

    ``x / sqrt(mean(x²) + eps)``. No fallback.
    """
    return _jit_norm("tessera.rmsnorm", "tessera_jit_rmsnorm", a, eps)


def jit_layer_norm(a: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Production-lane unweighted LayerNorm over the innermost axis (f32).

    ``(x - mean) / sqrt(var + eps)``. No fallback.
    """
    return _jit_norm("tessera.layer_norm", "tessera_jit_layer_norm", a, eps)


# ── Activations (Phase 1 Sprint 1.6: unary math family) ──────────────────────


def _jit_unary(op_name: str, sym: str, a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    _f32_envelope_check([a])  # f32 keeps the oracle tight (bf16 acts: later)
    if a.ndim < 1:
        raise TesseraJitError("Phase 1 unary requires rank>=1")
    a = np.ascontiguousarray(a)
    t = "tensor<" + "x".join(str(s) for s in a.shape) + "xf32>"
    mlir = (
        f"func.func @{sym}(%x: {t}) -> {t} {{\n"
        f"  %0 = {op_name} %x : ({t}) -> {t}\n"
        f"  return %0 : {t}\n"
        f"}}\n"
    )
    handle = compile_module(mlir)
    try:
        out = np.empty_like(a)
        invoke(handle, sym, [a], out)
        return out
    finally:
        destroy(handle)


def jit_relu(a: np.ndarray) -> np.ndarray:
    """Production-lane ReLU = max(x, 0) (f32). No fallback."""
    return _jit_unary("tessera.relu", "tessera_jit_relu", a)


def jit_sigmoid(a: np.ndarray) -> np.ndarray:
    """Production-lane sigmoid = 1/(1+exp(-x)) (f32). No fallback."""
    return _jit_unary("tessera.sigmoid", "tessera_jit_sigmoid", a)


def jit_tanh(a: np.ndarray) -> np.ndarray:
    """Production-lane tanh (f32). No fallback."""
    return _jit_unary("tessera.tanh", "tessera_jit_tanh", a)


def jit_silu(a: np.ndarray) -> np.ndarray:
    """Production-lane SiLU/swish = x*sigmoid(x) (f32). No fallback."""
    return _jit_unary("tessera.silu", "tessera_jit_silu", a)


def jit_gelu(a: np.ndarray) -> np.ndarray:
    """Production-lane GELU, tanh approximation (GPT-2/BERT form, f32).

    ``0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x³)))``. No fallback.
    """
    return _jit_unary("tessera.gelu", "tessera_jit_gelu", a)
