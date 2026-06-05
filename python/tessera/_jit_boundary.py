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

# Tessera dtype name → (numpy dtype, ctypes elem type, MLIR element-type spell).
# RUNTIME_ABI_SPEC.md §12.5. f16/bf16 land in a later sprint under this table.
_DTYPE_TABLE: dict[str, tuple[Any, Any, str]] = {
    "f32": (np.float32, ctypes.c_float, "f32"),
}


def _dtype_entry(arr: np.ndarray):
    for tag, (np_dt, ct_dt, mlir) in _DTYPE_TABLE.items():
        if arr.dtype == np_dt:
            return tag, ct_dt, mlir
    raise TesseraJitError(
        f"Phase 1 boundary supports {sorted(_DTYPE_TABLE)} only (got {arr.dtype})"
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


def _mlir_for_binary(op_name: str, sym: str, shape: tuple[int, ...]) -> str:
    t = "tensor<" + "x".join(str(s) for s in shape) + "xf32>"
    return (
        f"func.func @{sym}(%a: {t}, %b: {t}) -> {t} {{\n"
        f"  %0 = {op_name} %a, %b : ({t}, {t}) -> {t}\n"
        f"  return %0 : {t}\n"
        f"}}\n"
    )


def _jit_binary(op_name: str, sym: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    b = np.asarray(b)
    _f32_envelope_check([a, b])
    if a.shape != b.shape:
        raise TesseraJitError(
            f"elementwise requires equal shapes (got {a.shape}, {b.shape})"
        )
    if a.ndim < 1:
        raise TesseraJitError("Phase 1 boundary requires rank>=1")
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    handle = compile_module(_mlir_for_binary(op_name, sym, tuple(a.shape)))
    try:
        out = np.empty_like(a)
        invoke(handle, sym, [a, b], out)
        return out
    finally:
        destroy(handle)


def jit_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Production-lane elementwise add (rank>=1 f32). No fallback."""
    return _jit_binary("tessera.add", "tessera_jit_add", a, b)


def jit_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Production-lane elementwise sub (rank>=1 f32). No fallback."""
    return _jit_binary("tessera.sub", "tessera_jit_sub", a, b)


def jit_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Production-lane elementwise mul (rank>=1 f32). No fallback."""
    return _jit_binary("tessera.mul", "tessera_jit_mul", a, b)


def jit_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Production-lane rank-2 f32 matmul. No transposes, no fallback."""
    a = np.asarray(a)
    b = np.asarray(b)
    _f32_envelope_check([a, b])
    if a.ndim != 2 or b.ndim != 2:
        raise TesseraJitError(
            f"Phase 1 jit_matmul is rank-2 only (got {a.shape}, {b.shape})"
        )
    if a.shape[1] != b.shape[0]:
        raise TesseraJitError(
            f"matmul shape mismatch: {a.shape} @ {b.shape}"
        )
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    M, K = int(a.shape[0]), int(a.shape[1])
    N = int(b.shape[1])
    ta = f"tensor<{M}x{K}xf32>"
    tb = f"tensor<{K}x{N}xf32>"
    tc = f"tensor<{M}x{N}xf32>"
    mlir = (
        f"func.func @tessera_jit_matmul(%a: {ta}, %b: {tb}) -> {tc} {{\n"
        f"  %0 = tessera.matmul %a, %b : ({ta}, {tb}) -> {tc}\n"
        f"  return %0 : {tc}\n"
        f"}}\n"
    )
    handle = compile_module(mlir)
    try:
        out = np.empty((M, N), dtype=np.float32)
        invoke(handle, "tessera_jit_matmul", [a, b], out)
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
