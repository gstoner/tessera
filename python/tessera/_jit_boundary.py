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

_TESSERA_OPT_PATH: Any = "unset"


def _find_tessera_opt():
    """Locate the built `tessera-opt` binary, or None. Honors $TESSERA_OPT_BIN,
    then the in-repo build dir, then PATH. Cached (used by run_via_target_ir)."""
    global _TESSERA_OPT_PATH
    if _TESSERA_OPT_PATH != "unset":
        return _TESSERA_OPT_PATH
    import shutil
    cands = []
    env = os.environ.get("TESSERA_OPT_BIN")
    if env:
        cands.append(env)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cands.append(os.path.join(root, "build/tools/tessera-opt/tessera-opt"))
    which = shutil.which("tessera-opt")
    if which:
        cands.append(which)
    _TESSERA_OPT_PATH = next((p for p in cands if p and os.path.exists(p)), None)
    return _TESSERA_OPT_PATH

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
    lib.tessera_jit_compile_count.restype = ctypes.c_int64
    lib.tessera_jit_compile_count.argtypes = []
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


def compile_count() -> int:
    """Successful JIT *compiles* since process start. With the compilation cache
    (below) on, repeated same-shape calls do NOT advance this."""
    return int(_load().tessera_jit_compile_count())


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


def _make_descriptor(arr: np.ndarray, elem_ct: Any = None):
    """Build a memref descriptor for `arr`. RUNTIME_ABI_SPEC §12.2 / §12.4:
    identity layout, C-contiguous, offset 0. The element ctype is derived from
    `arr`'s own dtype (so different args/outs may have different dtypes).
    """
    if not arr.flags.c_contiguous:
        raise TesseraJitError(
            "boundary buffers must be C-contiguous (§12.4); "
            "caller must materialize a contiguous copy"
        )
    if elem_ct is None:
        _, elem_ct, _ = _dtype_entry(arr)
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


# ── Compile + invoke (with transparent compilation cache) ───────────────────
#
# Parse→lower→JIT is expensive and deterministic in the MLIR text, so we cache
# the compiled handle keyed on that text. Repeated same-shape calls reuse the
# cached ExecutionEngine — no recompile (proven by `compile_count()` not
# advancing). Caching is transparent: `compile_module` returns a cached handle
# and `destroy` is a no-op for cache-owned handles; the cache is freed at exit
# (or explicitly via `clear_cache()`). Each invoke still runs (and counts).

import atexit  # noqa: E402  (kept local to the cache section)
import threading  # noqa: E402

_COMPILE_CACHE: dict[str, int] = {}
_CACHED_HANDLES: set[int] = set()
_CACHE_LOCK = threading.Lock()
_CACHE_ENABLED = True


def _raw_compile(mlir_text: str) -> int:
    lib = _load()
    handle = lib.tessera_jit_compile(mlir_text.encode("utf-8"))
    if not handle:
        raise TesseraJitError(lib.tessera_jit_last_error().decode("utf-8", "replace"))
    return handle


def compile_module(mlir_text: str) -> int:
    """Compile an MLIR module through the production pipeline (cache-backed).

    Returns an opaque integer handle. On a cache hit the cached handle is
    returned without recompiling. Raises :class:`TesseraJitError` on failure.
    """
    if not _CACHE_ENABLED:
        return _raw_compile(mlir_text)
    with _CACHE_LOCK:
        hit = _COMPILE_CACHE.get(mlir_text)
        if hit is not None:
            return hit
    # Compile outside the lock (slow); double-check on insert.
    handle = _raw_compile(mlir_text)
    with _CACHE_LOCK:
        existing = _COMPILE_CACHE.get(mlir_text)
        if existing is not None:
            # Lost a race; keep the first, drop ours.
            _load().tessera_jit_destroy(handle)
            return existing
        _COMPILE_CACHE[mlir_text] = handle
        _CACHED_HANDLES.add(handle)
        return handle


def destroy(handle: int) -> None:
    """Destroy a handle. No-op for cache-owned handles (freed at exit)."""
    if not handle:
        return
    with _CACHE_LOCK:
        if handle in _CACHED_HANDLES:
            return  # cache owns the lifetime
    _load().tessera_jit_destroy(handle)


def set_cache_enabled(enabled: bool) -> None:
    """Enable/disable the compilation cache (mostly for benchmarking/tests)."""
    global _CACHE_ENABLED
    _CACHE_ENABLED = enabled


def cache_size() -> int:
    with _CACHE_LOCK:
        return len(_COMPILE_CACHE)


def clear_cache() -> None:
    """Destroy all cached compiled functions and empty the cache."""
    with _CACHE_LOCK:
        handles = list(_CACHED_HANDLES)
        _COMPILE_CACHE.clear()
        _CACHED_HANDLES.clear()
    lib = _load()
    for h in handles:
        lib.tessera_jit_destroy(h)


@atexit.register
def _free_cache_at_exit() -> None:  # pragma: no cover - process teardown
    try:
        clear_cache()
    except Exception:
        pass


def invoke(
    handle: int,
    symbol: str,
    arrays: Sequence[np.ndarray],
    out,
) -> None:
    """Invoke a compiled function via the C ABI.

    ``arrays`` is the list of *input* arrays (c-iface order). ``out`` is the
    caller-allocated destination — a single ndarray, or a list/tuple of them for
    a multi-result function (DPS out-params in result order, §12.3). Each buffer's
    dtype is read from the array itself. Raises :class:`TesseraJitError` on any
    failure — never silently falls back.
    """
    lib = _load()
    outs = list(out) if isinstance(out, (list, tuple)) else [out]
    descs = [_make_descriptor(a) for a in (*arrays, *outs)]
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


def jit_select(cond: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Production-lane elementwise select: cond != 0 ? a : b (f32). No fallback."""
    cond = np.asarray(cond)
    a = np.asarray(a)
    b = np.asarray(b)
    elem, _ = _resolve_elem([cond, a, b])
    if not (cond.shape == a.shape == b.shape):
        raise TesseraJitError(
            f"select needs equal shapes ({cond.shape},{a.shape},{b.shape})"
        )
    cond, a, b = map(np.ascontiguousarray, (cond, a, b))
    t = "tensor<" + "x".join(str(s) for s in a.shape) + f"x{elem}>"
    sym = "tessera_jit_select"
    mlir = (
        f"func.func @{sym}(%c: {t}, %a: {t}, %b: {t}) -> {t} {{\n"
        f"  %0 = tessera.select %c, %a, %b : ({t}, {t}, {t}) -> {t}\n"
        f"  return %0 : {t}\n}}\n"
    )
    handle = compile_module(mlir)
    try:
        out = np.empty_like(a)
        invoke(handle, sym, [cond, a, b], out)
        return out
    finally:
        destroy(handle)


def jit_write_row(buffer: np.ndarray, value: np.ndarray, row: int) -> np.ndarray:
    """Functional KV-cache update: `buffer` (T,D) with row `row` set to `value`
    (1,D), returning the updated buffer (f32). Value-semantic — thread the result
    into the next decode step. No fallback."""
    buffer = np.asarray(buffer)
    value = np.asarray(value)
    elem, _ = _resolve_elem([buffer, value])
    if buffer.ndim != 2 or value.shape != (1, buffer.shape[1]):
        raise TesseraJitError(
            f"write_row: buffer (T,D), value (1,D); got {buffer.shape},{value.shape}"
        )
    if not (0 <= row < buffer.shape[0]):
        raise TesseraJitError(f"row {row} out of range for {buffer.shape}")
    buffer = np.ascontiguousarray(buffer)
    value = np.ascontiguousarray(value)
    tb = "tensor<" + "x".join(map(str, buffer.shape)) + f"x{elem}>"
    tv = "tensor<" + "x".join(map(str, value.shape)) + f"x{elem}>"
    sym = "tessera_jit_write_row"
    mlir = (
        f"func.func @{sym}(%b: {tb}, %v: {tv}) -> {tb} {{\n"
        f"  %0 = tessera.write_row %b, %v {{row = {int(row)} : i64}} "
        f": ({tb}, {tv}) -> {tb}\n"
        f"  return %0 : {tb}\n}}\n"
    )
    handle = compile_module(mlir)
    try:
        out = np.empty_like(buffer)
        invoke(handle, sym, [buffer, value], out)
        return out
    finally:
        destroy(handle)


def jit_masked_fill(
    x: np.ndarray, mask: np.ndarray, value: float = -1e9
) -> np.ndarray:
    """Production-lane masked fill: mask != 0 ? x : value (f32).

    The attention-masking primitive: ``masked_fill(scores, causal_mask, -1e9)``
    before softmax. No fallback.
    """
    x = np.asarray(x)
    mask = np.asarray(mask)
    elem, _ = _resolve_elem([x, mask])
    if x.shape != mask.shape:
        raise TesseraJitError(f"masked_fill shape mismatch ({x.shape},{mask.shape})")
    x, mask = np.ascontiguousarray(x), np.ascontiguousarray(mask)
    t = "tensor<" + "x".join(str(s) for s in x.shape) + f"x{elem}>"
    sym = "tessera_jit_masked_fill"
    mlir = (
        f"func.func @{sym}(%x: {t}, %m: {t}) -> {t} {{\n"
        f"  %0 = tessera.masked_fill %x, %m {{value = {value:.9e} : f64}} "
        f": ({t}, {t}) -> {t}\n"
        f"  return %0 : {t}\n}}\n"
    )
    handle = compile_module(mlir)
    try:
        out = np.empty_like(x)
        invoke(handle, sym, [x, mask], out)
        return out
    finally:
        destroy(handle)


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


def jit_bmm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Production-lane batched matmul C[i] = A[i] @ B[i] (rank-3, f32 or bf16).

    A is (B, M, K), B is (B, K, N), result (B, M, N). bf16 accumulates in f32
    then truncates (ABI §12.5). No fallback.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    elem, npdt = _resolve_elem([a, b])
    if a.ndim != 3 or b.ndim != 3:
        raise TesseraJitError(f"jit_bmm is rank-3 only (got {a.shape}, {b.shape})")
    if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise TesseraJitError(f"bmm shape mismatch: {a.shape} @ {b.shape}")
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    Bb, M, K = (int(s) for s in a.shape)
    N = int(b.shape[2])
    ta = f"tensor<{Bb}x{M}x{K}x{elem}>"
    tb = f"tensor<{Bb}x{K}x{N}x{elem}>"
    tc = f"tensor<{Bb}x{M}x{N}x{elem}>"
    sym = "tessera_jit_bmm"
    mlir = (
        f"func.func @{sym}(%a: {ta}, %b: {tb}) -> {tc} {{\n"
        f"  %0 = tessera.batched_gemm %a, %b : ({ta}, {tb}) -> {tc}\n"
        f"  return %0 : {tc}\n"
        f"}}\n"
    )
    handle = compile_module(mlir)
    try:
        out = np.empty((Bb, M, N), dtype=npdt)
        invoke(handle, sym, [a, b], out)
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


# ── Graph compilation (Phase 1 Sprint 1.8) ───────────────────────────────────
#
# The leap from op-at-a-time (each jit_* helper compiles a single-op module and
# round-trips through the boundary) to compiling a whole MULTI-OP tessera
# function as ONE JIT'd unit. Intermediates stay inside the compiled function —
# no per-op descriptor packing, and the lowering can fuse across ops. The harness
# already supports this (markCInterface + the DPS rewrite walk every function);
# GraphFn is just the Python surface to express the graph.

_ELEM_TO_NP: dict[str, Any] = {"f32": np.float32}
if _BF16 is not None:
    _ELEM_TO_NP["bf16"] = _BF16


def _i32_array(xs) -> str:
    """MLIR ``DenseI32ArrayAttr`` text. Empty must be ``array<i32>`` (no colon)."""
    xs = list(xs)
    return "array<i32: " + ", ".join(str(int(x)) for x in xs) + ">" if xs \
        else "array<i32>"


def _f32_array(xs) -> str:
    xs = list(xs)
    return "array<f32: " + ", ".join(f"{x:.9e}" for x in xs) + ">" if xs \
        else "array<f32>"


def _i64_array(xs) -> str:
    xs = list(xs)
    return "array<i64: " + ", ".join(str(int(x)) for x in xs) + ">" if xs \
        else "array<i64>"


class _Val:
    """An SSA value in a GraphFn: its name and static shape."""

    __slots__ = ("ssa", "shape")

    def __init__(self, ssa: str, shape: tuple[int, ...]):
        self.ssa = ssa
        self.shape = shape


class _GraphOp:
    """A structured record of one straight-line graph op, used by the Apple GPU
    executor (Sprint 3.3). The CPU lane ignores these and lowers the MLIR text."""

    __slots__ = ("op", "ins", "out", "meta")

    def __init__(self, op: str, ins, out: _Val, meta: dict):
        self.op = op
        self.ins = list(ins)
        self.out = out
        self.meta = meta


class GraphFn:
    """Build a multi-op `tessera` function and compile/run it as one unit.

    Example (single-head attention, scale folded into Q):
        g = GraphFn()
        q, k, v = g.arg((T, d)), g.arg((T, d)), g.arg((T, d))
        s = g.matmul(q, k, transpose_b=True)   # Q Kᵀ
        p = g.softmax(s)
        o = g.matmul(p, v)
        g.ret(o)
        out = g.run(q_arr, k_arr, v_arr)        # ONE compile, ONE invoke
    """

    def __init__(
        self,
        name: str = "tessera_jit_graph",
        elem: str = "f32",
        target: str = "cpu",
    ):
        if elem not in _ELEM_TO_NP:
            raise TesseraJitError(f"GraphFn elem must be one of {sorted(_ELEM_TO_NP)}")
        if target not in ("cpu", "apple_gpu"):
            raise TesseraJitError("GraphFn target must be 'cpu' or 'apple_gpu'")
        self._name = name
        self._elem = elem
        self._target = target
        self._args: list[tuple[str, tuple[int, ...]]] = []
        self._lines: list[str] = []
        # Scope stack for nested regions (scf.for/scf.if bodies). _emit appends to
        # the innermost open scope; the function body is scopes[0] == _lines.
        self._scopes: list[list[str]] = [self._lines]
        self._ctr = 0
        self._idx = 0  # unique suffix for block args / index constants
        self._rets: list[_Val] = []
        # Structured straight-line op log + flags for the Apple GPU executor.
        self._ops: list[_GraphOp] = []
        self._has_control_flow = False
        self._last_dispatch: list[str] = []  # kernels fired by the last GPU run
        # Cached mlpkg pipeline (whole-graph → one MPSGraph dispatch lane).
        self._mlpkg_pipe: Any = None
        self._mlpkg_dir: Any = None
        self._mlpkg_out_shape: Any = None
        # Structured control-flow records for the apple_gpu lane: a bounded
        # for_loop (G-A, MPSGraph forLoop) and/or a cond (G-A.2, MPSGraph if).
        self._loop: Any = None
        self._cond: Any = None
        self._while: Any = None

    def _t(self, shape: tuple[int, ...]) -> str:
        return "tensor<" + "x".join(str(s) for s in shape) + f"x{self._elem}>"

    def _fresh(self) -> str:
        self._ctr += 1
        return f"%v{self._ctr}"

    def arg(self, shape) -> _Val:
        ssa = f"%arg{len(self._args)}"
        shape = tuple(int(s) for s in shape)
        self._args.append((ssa, shape))
        return _Val(ssa, shape)

    def _emit(self, op: str, ins, out_shape, attrs=None, meta=None) -> _Val:
        res = self._fresh()
        in_types = ", ".join(self._t(o.shape) for o in ins)
        in_ssas = ", ".join(o.ssa for o in ins)
        attr_str = (" {" + ", ".join(attrs) + "}") if attrs else ""
        self._scopes[-1].append(
            f"  {res} = {op} {in_ssas}{attr_str} : "
            f"({in_types}) -> {self._t(tuple(out_shape))}"
        )
        out = _Val(res, tuple(out_shape))
        self._ops.append(_GraphOp(op, ins, out, meta or {}))
        return out

    # elementwise (same shape)
    def add(self, a, b):
        return self._binary("tessera.add", a, b)

    def sub(self, a, b):
        return self._binary("tessera.sub", a, b)

    def mul(self, a, b):
        return self._binary("tessera.mul", a, b)

    def div(self, a, b):
        return self._binary("tessera.div", a, b)

    def _binary(self, op, a, b):
        if a.shape != b.shape:
            raise TesseraJitError(f"{op} needs equal shapes ({a.shape} vs {b.shape})")
        return self._emit(op, [a, b], a.shape)

    # unary (same shape)
    def relu(self, a):
        return self._emit("tessera.relu", [a], a.shape)

    def sigmoid(self, a):
        return self._emit("tessera.sigmoid", [a], a.shape)

    def tanh(self, a):
        return self._emit("tessera.tanh", [a], a.shape)

    def silu(self, a):
        return self._emit("tessera.silu", [a], a.shape)

    def gelu(self, a):
        return self._emit("tessera.gelu", [a], a.shape)

    def softmax(self, a, axis: int = -1):
        ax = axis + len(a.shape) if axis < 0 else axis
        return self._emit(
            "tessera.softmax", [a], a.shape, [f"axis = {ax} : i64"],
            meta={"axis": ax, "rank": len(a.shape)},
        )

    def rmsnorm(self, a, eps: float = 1e-5):
        return self._emit(
            "tessera.rmsnorm", [a], a.shape, [f"eps = {eps:.9e} : f64"],
            meta={"eps": float(eps)},
        )

    def layer_norm(self, a, eps: float = 1e-5):
        return self._emit(
            "tessera.layer_norm", [a], a.shape, [f"eps = {eps:.9e} : f64"],
            meta={"eps": float(eps)},
        )

    def transpose(self, a):
        if len(a.shape) != 2:
            raise TesseraJitError("GraphFn.transpose is rank-2 only")
        return self._emit("tessera.transpose", [a], (a.shape[1], a.shape[0]))

    def matmul(self, a, b, transpose_a: bool = False, transpose_b: bool = False):
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise TesseraJitError("GraphFn.matmul is rank-2 only")
        M, Ka = (a.shape[1], a.shape[0]) if transpose_a else (a.shape[0], a.shape[1])
        Kb, N = (b.shape[1], b.shape[0]) if transpose_b else (b.shape[0], b.shape[1])
        if Ka != Kb:
            raise TesseraJitError(f"matmul contracting mismatch: {a.shape},{b.shape}")
        attrs = []
        if transpose_a:
            attrs.append("transposeA = true")
        if transpose_b:
            attrs.append("transposeB = true")
        return self._emit(
            "tessera.matmul", [a, b], (M, N), attrs or None,
            meta={"ta": bool(transpose_a), "tb": bool(transpose_b)},
        )

    def select(self, cond, a, b):
        if not (cond.shape == a.shape == b.shape):
            raise TesseraJitError("GraphFn.select needs equal shapes")
        return self._emit("tessera.select", [cond, a, b], a.shape)

    def masked_fill(self, x, mask, value: float = -1e9):
        if x.shape != mask.shape:
            raise TesseraJitError("GraphFn.masked_fill shape mismatch")
        return self._emit(
            "tessera.masked_fill", [x, mask], x.shape, [f"value = {value:.9e} : f64"]
        )

    def bmm(self, a, b):
        if len(a.shape) != 3 or len(b.shape) != 3:
            raise TesseraJitError("GraphFn.bmm is rank-3 only")
        Bb, M, K = a.shape
        Bb2, K2, N = b.shape
        if Bb != Bb2 or K != K2:
            raise TesseraJitError(f"bmm shape mismatch: {a.shape},{b.shape}")
        return self._emit("tessera.batched_gemm", [a, b], (Bb, M, N))

    def for_loop(self, count: int, init: _Val, body):
        """Emit an scf.for loop with a single tensor carry.

        `body(carry) -> _Val` builds the loop body (it may reference outer SSA
        values — function args, earlier results) and returns the next carry. The
        loop runs `count` times; the final carry is returned. Control flow
        through tessera→linalg→scf→llvm, one compiled function.
        """
        self._has_control_flow = True
        T = self._t(init.shape)
        self._idx += 1
        i = self._idx
        lb, ub, st = f"%lb{i}", f"%ub{i}", f"%st{i}"
        iv, carry = f"%i{i}", f"%carry{i}"
        cur = self._scopes[-1]
        cur.append(f"  {lb} = arith.constant 0 : index")
        cur.append(f"  {ub} = arith.constant {int(count)} : index")
        cur.append(f"  {st} = arith.constant 1 : index")
        res = self._fresh()
        self._scopes.append([])  # open the loop body scope
        _body_start = len(self._ops)  # capture the structured body for the GPU lane
        nxt = body(_Val(carry, init.shape))
        if nxt.shape != init.shape:
            raise TesseraJitError("for_loop body must preserve the carry shape")
        body_lines = self._scopes.pop()
        # Structured loop record for the apple_gpu forLoop lane (G-A). v1 supports
        # a single loop; a second one disables the GPU loop path.
        if self._loop is None:
            self._loop = {
                "trip": int(count),
                "carry_ssa": carry,
                "next_carry_ssa": nxt.ssa,
                "init_ssa": init.ssa,
                "body_ops": list(self._ops[_body_start:]),
                "result_ssa": res,
                "carry_shape": tuple(init.shape),
            }
        else:
            self._loop = {"_unsupported": "more than one for_loop"}
        cur.append(
            f"  {res} = scf.for {iv} = {lb} to {ub} step {st} "
            f"iter_args({carry} = {init.ssa}) -> {T} {{"
        )
        cur.extend(body_lines)
        cur.append(f"    scf.yield {nxt.ssa} : {T}")
        cur.append("  }")
        return _Val(res, init.shape)

    def cond(self, flag: _Val, then_fn, else_fn):
        """Emit an scf.if: if `flag[0] > 0` run `then_fn()` else `else_fn()`.

        `flag` is a shape-(1,) tensor input (the runtime condition); only the
        taken branch executes (divergent control flow, not data-parallel select).
        Both branches must return _Vals of the same shape.
        """
        if flag.shape != (1,):
            raise TesseraJitError("GraphFn.cond flag must be shape (1,)")
        self._has_control_flow = True
        self._idx += 1
        i = self._idx
        c0, fv, zero, p = f"%c0_{i}", f"%f{i}", f"%z{i}", f"%p{i}"
        cur = self._scopes[-1]
        cur.append(f"  {c0} = arith.constant 0 : index")
        cur.append(f"  {fv} = tensor.extract {flag.ssa}[{c0}] : {self._t((1,))}")
        cur.append(f"  {zero} = arith.constant 0.0 : {self._elem}")
        cur.append(f"  {p} = arith.cmpf ogt, {fv}, {zero} : {self._elem}")
        res = self._fresh()
        self._scopes.append([])
        _then_start = len(self._ops)
        tval = then_fn()
        tbody = self._scopes.pop()
        self._scopes.append([])
        _else_start = len(self._ops)
        eval_ = else_fn()
        ebody = self._scopes.pop()
        if tval.shape != eval_.shape:
            raise TesseraJitError("cond branches must return the same shape")
        # Structured cond record for the apple_gpu MPSGraph-`if` lane (G-A.2).
        if self._cond is None:
            self._cond = {
                "flag_ssa": flag.ssa,
                "then_ops": list(self._ops[_then_start:_else_start]),
                "then_out_ssa": tval.ssa,
                "else_ops": list(self._ops[_else_start:]),
                "else_out_ssa": eval_.ssa,
                "result_ssa": res,
                "out_shape": tuple(tval.shape),
            }
        else:
            self._cond = {"_unsupported": "more than one cond"}
        T = self._t(tval.shape)
        cur.append(f"  {res} = scf.if {p} -> {T} {{")
        cur.extend(tbody)
        cur.append(f"    scf.yield {tval.ssa} : {T}")
        cur.append("  } else {")
        cur.extend(ebody)
        cur.append(f"    scf.yield {eval_.ssa} : {T}")
        cur.append("  }")
        return _Val(res, tval.shape)

    def while_loop(self, max_iters: int, cond, body, init: _Val):
        """Bounded ``while``: for up to ``max_iters`` steps, while
        ``cond(carry) > 0`` keep updating ``carry = body(carry)``; return the
        final carry. Lowers to an MPSGraph ``forLoop`` + select-masking on the
        apple_gpu lane (MPSGraph's native ``while`` is unstable) — once the
        predicate goes false the carry freezes. ``cond(carry) -> _Val`` (the
        predicate source; ``> 0`` means continue), ``body(carry) -> _Val`` (the
        next carry, same shape). v1: apple_gpu only (no CPU scf.while lane), f32,
        init must be a function arg."""
        if self._target != "apple_gpu":
            raise TesseraJitError("while_loop is apple_gpu-only (no CPU scf.while lane)")
        self._has_control_flow = True
        self._idx += 1
        carry_ssa = f"%wcarry{self._idx}"
        cv = _Val(carry_ssa, init.shape)
        body_start = len(self._ops)
        nxt = body(cv)
        if nxt.shape != init.shape:
            raise TesseraJitError("while_loop body must preserve the carry shape")
        cond_start = len(self._ops)
        pred = cond(cv)
        res = self._fresh()
        if self._while is None:
            self._while = {
                "max_iters": int(max_iters),
                "carry_ssa": carry_ssa,
                "init_ssa": init.ssa,
                "next_carry_ssa": nxt.ssa,
                "pred_ssa": pred.ssa,
                "result_ssa": res,
                "body_ops": list(self._ops[body_start:cond_start]),
                "cond_ops": list(self._ops[cond_start:]),
                "carry_shape": tuple(init.shape),
            }
        else:
            self._while = {"_unsupported": "more than one while_loop"}
        return _Val(res, init.shape)

    def write_row(self, buffer, value, row: int):
        """Functional KV-cache update: buffer (T,D) with row `row` set to value
        (1,D). Returns the updated buffer (thread it for the next step)."""
        if len(buffer.shape) != 2 or value.shape != (1, buffer.shape[1]):
            raise TesseraJitError("write_row: buffer (T,D), value (1,D)")
        return self._emit(
            "tessera.write_row", [buffer, value], buffer.shape, [f"row = {int(row)} : i64"]
        )

    def ret(self, *vals: _Val):
        """Set the function result(s). One or more values (multi-result → the
        decode step returns out + updated caches in a single compiled function)."""
        if not vals:
            raise TesseraJitError("ret() needs at least one value")
        self._rets = list(vals)

    def build(self) -> str:
        if not self._rets:
            raise TesseraJitError("GraphFn has no return value (call ret())")
        arg_decl = ", ".join(f"{s}: {self._t(sh)}" for s, sh in self._args)
        rtypes = [self._t(r.shape) for r in self._rets]
        rt = rtypes[0] if len(rtypes) == 1 else "(" + ", ".join(rtypes) + ")"
        ret_ssas = ", ".join(r.ssa for r in self._rets)
        body = "\n".join(self._lines)
        return (
            f"func.func @{self._name}({arg_decl}) -> {rt} {{\n"
            f"{body}\n"
            f"  return {ret_ssas} : {', '.join(rtypes)}\n"
            f"}}\n"
        )

    def run(self, *arrays: np.ndarray):
        """Compile the whole graph ONCE and invoke ONCE. Returns the result
        array, or a tuple of arrays for a multi-result function.

        For `target="apple_gpu"` there is no MLIR Metal backend (D2), so the
        graph is interpreted op-by-op against the bespoke Metal back-half
        (`_apple_gpu_backend`), with the canonical fused chains
        (matmul→softmax→matmul, matmul→softmax, matmul→gelu, matmul→rmsnorm)
        collapsed to single fused Metal kernels."""
        if self._target == "apple_gpu":
            if sum(x is not None for x in (self._loop, self._cond, self._while)) > 1:
                raise TesseraJitError(
                    "apple_gpu GraphFn: mixing for_loop/cond/while is not supported (v1)")
            if self._loop is not None:
                return self._run_apple_gpu_loop(arrays)
            if self._cond is not None:
                return self._run_apple_gpu_cond(arrays)
            if self._while is not None:
                return self._run_apple_gpu_while(arrays)
            return self._run_apple_gpu(arrays)
        if not self._rets:
            raise TesseraJitError("GraphFn has no return value (call ret())")
        if len(arrays) != len(self._args):
            raise TesseraJitError(
                f"expected {len(self._args)} args, got {len(arrays)}"
            )
        npdt = _ELEM_TO_NP[self._elem]
        ins = [np.ascontiguousarray(np.asarray(a)) for a in arrays]
        for a, (_ssa, sh) in zip(ins, self._args):
            if a.dtype != npdt or tuple(a.shape) != sh:
                raise TesseraJitError(
                    f"arg dtype/shape mismatch: got {a.dtype}{a.shape}, want {npdt}{sh}"
                )
        handle = compile_module(self.build())
        try:
            outs = [np.empty(r.shape, dtype=npdt) for r in self._rets]
            invoke(handle, self._name, ins, outs)
            return outs[0] if len(outs) == 1 else tuple(outs)
        finally:
            destroy(handle)

    # ── Apple GPU graph executor (Sprint 3.3) ────────────────────────────────

    def last_dispatch(self) -> list[str]:
        """Kernel names fired by the most recent ``apple_gpu`` ``run()``. Lets a
        test prove fusion happened — an attention graph fires ONE
        ``matmul_softmax_matmul`` kernel, not three separate ones."""
        return list(self._last_dispatch)

    # ── Apple GPU bounded-loop lane (Phase-G G-A) ────────────────────────────
    #
    # When the graph is `(args) → for_loop(init=arg, body) → ret(final_carry)`,
    # author it as ONE MPSGraph `forLoop` and run it in one dispatch (vs the host
    # per-iteration interpreter). The body is the recorded straight-line op-list;
    # carry threads through the loop iteration argument. f32, single carry, init
    # must be a function arg, body references only args + carry (v1).

    def _serialize_loop_spec(self):
        """Serialize the captured `_loop` to the run_graph_loop op-list ABI:
        returns (carry_arg_index, trip, body_ops, body_out_id, carry_shape).
        Shared by the direct dispatch (`_run_apple_gpu_loop`) and the Target-IR
        emit (`_emit_control_for_mlir`)."""
        loop = self._loop
        if "_unsupported" in loop:
            raise TesseraJitError(
                f"apple_gpu GraphFn loop: {loop['_unsupported']} not supported (v1)")
        if self._elem not in ("f32", "bf16"):
            raise TesseraJitError(
                f"apple_gpu GraphFn loop supports f32/bf16, got {self._elem!r}")
        if len(self._rets) != 1 or self._rets[0].ssa != loop["result_ssa"]:
            raise TesseraJitError(
                "apple_gpu GraphFn loop must return exactly the loop result")
        arg_index = {ssa: k for k, (ssa, _sh) in enumerate(self._args)}
        if loop["init_ssa"] not in arg_index:
            raise TesseraJitError(
                "apple_gpu GraphFn loop: the init carry must be a function arg (v1)")
        n_args = len(self._args)
        idof = dict(arg_index)
        idof[loop["carry_ssa"]] = n_args  # carry id sits right after the args
        body_ops, idof = self._serialize_branch(
            loop["body_ops"], idof, n_args + 1, "loop body")
        body_out_id = idof.get(loop["next_carry_ssa"])
        if body_out_id is None:
            raise TesseraJitError(
                "apple_gpu GraphFn loop: next carry is not produced by the body")
        return (arg_index[loop["init_ssa"]], loop["trip"], body_ops, body_out_id,
                loop["carry_shape"])

    def _loop_elem_np(self):
        """numpy dtype the loop's args must carry, per the GraphFn ``_elem``."""
        if self._elem == "f32":
            return np.float32
        if self._elem == "bf16":
            if _ml_dtypes is None:
                raise TesseraJitError(
                    "apple_gpu GraphFn bf16 loop needs the ml_dtypes package "
                    "(pip install '.[ml_dtypes]')")
            return _ml_dtypes.bfloat16
        raise TesseraJitError(
            f"apple_gpu GraphFn loop supports f32/bf16, got {self._elem!r}")

    def _coerce_loop_args(self, arrays):
        """Validate args against the declared ``_elem`` + recorded shapes and
        return **f32** arrays for the executor. ``run_graph_loop_f32`` (and the
        cond/while executors) are f32; bf16 control flow is host-upcast to f32 for
        the run and the result is downcast back to bf16 by the caller — so a bf16
        loop computes in f32 with an f32 carry (more accurate than per-step bf16
        rounding; a native ``run_graph_loop_bf16`` is a perf/exact-rounding
        follow-on). See ``_finalize_loop_out``."""
        want = self._loop_elem_np()
        if len(arrays) != len(self._args):
            raise TesseraJitError(
                f"expected {len(self._args)} args, got {len(arrays)}")
        ins = []
        for (_ssa, sh), arr in zip(self._args, arrays):
            a = np.ascontiguousarray(np.asarray(arr))
            if a.dtype != want or tuple(a.shape) != sh:
                raise TesseraJitError(
                    f"arg dtype/shape mismatch: got {a.dtype}{a.shape}, "
                    f"want {np.dtype(want).name}{sh}")
            ins.append(a.astype(np.float32) if want != np.float32 else a)
        return ins

    def _finalize_loop_out(self, out):
        """Downcast an f32 loop result back to the declared ``_elem`` (bf16)."""
        if self._elem == "bf16":
            return out.astype(self._loop_elem_np())
        return out

    def _run_apple_gpu_loop(self, arrays):
        from tessera import apple_mlpkg as mp

        # Serialize first (elem/loop-shape checks) so a bad elem is reported
        # before the per-arg dtype check; bf16 is host-upcast in _coerce_loop_args.
        carry_arg_index, trip, body_ops, body_out_id, carry_shape = \
            self._serialize_loop_spec()
        ins = self._coerce_loop_args(arrays)
        arg_shapes = [tuple(sh) for (_ssa, sh) in self._args]
        out = mp.run_graph_loop_f32(
            ins, arg_shapes, carry_arg_index, trip, body_ops, body_out_id,
            carry_shape)
        if out is None:
            raise TesseraJitError(
                "apple_gpu GraphFn loop dispatch failed / runtime unavailable")
        self._last_dispatch = ["forloop"]  # one MPSGraph forLoop dispatch
        return self._finalize_loop_out(out.copy())

    # ── MLIR-driven execution (Phase-G G-B.2) ────────────────────────────────
    #
    # `run_via_target_ir()` proves the lowered Target IR executes: emit the
    # `tessera.control_for` op (with the serialized op-list payload), lower it
    # through `tessera-opt --tessera-control-for-to-apple_gpu` to
    # `tessera_apple.gpu.control_loop`, then dispatch off the lowered op's
    # recorded runtime `symbol` (vs run()'s direct in-memory dispatch).

    def _emit_control_for_mlir(self) -> str:
        from tessera.apple_mlpkg import GRAPH_OP

        carry_arg_index, trip, body_ops, body_out_id, carry_shape = \
            self._serialize_loop_spec()
        codes, i0, i1, ia, fa = [], [], [], [], []
        for o in body_ops:
            codes.append(GRAPH_OP[o["op"]])
            i0.append(int(o["in0"]))
            i1.append(int(o.get("in1", -1)))
            ia.append((1 if o.get("transpose_a") else 0)
                      | (2 if o.get("transpose_b") else 0))
            fa.append(float(o.get("eps", 1e-5)))



        arg_decl = ", ".join(f"{s}: {self._t(sh)}" for s, sh in self._args)
        operand_ssas = ", ".join(s for s, _ in self._args)
        in_types = ", ".join(self._t(sh) for _, sh in self._args)
        out_ty = self._t(carry_shape)
        payload = (
            f"carry_arg_index = {carry_arg_index} : i64, "
            f"body_opcodes = {_i32_array(codes)}, body_in0 = {_i32_array(i0)}, "
            f"body_in1 = {_i32_array(i1)}, body_iattr = {_i32_array(ia)}, "
            f"body_fattr = {_f32_array(fa)}, body_out_id = {body_out_id} : i64")
        op = (f'  %r = "tessera.control_for"({operand_ssas}) '
              f"{{body = @loop_body, start = 0 : i64, stop = {trip} : i64, "
              f"step = 1 : i64, {payload}}} "
              f": ({in_types}) -> {out_ty}")
        return (f"func.func private @loop_body({out_ty}) -> {out_ty}\n"
                f"func.func @graph({arg_decl}) -> {out_ty} {{\n"
                f"{op}\n  return %r : {out_ty}\n}}\n")

    def run_via_target_ir(self, *arrays: np.ndarray):
        """Emit `tessera.control_for`, lower it through tessera-opt to
        `tessera_apple.gpu.control_loop`, and execute off the lowered op's
        recorded runtime symbol. apple_gpu bounded for_loop only."""
        import subprocess

        from tessera import apple_mlpkg as mp

        if self._target != "apple_gpu" or self._loop is None:
            raise TesseraJitError(
                "run_via_target_ir requires an apple_gpu for_loop graph")
        mlir = self._emit_control_for_mlir()  # serialize + elem check first
        ins = self._coerce_loop_args(arrays)
        opt = _find_tessera_opt()
        if opt is None:
            raise TesseraJitError("tessera-opt binary not found (build it first)")
        proc = subprocess.run(
            [opt, "--tessera-control-for-to-apple_gpu"],
            input=mlir, capture_output=True, text=True)
        if proc.returncode != 0:
            raise TesseraJitError(f"tessera-opt lowering failed: {proc.stderr}")
        out = mp.execute_control_loop_mlir(proc.stdout, ins)
        if out is None:
            raise TesseraJitError(
                "control_loop execution failed (op/payload absent or runtime down)")
        self._last_dispatch = ["control_loop"]
        return self._finalize_loop_out(out.copy())

    # ── MLIR-driven cond execution (Phase-G close-out C) ─────────────────────
    #
    # Parallel to the for_loop pair: emit `tessera.control_if` (with the then/else
    # op-list payload), lower it through `tessera-opt --tessera-control-if-to-
    # apple_gpu` to `tessera_apple.gpu.control_if`, then dispatch off the lowered
    # op's recorded `symbol` (`tessera_apple_gpu_run_graph_cond_f32`).

    @staticmethod
    def _encode_branch(ops):
        """Encode a serialized branch op-list to the (opcodes,in0,in1,iattr,fattr)
        i32/f32 arrays the control_if payload carries."""
        from tessera.apple_mlpkg import GRAPH_OP

        codes, i0, i1, ia, fa = [], [], [], [], []
        for o in ops:
            codes.append(GRAPH_OP[o["op"]])
            i0.append(int(o["in0"]))
            i1.append(int(o.get("in1", -1)))
            ia.append((1 if o.get("transpose_a") else 0)
                      | (2 if o.get("transpose_b") else 0))
            fa.append(float(o.get("eps", 1e-5)))
        return codes, i0, i1, ia, fa

    def _emit_control_if_mlir(self) -> str:
        (flag_arg_index, then_ops, then_out_id, else_ops, else_out_id,
         out_shape) = self._serialize_cond_spec()
        tc, ti0, ti1, tia, tfa = self._encode_branch(then_ops)
        ec, ei0, ei1, eia, efa = self._encode_branch(else_ops)




        arg_decl = ", ".join(f"{s}: {self._t(sh)}" for s, sh in self._args)
        operand_ssas = ", ".join(s for s, _ in self._args)
        in_types = ", ".join(self._t(sh) for _, sh in self._args)
        out_ty = self._t(out_shape)
        payload = (
            f"flag_arg_index = {flag_arg_index} : i64, "
            f"then_opcodes = {_i32_array(tc)}, then_in0 = {_i32_array(ti0)}, "
            f"then_in1 = {_i32_array(ti1)}, then_iattr = {_i32_array(tia)}, "
            f"then_fattr = {_f32_array(tfa)}, then_out_id = {then_out_id} : i64, "
            f"else_opcodes = {_i32_array(ec)}, else_in0 = {_i32_array(ei0)}, "
            f"else_in1 = {_i32_array(ei1)}, else_iattr = {_i32_array(eia)}, "
            f"else_fattr = {_f32_array(efa)}, else_out_id = {else_out_id} : i64, "
            f"out_shape = {_i64_array(out_shape)}")
        op = (f'  %r = "tessera.control_if"({operand_ssas}) '
              f"{{then_branch = @then_body, else_branch = @else_body, {payload}}} "
              f": ({in_types}) -> {out_ty}")
        return (f"func.func private @then_body({out_ty}) -> {out_ty}\n"
                f"func.func private @else_body({out_ty}) -> {out_ty}\n"
                f"func.func @graph({arg_decl}) -> {out_ty} {{\n"
                f"{op}\n  return %r : {out_ty}\n}}\n")

    def run_cond_via_target_ir(self, *arrays: np.ndarray):
        """Emit `tessera.control_if`, lower it through tessera-opt to
        `tessera_apple.gpu.control_if`, and execute off the lowered op's recorded
        runtime symbol. apple_gpu cond only."""
        import subprocess

        from tessera import apple_mlpkg as mp

        if self._target != "apple_gpu" or self._cond is None:
            raise TesseraJitError(
                "run_cond_via_target_ir requires an apple_gpu cond graph")
        mlir = self._emit_control_if_mlir()  # serialize + elem check first
        ins = self._coerce_loop_args(arrays)
        opt = _find_tessera_opt()
        if opt is None:
            raise TesseraJitError("tessera-opt binary not found (build it first)")
        proc = subprocess.run(
            [opt, "--tessera-control-if-to-apple_gpu"],
            input=mlir, capture_output=True, text=True)
        if proc.returncode != 0:
            raise TesseraJitError(f"tessera-opt lowering failed: {proc.stderr}")
        out = mp.execute_control_if_mlir(proc.stdout, ins)
        if out is None:
            raise TesseraJitError(
                "control_if execution failed (op/payload absent or runtime down)")
        self._last_dispatch = ["control_if"]
        return self._finalize_loop_out(out.copy())

    # ── MLIR-driven while execution (Phase-G close-out D) ────────────────────
    #
    # Parallel to the for/cond pairs: emit `tessera.control_while` (with the
    # body+cond op-list payload), lower it through `tessera-opt
    # --tessera-control-while-to-apple_gpu` to `tessera_apple.gpu.control_while`,
    # then dispatch off the recorded `symbol`
    # (`tessera_apple_gpu_run_graph_while_f32`).

    def _emit_control_while_mlir(self) -> str:
        (carry_arg_index, max_iters, body_ops, body_out_id, cond_ops,
         cond_out_id, carry_shape) = self._serialize_while_spec()
        bc, bi0, bi1, bia, bfa = self._encode_branch(body_ops)
        cc, ci0, ci1, cia, cfa = self._encode_branch(cond_ops)



        arg_decl = ", ".join(f"{s}: {self._t(sh)}" for s, sh in self._args)
        operand_ssas = ", ".join(s for s, _ in self._args)
        in_types = ", ".join(self._t(sh) for _, sh in self._args)
        out_ty = self._t(carry_shape)
        payload = (
            f"carry_arg_index = {carry_arg_index} : i64, "
            f"max_iters = {max_iters} : i64, "
            f"body_opcodes = {_i32_array(bc)}, body_in0 = {_i32_array(bi0)}, "
            f"body_in1 = {_i32_array(bi1)}, body_iattr = {_i32_array(bia)}, "
            f"body_fattr = {_f32_array(bfa)}, body_out_id = {body_out_id} : i64, "
            f"cond_opcodes = {_i32_array(cc)}, cond_in0 = {_i32_array(ci0)}, "
            f"cond_in1 = {_i32_array(ci1)}, cond_iattr = {_i32_array(cia)}, "
            f"cond_fattr = {_f32_array(cfa)}, cond_out_id = {cond_out_id} : i64")
        op = (f'  %r = "tessera.control_while"({operand_ssas}) '
              f"{{body = @while_body, cond = @while_cond, {payload}}} "
              f": ({in_types}) -> {out_ty}")
        return (f"func.func private @while_body({out_ty}) -> {out_ty}\n"
                f"func.func private @while_cond({out_ty}) -> {out_ty}\n"
                f"func.func @graph({arg_decl}) -> {out_ty} {{\n"
                f"{op}\n  return %r : {out_ty}\n}}\n")

    def run_while_via_target_ir(self, *arrays: np.ndarray):
        """Emit `tessera.control_while`, lower it through tessera-opt to
        `tessera_apple.gpu.control_while`, and execute off the lowered op's
        recorded runtime symbol. apple_gpu bounded while only."""
        import subprocess

        from tessera import apple_mlpkg as mp

        if self._target != "apple_gpu" or self._while is None:
            raise TesseraJitError(
                "run_while_via_target_ir requires an apple_gpu while graph")
        mlir = self._emit_control_while_mlir()  # serialize + elem check first
        ins = self._coerce_loop_args(arrays)
        opt = _find_tessera_opt()
        if opt is None:
            raise TesseraJitError("tessera-opt binary not found (build it first)")
        proc = subprocess.run(
            [opt, "--tessera-control-while-to-apple_gpu"],
            input=mlir, capture_output=True, text=True)
        if proc.returncode != 0:
            raise TesseraJitError(f"tessera-opt lowering failed: {proc.stderr}")
        out = mp.execute_control_while_mlir(proc.stdout, ins)
        if out is None:
            raise TesseraJitError(
                "control_while execution failed (op/payload absent or runtime down)")
        self._last_dispatch = ["control_while"]
        return self._finalize_loop_out(out.copy())

    def _serialize_branch(self, ops, idof, base, what):
        """Serialize a straight-line op-list (loop body / cond branch) to the
        author-dict form ``run_graph_*`` expects. Op ``j``'s output gets id
        ``base + j``; ``idof`` maps already-bound ssas (args, and the carry for a
        loop) to ids. Returns ``(op_dicts, updated_idof)``. Raises on an
        unexpressible op or a reference to a value not in scope (v1)."""
        idof = dict(idof)
        out: list = []
        for j, op in enumerate(ops):
            name = self._MLPKG_OP.get(op.op)
            if name is None:
                raise TesseraJitError(
                    f"apple_gpu GraphFn {what} cannot express op {op.op!r}")
            entry: dict = {"op": name}
            try:
                entry["in0"] = idof[op.ins[0].ssa]
                if name == "matmul":
                    entry["in1"] = idof[op.ins[1].ssa]
                    entry["transpose_a"] = bool(op.meta.get("ta"))
                    entry["transpose_b"] = bool(op.meta.get("tb"))
                elif name in ("add", "sub", "mul", "div"):
                    entry["in1"] = idof[op.ins[1].ssa]
                elif name == "softmax":
                    if op.meta.get("axis") != op.meta.get("rank", 0) - 1:
                        raise TesseraJitError(
                            f"apple_gpu GraphFn {what} softmax is last-axis only")
                elif name in ("rmsnorm", "layer_norm"):
                    entry["eps"] = float(op.meta.get("eps", 1e-5))
            except KeyError:
                raise TesseraJitError(
                    f"apple_gpu GraphFn {what} references a value not in scope "
                    "(v1: only function args, the carry, and earlier body ops)")
            idof[op.out.ssa] = base + j
            out.append(entry)
        return out, idof

    # ── Apple GPU cond lane (Phase-G G-A.2) ──────────────────────────────────
    #
    # `(args) → cond(flag=arg, then, else) → ret(result)` authors ONE MPSGraph
    # `if` (predicate = flag[0] > 0) and runs it in one dispatch — only the taken
    # branch executes. Each branch is a straight-line op-list over the args; both
    # produce the same shape. f32, flag must be a function arg (v1).

    def _serialize_cond_spec(self):
        """Serialize the captured `_cond` to the run_graph_cond op-list ABI:
        returns (flag_arg_index, then_ops, then_out_id, else_ops, else_out_id,
        out_shape). Shared by the direct dispatch (`_run_apple_gpu_cond`) and the
        Target-IR emit (`_emit_control_if_mlir`)."""
        cond = self._cond
        if "_unsupported" in cond:
            raise TesseraJitError(
                f"apple_gpu GraphFn cond: {cond['_unsupported']} not supported (v1)")
        if self._elem not in ("f32", "bf16"):
            raise TesseraJitError(
                f"apple_gpu GraphFn cond supports f32/bf16, got {self._elem!r}")
        if len(self._rets) != 1 or self._rets[0].ssa != cond["result_ssa"]:
            raise TesseraJitError(
                "apple_gpu GraphFn cond must return exactly the cond result")
        arg_index = {ssa: k for k, (ssa, _sh) in enumerate(self._args)}
        if cond["flag_ssa"] not in arg_index:
            raise TesseraJitError(
                "apple_gpu GraphFn cond: the flag must be a function arg (v1)")
        n_args = len(self._args)
        then_ops, tlocal = self._serialize_branch(
            cond["then_ops"], arg_index, n_args, "cond then-branch")
        else_ops, elocal = self._serialize_branch(
            cond["else_ops"], arg_index, n_args, "cond else-branch")
        then_out_id = tlocal.get(cond["then_out_ssa"])
        else_out_id = elocal.get(cond["else_out_ssa"])
        if then_out_id is None or else_out_id is None:
            raise TesseraJitError(
                "apple_gpu GraphFn cond: a branch result is not in scope")
        return (arg_index[cond["flag_ssa"]], then_ops, then_out_id, else_ops,
                else_out_id, cond["out_shape"])

    def _run_apple_gpu_cond(self, arrays):
        from tessera import apple_mlpkg as mp

        flag_arg_index, then_ops, then_out_id, else_ops, else_out_id, out_shape = \
            self._serialize_cond_spec()
        ins = self._coerce_loop_args(arrays)
        arg_shapes = [tuple(sh) for (_ssa, sh) in self._args]
        out = mp.run_graph_cond_f32(
            ins, arg_shapes, flag_arg_index,
            then_ops, then_out_id, else_ops, else_out_id, out_shape)
        if out is None:
            raise TesseraJitError(
                "apple_gpu GraphFn cond dispatch failed / runtime unavailable")
        self._last_dispatch = ["cond"]  # one MPSGraph if dispatch
        return self._finalize_loop_out(out.copy())

    # ── Apple GPU bounded-while lane (Phase-G G-A.3) ─────────────────────────
    #
    # `(args) → while_loop(max, cond, body, init=arg) → ret(result)` authors ONE
    # MPSGraph `forLoop` with select-masking (carry freezes once cond goes false)
    # — MPSGraph's native `while` is unstable. f32, init must be a function arg.

    def _serialize_while_spec(self):
        """Serialize the captured `_while` to the run_graph_while op-list ABI:
        returns (carry_arg_index, max_iters, body_ops, body_out_id, cond_ops,
        cond_out_id, carry_shape). Shared by the direct dispatch
        (`_run_apple_gpu_while`) and the Target-IR emit
        (`_emit_control_while_mlir`)."""
        wl = self._while
        if "_unsupported" in wl:
            raise TesseraJitError(
                f"apple_gpu GraphFn while: {wl['_unsupported']} not supported (v1)")
        if self._elem not in ("f32", "bf16"):
            raise TesseraJitError(
                f"apple_gpu GraphFn while supports f32/bf16, got {self._elem!r}")
        if len(self._rets) != 1 or self._rets[0].ssa != wl["result_ssa"]:
            raise TesseraJitError(
                "apple_gpu GraphFn while must return exactly the while result")
        arg_index = {ssa: k for k, (ssa, _sh) in enumerate(self._args)}
        if wl["init_ssa"] not in arg_index:
            raise TesseraJitError(
                "apple_gpu GraphFn while: the init carry must be a function arg (v1)")
        n_args = len(self._args)
        idof0 = dict(arg_index)
        idof0[wl["carry_ssa"]] = n_args
        body_ops, blocal = self._serialize_branch(
            wl["body_ops"], idof0, n_args + 1, "while body")
        cond_ops, clocal = self._serialize_branch(
            wl["cond_ops"], idof0, n_args + 1, "while cond")
        body_out_id = blocal.get(wl["next_carry_ssa"])
        cond_out_id = clocal.get(wl["pred_ssa"])
        if body_out_id is None or cond_out_id is None:
            raise TesseraJitError(
                "apple_gpu GraphFn while: body/cond output is not in scope")
        return (arg_index[wl["init_ssa"]], wl["max_iters"], body_ops, body_out_id,
                cond_ops, cond_out_id, wl["carry_shape"])

    def _run_apple_gpu_while(self, arrays):
        from tessera import apple_mlpkg as mp

        (carry_arg_index, max_iters, body_ops, body_out_id, cond_ops, cond_out_id,
         carry_shape) = self._serialize_while_spec()
        ins = self._coerce_loop_args(arrays)
        arg_shapes = [tuple(sh) for (_ssa, sh) in self._args]
        out = mp.run_graph_while_f32(
            ins, arg_shapes, carry_arg_index, max_iters,
            body_ops, body_out_id, cond_ops, cond_out_id, carry_shape)
        if out is None:
            raise TesseraJitError(
                "apple_gpu GraphFn while dispatch failed / runtime unavailable")
        self._last_dispatch = ["while"]  # one MPSGraph forLoop+select dispatch
        return self._finalize_loop_out(out.copy())

    # ── Whole-graph lane: compile the GraphFn to ONE MPSGraph dispatch ───────
    #
    # `run()` interprets the graph op-by-op against the bespoke Metal kernels
    # (with hand fusions). `run_mlpkg()` instead authors the WHOLE straight-line
    # graph into one serialized MPSGraph package (PK8c) and dispatches it as a
    # single Metal ML pass — MPSGraph fuses globally, no per-kernel interpreter.
    # The compiled package is cached on the instance (author+compile happen once;
    # subsequent runs just re-fill inputs + dispatch). f32, single output,
    # straight-line only.

    _MLPKG_OP = {
        "tessera.matmul": "matmul",
        "tessera.add": "add", "tessera.sub": "sub",
        "tessera.mul": "mul", "tessera.div": "div",
        "tessera.softmax": "softmax",
        "tessera.rmsnorm": "rmsnorm", "tessera.layer_norm": "layer_norm",
        "tessera.silu": "silu", "tessera.relu": "relu",
        "tessera.sigmoid": "sigmoid", "tessera.tanh": "tanh",
        "tessera.gelu": "gelu",
    }

    def _serialize_mlpkg(self):
        """Map the recorded straight-line ops to the author_graph_package form:
        per-arg shapes, an op list (opcode + tensor ids + attrs), and the single
        output tensor id. Tensor ids: args 0.., op j → n_args+j."""
        idof: dict[str, int] = {}
        arg_shapes: list = []
        for k, (ssa, sh) in enumerate(self._args):
            if len(sh) != 2:
                raise TesseraJitError("run_mlpkg args must be rank-2")
            idof[ssa] = k
            arg_shapes.append((int(sh[0]), int(sh[1])))
        n_args = len(self._args)
        ops: list = []
        for j, op in enumerate(self._ops):
            name = self._MLPKG_OP.get(op.op)
            if name is None:
                raise TesseraJitError(
                    f"run_mlpkg cannot express op {op.op!r} (whole-graph lane "
                    "supports matmul/elementwise/softmax/norms/activations)")
            entry: dict = {"op": name, "in0": idof[op.ins[0].ssa]}
            if name == "matmul":
                entry["in1"] = idof[op.ins[1].ssa]
                entry["transpose_a"] = bool(op.meta.get("ta"))
                entry["transpose_b"] = bool(op.meta.get("tb"))
            elif name in ("add", "sub", "mul", "div"):
                entry["in1"] = idof[op.ins[1].ssa]
            elif name == "softmax":
                if op.meta.get("axis") != op.meta.get("rank", 0) - 1:
                    raise TesseraJitError("run_mlpkg softmax is last-axis only")
            elif name in ("rmsnorm", "layer_norm"):
                entry["eps"] = float(op.meta.get("eps", 1e-5))
            idof[op.out.ssa] = n_args + j
            ops.append(entry)
        output_id = idof[self._rets[0].ssa]
        return arg_shapes, ops, output_id, self._rets[0].shape

    def _ensure_mlpkg_pipeline(self, mp):
        if self._mlpkg_pipe is not None:
            return self._mlpkg_pipe, self._mlpkg_out_shape
        import os
        import tempfile
        arg_shapes, ops, output_id, out_shape = self._serialize_mlpkg()
        d = tempfile.mkdtemp(prefix="tessera_mlpkg_")
        pkg = os.path.join(d, "graph.mtlpackage")  # loader needs the extension
        # The package is ALWAYS authored f32: the mlpkg reflection / prepare_tensors
        # path (MTLTensorDataTypeFromMPSDataType) asserts on bf16 bindings today, so
        # bf16 is handled at the Python boundary in run_mlpkg (bf16 in/out, f32
        # package internally — still f32-accumulate). The C `io_bf16=1` boundary
        # capability stays for when bf16 bindings are reflectable.
        if not mp.author_graph_package(pkg, arg_shapes, ops, output_id):
            raise TesseraJitError(
                f"mlpkg author_graph failed (err={mp.last_error_kind()})")
        fn = mp.first_function_name(pkg) or "main"
        pipe = mp.compile_mlpackage(pkg, function_name=fn)
        if pipe is None:
            raise TesseraJitError(
                f"mlpkg compile failed (err={mp.last_error_kind()})")
        if not pipe.prepare_tensors():
            raise TesseraJitError("mlpkg prepare_tensors failed")
        self._mlpkg_dir = d
        self._mlpkg_pipe = pipe
        self._mlpkg_out_shape = tuple(out_shape)
        return pipe, self._mlpkg_out_shape

    def run_mlpkg(self, *arrays: np.ndarray):
        """Compile the WHOLE graph to one MPSGraph package and run it as a SINGLE
        Metal dispatch (vs `run()`'s per-kernel interpreter). apple_gpu, f32 or
        bf16 (bf16 = bf16 boundary, f32 internal compute), single output,
        straight-line only. The package is authored+compiled once and cached;
        later calls just re-fill inputs and dispatch."""
        from tessera import apple_mlpkg as mp

        if self._target != "apple_gpu":
            raise TesseraJitError("run_mlpkg requires target='apple_gpu'")
        if self._elem not in ("f32", "bf16"):
            raise TesseraJitError(f"run_mlpkg supports f32/bf16, not {self._elem!r}")
        if self._has_control_flow:
            raise TesseraJitError("run_mlpkg does not support scf control flow")
        if len(self._rets) != 1:
            raise TesseraJitError("run_mlpkg supports exactly one return value")
        if len(arrays) != len(self._args):
            raise TesseraJitError(
                f"expected {len(self._args)} args, got {len(arrays)}")
        npdt = _ELEM_TO_NP[self._elem]
        is_bf16 = self._elem == "bf16"
        ins = []
        for (ssa, sh), arr in zip(self._args, arrays):
            a = np.ascontiguousarray(np.asarray(arr))
            if a.dtype != npdt or tuple(a.shape) != sh:
                raise TesseraJitError(
                    f"arg dtype/shape mismatch: got {a.dtype}{a.shape}, "
                    f"want {npdt}{sh}")
            # f32 package: upcast bf16 inputs at the boundary (bf16 rounding of the
            # caller's data is preserved; compute is f32; output rounds back).
            ins.append(a.astype(np.float32) if is_bf16 else a)
        pipe, out_shape = self._ensure_mlpkg_pipeline(mp)
        for i, a in enumerate(ins):
            if not pipe.fill_input_at(i, a.tobytes()):
                raise TesseraJitError(f"mlpkg fill_input_at({i}) failed")
        if not pipe.dispatch(timeout_ms=30_000):
            raise TesseraJitError("mlpkg dispatch failed / timed out")
        nbytes = int(np.prod(out_shape)) * 4  # f32 package output
        raw = pipe.read_output_at(len(self._args), nbytes)
        if raw is None:
            raise TesseraJitError("mlpkg read_output failed")
        out = np.frombuffer(raw, dtype=np.float32).reshape(out_shape)
        return out.astype(npdt).copy() if is_bf16 else out.copy()

    def close(self) -> None:
        """Release the cached mlpkg pipeline + its on-disk package (if any)."""
        pipe = self._mlpkg_pipe
        if pipe is not None:
            try:
                pipe.destroy()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._mlpkg_pipe = None
        d = self._mlpkg_dir
        if d:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
            self._mlpkg_dir = None

    def __del__(self):  # best-effort cleanup of the cached package
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def _run_apple_gpu(self, arrays):
        """Interpret the straight-line graph on the bespoke Metal back-half.

        There is no MLIR Metal backend (D2), so instead of ``compile_module`` we
        thread numpy intermediates between ``_apple_gpu_backend`` kernel
        dispatches, after collapsing the canonical fused chains. The compiled CPU
        lane stays the oracle (same graph, ``target="cpu"``)."""
        from tessera import _apple_gpu_backend as agb

        if not self._rets:
            raise TesseraJitError("GraphFn has no return value (call ret())")
        if self._elem not in ("f32", "bf16"):
            raise TesseraJitError(
                f"apple_gpu GraphFn supports f32/bf16, not {self._elem!r}")
        if self._has_control_flow:
            raise TesseraJitError(
                "apple_gpu GraphFn does not support scf control flow yet "
                "(Sprint 3.3 is straight-line tensor algebra)"
            )
        if len(arrays) != len(self._args):
            raise TesseraJitError(f"expected {len(self._args)} args, got {len(arrays)}")
        npdt = _ELEM_TO_NP[self._elem]
        env: dict[str, np.ndarray] = {}
        for (ssa, sh), arr in zip(self._args, arrays):
            a = np.ascontiguousarray(np.asarray(arr))
            if a.dtype != npdt or tuple(a.shape) != sh:
                raise TesseraJitError(
                    f"arg dtype/shape mismatch: got {a.dtype}{a.shape}, "
                    f"want {npdt}{sh}")
            env[ssa] = a
        self._last_dispatch = []
        for node in self._fuse_for_gpu():
            res = self._dispatch_gpu(node, env, agb)
            if node.op == "__qkv_concat__":
                for out_val, arr in zip(node.meta["outs"], res):
                    env[out_val.ssa] = arr  # multi-output: Q, K, V, …
            else:
                env[node.out.ssa] = res
        outs = [env[r.ssa] for r in self._rets]
        return outs[0] if len(outs) == 1 else tuple(outs)

    def _fuse_for_gpu(self) -> list:
        """Collapse fused chains into single synthetic kernels:

        * SwiGLU MLP DAG ``(silu(X@Wg) ⊙ (X@Wu)) @ Wd`` → ``__swiglu__``
        * attention ``softmax(A@B)@C`` → ``__attention__``
        * ``matmul→softmax`` / ``matmul→gelu`` / ``matmul→rmsnorm``

        A group is fused only when every internal intermediate has exactly ONE
        consumer and is not a function result, so fusion can never change an
        observable value — only how many kernels fire. Each group is anchored at
        its earliest member (all external inputs are available by then) and
        emitted in original op order via a single pass."""
        ops = self._ops
        returned = {r.ssa for r in self._rets}
        consumers: dict[str, list] = {}
        producer: dict[str, int] = {}
        for i, op in enumerate(ops):
            producer[op.out.ssa] = i
            for pos, val in enumerate(op.ins):
                consumers.setdefault(val.ssa, []).append((i, pos))

        def lone(ssa):
            cs = consumers.get(ssa, [])
            return cs[0] if len(cs) == 1 and ssa not in returned else None

        def prod(ssa):
            idx = producer.get(ssa)
            return (idx, ops[idx]) if idx is not None else None

        def plain_matmul(op):
            return (
                op.op == "tessera.matmul"
                and not op.meta.get("ta", False)
                and not op.meta.get("tb", False)
            )

        fused_at: dict[int, _GraphOp] = {}
        consumed: set[int] = set()

        # ── Pass 1: SwiGLU DAG, anchored at the elementwise gate-multiply ────
        for i, op in enumerate(ops):
            if op.op != "tessera.mul":
                continue
            c = lone(op.out.ssa)  # mul → down-projection matmul
            if c is None:
                continue
            di, dpos = c
            down = ops[di]
            if not plain_matmul(down) or dpos != 0:
                continue
            br = [prod(v.ssa) for v in op.ins]
            if any(b is None for b in br):
                continue
            (ai, aop), (bi, bop) = br
            if aop.op == "tessera.silu" and bop.op == "tessera.matmul":
                (gi, gop), (ui, uop) = (ai, aop), (bi, bop)
            elif bop.op == "tessera.silu" and aop.op == "tessera.matmul":
                (gi, gop), (ui, uop) = (bi, bop), (ai, aop)
            else:
                continue
            gm = prod(gop.ins[0].ssa)  # silu input ← gate-projection matmul
            if gm is None or not plain_matmul(gm[1]) or not plain_matmul(uop):
                continue
            gmi, gmop = gm
            if gmop.ins[0].ssa != uop.ins[0].ssa:  # gate & up share X
                continue
            if any(lone(s) is None for s in
                   (gmop.out.ssa, gop.out.ssa, uop.out.ssa, op.out.ssa)):
                continue
            members = [gmi, gi, ui, i, di]
            anchor = min(members)
            fused_at[anchor] = _GraphOp(
                "__swiglu__",
                [gmop.ins[0], gmop.ins[1], uop.ins[1], down.ins[1]],
                down.out, {})
            consumed.update(m for m in members if m != anchor)

        # ── Pass 1a: QKV-concat — ≥2 plain matmuls sharing one input X fuse to
        # a single concatenated projection (one matmul over [W0|W1|…]) + column
        # split. If X is a single-use pre-norm of the group, the norm folds in
        # too (one rmsnorm_matmul on the concat weight). It's a multi-output node
        # (`meta["outs"]` lists the per-projection results, written by the
        # executor). No single-use constraint on the outputs — the split values
        # are identical to the separate matmuls, so downstream use is unaffected.
        groups: dict[str, list] = {}
        for i, op in enumerate(ops):
            if i in consumed or i in fused_at:
                continue
            if (op.op == "tessera.matmul"
                    and not op.meta.get("ta", False)
                    and not op.meta.get("tb", False)):
                groups.setdefault(op.ins[0].ssa, []).append(i)
        for x_ssa, members in groups.items():
            if len(members) < 2:
                continue
            anchor = members[0]
            outs = [ops[mi].out for mi in members]
            weights = [ops[mi].ins[1] for mi in members]
            meta = {"splits": [int(o.shape[1]) for o in outs], "outs": outs}
            x_val = ops[anchor].ins[0]
            xp = prod(x_ssa)  # fold a single-use pre-norm of X into the concat
            if (xp is not None and xp[1].op == "tessera.rmsnorm"
                    and x_ssa not in returned
                    and xp[0] not in fused_at and xp[0] not in consumed
                    and len(consumers.get(x_ssa, [])) == len(members)):
                meta["prenorm_eps"] = xp[1].meta.get("eps", 1e-5)
                x_val = xp[1].ins[0]
                consumed.add(xp[0])
            fused_at[anchor] = _GraphOp(
                "__qkv_concat__", [x_val] + weights, ops[anchor].out, meta)
            consumed.update(m for m in members if m != anchor)

        # ── Pass 1b: pre-norm + projection (rmsnorm → matmul) ────────────────
        for i, op in enumerate(ops):
            if i in consumed or i in fused_at or op.op != "tessera.rmsnorm":
                continue
            c = lone(op.out.ssa)  # rmsnorm feeds exactly one plain matmul (op0)
            if c is None or c[0] in consumed or c[0] in fused_at:
                continue
            mi, mpos = c
            mm = ops[mi]
            if not plain_matmul(mm) or mpos != 0:
                continue
            fused_at[i] = _GraphOp(
                "__rmsnorm_matmul__", [op.ins[0], mm.ins[1]], mm.out,
                {"eps": op.meta.get("eps", 1e-5)})
            consumed.add(mi)

        # ── Pass 2: attention / matmul-chain, skipping SwiGLU-claimed ops ────
        for i, op in enumerate(ops):
            if i in consumed or i in fused_at or op.op != "tessera.matmul":
                continue
            c = lone(op.out.ssa)
            if c is None or c[0] in consumed or c[0] in fused_at:
                continue
            ci = c[0]
            cons = ops[ci]
            base = {
                "a_t": op.meta.get("ta", False),
                "b_t": op.meta.get("tb", False),
            }
            is_last_axis_softmax = (
                cons.op == "tessera.softmax"
                and cons.meta.get("axis") == cons.meta.get("rank", 0) - 1
            )
            if is_last_axis_softmax:
                c2 = lone(cons.out.ssa)
                if c2 is not None and c2[0] not in consumed and c2[0] not in fused_at:
                    c2i, c2pos = c2
                    m2 = ops[c2i]
                    if m2.op == "tessera.matmul" and c2pos == 0 \
                            and not m2.meta.get("ta", False):
                        fused_at[i] = _GraphOp(
                            "__attention__",
                            [op.ins[0], op.ins[1], m2.ins[1]],
                            m2.out, {**base, "c_t": m2.meta.get("tb", False)})
                        consumed.update((ci, c2i))
                        continue
                fused_at[i] = _GraphOp(
                    "__matmul_softmax__", [op.ins[0], op.ins[1]], cons.out, base)
                consumed.add(ci)
            elif cons.op == "tessera.gelu":
                fused_at[i] = _GraphOp(
                    "__matmul_gelu__", [op.ins[0], op.ins[1]], cons.out, base)
                consumed.add(ci)
            elif cons.op == "tessera.rmsnorm":
                fused_at[i] = _GraphOp(
                    "__matmul_rmsnorm__", [op.ins[0], op.ins[1]], cons.out,
                    {**base, "eps": cons.meta.get("eps", 1e-5)})
                consumed.add(ci)

        # ── Emit in original order ───────────────────────────────────────────
        plan: list = []
        for i, op in enumerate(ops):
            if i in fused_at:
                plan.append(fused_at[i])
            elif i not in consumed:
                plan.append(op)
        return plan

    def _dispatch_gpu(self, node, env, agb):
        def val(x):
            return env[x.ssa]

        def tr(arr, flag):
            return np.ascontiguousarray(arr.T) if flag else arr

        op, m = node.op, node.meta
        if op == "__qkv_concat__":
            x = val(node.ins[0])
            wcat = np.ascontiguousarray(
                np.concatenate([val(w) for w in node.ins[1:]], axis=1))
            if "prenorm_eps" in m:
                self._last_dispatch.append("qkv_concat_prenorm")
                full = agb.gpu_rmsnorm_matmul(x, wcat, eps=m["prenorm_eps"])
            else:
                self._last_dispatch.append("qkv_concat")
                full = agb.gpu_matmul(x, wcat)
            cols, off = [], 0
            for n in m["splits"]:
                cols.append(np.ascontiguousarray(full[:, off:off + n]))
                off += n
            return cols
        if op == "__rmsnorm_matmul__":
            self._last_dispatch.append("rmsnorm_matmul")
            return agb.gpu_rmsnorm_matmul(
                val(node.ins[0]), val(node.ins[1]), eps=m.get("eps", 1e-5))
        if op == "__swiglu__":
            self._last_dispatch.append("swiglu")
            return agb.gpu_swiglu(
                val(node.ins[0]), val(node.ins[1]),
                val(node.ins[2]), val(node.ins[3]))
        if op == "__attention__":
            self._last_dispatch.append("matmul_softmax_matmul")
            return agb.gpu_attention(
                tr(val(node.ins[0]), m["a_t"]),
                tr(val(node.ins[1]), m["b_t"]),
                tr(val(node.ins[2]), m["c_t"]),
            )
        if op == "__matmul_softmax__":
            self._last_dispatch.append("matmul_softmax")
            return agb.gpu_matmul_softmax(
                tr(val(node.ins[0]), m["a_t"]), tr(val(node.ins[1]), m["b_t"]))
        if op == "__matmul_gelu__":
            self._last_dispatch.append("matmul_gelu")
            return agb.gpu_matmul_gelu(
                tr(val(node.ins[0]), m["a_t"]), tr(val(node.ins[1]), m["b_t"]))
        if op == "__matmul_rmsnorm__":
            self._last_dispatch.append("matmul_rmsnorm")
            return agb.gpu_matmul_rmsnorm(
                tr(val(node.ins[0]), m["a_t"]), tr(val(node.ins[1]), m["b_t"]),
                eps=m.get("eps", 1e-5))
        if op == "tessera.matmul":
            self._last_dispatch.append("matmul")
            return agb.gpu_matmul(
                tr(val(node.ins[0]), m.get("ta", False)),
                tr(val(node.ins[1]), m.get("tb", False)))
        if op == "tessera.softmax":
            if m.get("axis") != m.get("rank", 0) - 1:
                raise TesseraJitError("apple_gpu softmax is last-axis only")
            self._last_dispatch.append("softmax")
            return agb.gpu_softmax(val(node.ins[0]))
        if op == "tessera.gelu":
            self._last_dispatch.append("gelu")
            return agb.gpu_gelu(val(node.ins[0]))
        if op in ("tessera.relu", "tessera.sigmoid", "tessera.tanh", "tessera.silu"):
            kind = op.split(".", 1)[1]
            self._last_dispatch.append(kind)
            return agb.gpu_unary(kind, val(node.ins[0]))
        if op == "tessera.rmsnorm":
            self._last_dispatch.append("rmsnorm")
            return agb.gpu_rmsnorm(val(node.ins[0]), eps=m.get("eps", 1e-5))
        if op == "tessera.layer_norm":
            self._last_dispatch.append("layer_norm")
            return agb.gpu_layer_norm(val(node.ins[0]), eps=m.get("eps", 1e-5))
        if op in ("tessera.add", "tessera.sub", "tessera.mul", "tessera.div"):
            kind = op.split(".", 1)[1]
            self._last_dispatch.append(kind)
            return agb.gpu_binary(kind, val(node.ins[0]), val(node.ins[1]))
        if op == "tessera.transpose":
            self._last_dispatch.append("transpose")
            return np.ascontiguousarray(val(node.ins[0]).T)
        raise TesseraJitError(
            f"apple_gpu GraphFn cannot dispatch op {op!r} (Sprint 3.3 supports "
            "matmul/softmax/norms/activations/elementwise + the fused matmul chains)"
        )


# --------------------------------------------------------------------------- #
# G-C — `@jit`-style bounded-loop front-end
#
# A natural one-call front-end that *traces* a bounded loop body into the
# `tessera.control_for` Graph-IR op and executes it. The body is written against
# the GraphFn build protocol (it calls `g.matmul` / `g.silu` / ... on _Val
# handles), which is the production lane's own authoring surface — so this reuses
# the entire G-A/G-B/G-B.2 machinery end-to-end:
#
#   build_fori_loop(...)  ->  GraphFn carrying a `tessera.control_for`
#   jit_fori_loop(...)    ->  build + execute (apple_gpu: control_for -> tessera-opt
#                             -> tessera_apple.gpu.control_loop -> run_graph_loop;
#                             cpu: scf.for compiled natively).
#
# This is the front-end half of "@jit -> tessera.control_for": the AST `@tessera.
# jit` decorator (canonical-compile lane) does not yet share a tracing protocol
# with the executing GraphFn lane, so wiring the decorator itself is a separate
# canonical-lane bridge. This API delivers the executing path today.
# --------------------------------------------------------------------------- #
def build_fori_loop(
    trip: int,
    body,
    *,
    init_shape,
    const_shapes=(),
    target: str = "apple_gpu",
    name: str = "tessera_fori",
    elem: str = "f32",
) -> "GraphFn":
    """Trace ``for _ in range(trip): carry = body(g, carry, *consts)`` into a
    ``GraphFn`` carrying a ``tessera.control_for`` op (single tensor carry, v1).

    ``body`` is ``(g, carry_val, *const_vals) -> carry_val`` operating on GraphFn
    handles. ``init_shape`` is the carry shape; ``const_shapes`` are the
    loop-invariant operand shapes (passed to ``body`` after the carry, in order).
    Returns the built ``GraphFn`` (not yet executed) so callers can inspect the
    emitted IR (``g._emit_control_for_mlir()``) or pick an execution path.
    """
    g = GraphFn(name=name, elem=elem, target=target)
    carry = g.arg(tuple(init_shape))
    consts = [g.arg(tuple(s)) for s in const_shapes]
    out = g.for_loop(int(trip), init=carry, body=lambda c: body(g, c, *consts))
    g.ret(out)
    return g


def jit_fori_loop(
    trip: int,
    body,
    *,
    init: "np.ndarray",
    consts=(),
    target: str = "apple_gpu",
    via_target_ir: "bool | None" = None,
    name: str = "tessera_fori",
) -> "np.ndarray":
    """Build (``build_fori_loop``) and execute a bounded loop, returning the final
    carry as an ``np.ndarray``.

    ``body`` is ``(g, carry_val, *const_vals) -> carry_val``. ``init`` is the
    initial carry; ``consts`` the loop-invariant arrays (matched to ``body``'s
    trailing params, in order). On ``apple_gpu`` the loop executes through the
    Target-IR path (emit ``tessera.control_for`` -> ``tessera-opt
    --tessera-control-for-to-apple_gpu`` -> ``tessera_apple.gpu.control_loop`` ->
    ``run_graph_loop_f32``) when a built ``tessera-opt`` is available; set
    ``via_target_ir=False`` to force the direct in-memory dispatch instead, or
    ``True`` to require the IR path. On ``cpu`` the ``scf.for`` is compiled
    natively.
    """
    init_arr = np.asarray(init)
    const_arrs = [np.asarray(c) for c in consts]
    # Infer the element type from the carry: bf16 inits build a bf16 GraphFn
    # (host-upcast to f32 for the executor, downcast on the way out — Phase B).
    elem = "bf16" if (_ml_dtypes is not None
                      and init_arr.dtype == _ml_dtypes.bfloat16) else "f32"
    g = build_fori_loop(
        int(trip), body,
        init_shape=init_arr.shape,
        const_shapes=[c.shape for c in const_arrs],
        target=target, name=name, elem=elem,
    )
    arrays = [init_arr, *const_arrs]
    if target == "apple_gpu":
        use_ir = (_find_tessera_opt() is not None) if via_target_ir is None \
            else bool(via_target_ir)
        return g.run_via_target_ir(*arrays) if use_ir else g.run(*arrays)
    return g.run(*arrays)
