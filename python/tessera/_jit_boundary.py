"""Phase 0 production-lane CPU JIT boundary (ctypes → ``libtessera_jit``).

EXPERIMENTAL production-lane plumbing — see ``docs/spec/RUNTIME_ABI_SPEC.md`` §12
and ``docs/spec/PRODUCTION_COMPILER_PLAN.md``. This is **not** "runtime v2".

The whole point of this lane: it executes the MLIR/LLVM compiled function and has
**no numpy fallback by construction**. Any compile/invoke failure raises
:class:`TesseraJitError`. Numerical equality alone does not prove the lane ran —
callers should also assert the JIT invocation counter advanced
(:func:`invocation_count`), which is incremented in C++ only on a real invoke.
"""

from __future__ import annotations

import ctypes
import glob
import os

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
    f32p = ctypes.POINTER(ctypes.c_float)
    lib.tessera_jit_compile.restype = ctypes.c_void_p
    lib.tessera_jit_compile.argtypes = [ctypes.c_char_p]
    lib.tessera_jit_add_2d_f32.restype = ctypes.c_int
    lib.tessera_jit_add_2d_f32.argtypes = [
        ctypes.c_void_p, f32p, f32p, f32p, ctypes.c_int64, ctypes.c_int64
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


def _mlir_for_add(d0: int, d1: int) -> str:
    t = f"tensor<{d0}x{d1}xf32>"
    return (
        f"func.func @tessera_jit_add(%a: {t}, %b: {t}) -> {t} {{\n"
        f"  %0 = tessera.add %a, %b : ({t}, {t}) -> {t}\n"
        f"  return %0 : {t}\n"
        f"}}\n"
    )


def _as_f32p(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def jit_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute ``a + b`` through the MLIR/LLVM production lane (rank-2 f32).

    Destination-passing: Python allocates the output, the compiled function
    writes into it (RUNTIME_ABI_SPEC §12.3). Raises :class:`TesseraJitError` on
    any unsupported input or compile/invoke failure — never falls back to numpy.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    # Phase-0 envelope (ratified guardrail): rank-2 f32 only, equal shapes, no
    # silent dtype coercion — reject so a fallback can never masquerade as success.
    if a.dtype != np.float32 or b.dtype != np.float32:
        raise TesseraJitError(
            f"Phase 0 jit_add is f32-only (got {a.dtype}, {b.dtype})"
        )
    if a.ndim != 2 or a.shape != b.shape:
        raise TesseraJitError(
            f"Phase 0 jit_add requires equal-shape rank-2 arrays "
            f"(got {a.shape}, {b.shape})"
        )
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    lib = _load()
    d0, d1 = int(a.shape[0]), int(a.shape[1])
    handle = lib.tessera_jit_compile(_mlir_for_add(d0, d1).encode("utf-8"))
    if not handle:
        raise TesseraJitError(lib.tessera_jit_last_error().decode("utf-8", "replace"))
    try:
        out = np.empty((d0, d1), dtype=np.float32)
        rc = lib.tessera_jit_add_2d_f32(
            handle, _as_f32p(a), _as_f32p(b), _as_f32p(out), d0, d1
        )
        if rc != 0:
            raise TesseraJitError(
                lib.tessera_jit_last_error().decode("utf-8", "replace")
            )
        return out
    finally:
        lib.tessera_jit_destroy(handle)
