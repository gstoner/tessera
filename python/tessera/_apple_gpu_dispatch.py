"""Shared lazy Apple GPU runtime loader.

This module is the foundation for closing the compiler/runtime
integration gap (see ``docs/status/ga_ebm_milestone.md`` § "Known
non-claims" #1). Before this module landed, every caller that wanted
the Apple GPU MSL kernels had to compile the runtime dylib themselves
via ``subprocess`` + ``clang++`` and bind ctypes signatures by hand —
the benchmark + tests both carry that scaffolding. With the loader
in place, the Python frontends (``tessera.ga.*`` / ``tessera.ebm.*``)
can route through ``apple_gpu_dispatch_*`` helpers that hide all of
that behind a stable API.

Lifecycle:

  1. **First call** to ``apple_gpu_runtime()`` triggers a one-time
     compile of ``apple_gpu_runtime.mm`` into a process-temp dylib
     and a ctypes load. Result cached in a module-global.
  2. **Subsequent calls** return the cached ``CDLL`` instance — no
     recompile, no recheck.
  3. **Non-Darwin / missing toolchain / compile failure** all return
     ``None``; callers are responsible for the fallback path
     (typically the existing Python reference).
  4. **Symbol binding** is also cached per ``(symbol, argtype_key)``
     so repeated dispatches don't pay the ``getattr`` + ``argtypes``
     assignment cost more than once.

The runtime sources are looked up relative to the package install
location first, then the repo source tree (so this works in both
``pip install -e .`` and "run from a clone" setups).
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional

__all__ = [
    "apple_gpu_runtime",
    "apple_gpu_available",
    "apple_gpu_skip_reason",
    "apple_gpu_runtime_handle",
    "bind_symbol",
]


# Module-global cache. Initialized lazily under `_lock` to be safe across
# threads (the test suite uses parametrized fixtures that can race here).
_lock = threading.Lock()
_handle: Optional[ctypes.CDLL] = None
_skip_reason: Optional[str] = None
_loaded: bool = False
_dylib_path: Optional[Path] = None
# Per-(symbol, argtypes-tuple, restype) binding cache.
_symbol_cache: dict[tuple, Callable] = {}


def _is_darwin() -> bool:
    return sys.platform == "darwin"


def _find_runtime_source() -> Optional[Path]:
    """Locate ``apple_gpu_runtime.mm``. Prefers the path adjacent to
    this module (when shipped in a wheel), falls back to the repo
    source tree (for dev installs)."""
    # Heuristic 1: package-relative (when copied into the wheel).
    pkg_root = Path(__file__).resolve().parent
    in_package = pkg_root / "runtime" / "apple_gpu_runtime.mm"
    if in_package.exists():
        return in_package
    # Heuristic 2: repo source tree. ``python/tessera/`` → repo root.
    repo_root = pkg_root.parent.parent
    in_repo = (repo_root /
               "src/compiler/codegen/Tessera_Apple_Backend/"
               "runtime/apple_gpu_runtime.mm")
    if in_repo.exists():
        return in_repo
    return None


def _compile_runtime() -> tuple[Optional[ctypes.CDLL], Optional[Path],
                                 Optional[str]]:
    """Compile ``apple_gpu_runtime.mm`` to a dylib + ctypes-load.

    Returns ``(handle, dylib_path, skip_reason)``. On Darwin with a
    working toolchain ``handle`` is non-None and ``skip_reason`` is
    None; otherwise ``handle`` is None and ``skip_reason`` is a short
    diagnostic string.
    """
    if not _is_darwin():
        return None, None, f"non-darwin host (sys.platform={sys.platform!r})"
    cxx = shutil.which("clang++") or shutil.which("c++")
    if cxx is None:
        return None, None, "clang++/c++ not found on PATH"
    source = _find_runtime_source()
    if source is None:
        return None, None, "apple_gpu_runtime.mm not found in package or repo"

    # Stable per-user cache directory — avoids recompiling on every
    # process start when the source hasn't changed. Bust the cache
    # via the source mtime so dev-install edits pick up.
    cache_root = Path(os.environ.get("TESSERA_APPLE_GPU_CACHE",
                                      tempfile.gettempdir())).expanduser()
    cache_dir = cache_root / "tessera_apple_gpu_runtime"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(source.stat().st_mtime)
    lib = cache_dir / f"libtessera_apple_gpu_runtime.{stamp}.dylib"
    if not lib.exists():
        # Clean older stamps so the cache doesn't grow unboundedly.
        for old in cache_dir.glob("libtessera_apple_gpu_runtime.*.dylib"):
            try:
                old.unlink()
            except OSError:
                pass
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", "-O2", "-fobjc-arc",
               "-x", "objective-c++", str(source), "-o", str(lib),
               "-framework", "Metal",
               "-framework", "MetalPerformanceShaders",
               # 2026-05-29: MPSGraph-backed Tier-1 / long-tail execution lane.
               "-framework", "MetalPerformanceShadersGraph",
               "-framework", "Foundation"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return None, None, (
                f"clang++ failed (rc={proc.returncode}): "
                f"{proc.stderr[-400:]}"
            )
    try:
        handle = ctypes.CDLL(str(lib))
    except OSError as exc:
        return None, None, f"ctypes failed to load {lib}: {exc}"
    return handle, lib, None


def apple_gpu_runtime() -> Optional[ctypes.CDLL]:
    """Return the cached Apple GPU runtime handle, compiling on first
    call. ``None`` when unavailable — check :func:`apple_gpu_skip_reason`
    for the reason."""
    global _handle, _skip_reason, _loaded, _dylib_path
    if _loaded:
        return _handle
    with _lock:
        if _loaded:
            return _handle  # type: ignore[unreachable]
        _handle, _dylib_path, _skip_reason = _compile_runtime()
        _loaded = True
    return _handle


def apple_gpu_available() -> bool:
    """``True`` iff the runtime compiled + loaded successfully."""
    return apple_gpu_runtime() is not None


def apple_gpu_skip_reason() -> Optional[str]:
    """Return the diagnostic string for why the runtime is unavailable,
    or ``None`` when it loaded successfully."""
    apple_gpu_runtime()  # trigger lazy load
    return _skip_reason


def apple_gpu_runtime_handle() -> tuple[Optional[ctypes.CDLL], Optional[Path],
                                          Optional[str]]:
    """``(handle, dylib_path, skip_reason)`` — full status tuple for
    callers that want to log / report the dylib location."""
    apple_gpu_runtime()
    return _handle, _dylib_path, _skip_reason


def bind_symbol(
    symbol: str,
    argtypes: tuple,
    restype: object = None,
) -> Optional[Callable]:
    """Bind a runtime symbol with ``argtypes`` + ``restype`` and cache
    the result. Returns the bound function or ``None`` if the runtime
    isn't available.

    The cache key includes ``argtypes`` + ``restype`` so two callers
    asking for the same symbol with conflicting signatures get
    independent bindings.  This is implemented by constructing a
    fresh function pointer per signature via
    :func:`ctypes.CFUNCTYPE`, which gives each caller its own
    ``argtypes``/``restype`` slots — earlier versions reused the
    ``getattr(CDLL, symbol)`` cached ``_FuncPtr`` and mutated its
    attributes, which silently aliased the signature across all
    callers (2026-05-22 fix).

    All work runs under :data:`_lock` so concurrent first-binders of
    the same ``(symbol, argtypes, restype)`` triple agree on a single
    cached pointer.
    """
    handle = apple_gpu_runtime()
    if handle is None:
        return None
    cache_key = (symbol, argtypes, restype)
    cached = _symbol_cache.get(cache_key)
    if cached is not None:
        return cached
    with _lock:
        # Re-check after acquiring the lock in case a concurrent
        # caller populated the slot while we were waiting.
        cached = _symbol_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            # Verify the symbol exists in the dylib (resolves
            # immediately on most platforms).  We do not keep the
            # returned _FuncPtr — its mutable .argtypes/.restype are
            # the source of the cross-signature aliasing bug.
            getattr(handle, symbol)
        except AttributeError:
            return None
        # Build a true-independent function prototype.  Each call to
        # CFUNCTYPE creates a fresh class with its own argtypes/restype;
        # binding it via the (symbol, handle) tuple resolves the symbol
        # against the dylib without touching the CDLL's _FuncPtr cache.
        # The ``restype`` parameter is typed ``object`` for ergonomic
        # call sites (callers pass ctypes types or ``None``); the
        # cast below narrows it back to what CFUNCTYPE actually expects.
        proto = ctypes.CFUNCTYPE(restype, *argtypes)  # type: ignore[arg-type]
        fn = proto((symbol, handle))
        _symbol_cache[cache_key] = fn
        return fn


# Convenience: test helpers and benchmark code occasionally need to
# clear the cache (e.g., to force a recompile after editing the source).
# This is intentionally NOT exported in ``__all__`` — it's an escape
# hatch, not a stable API.
def _reset_for_testing() -> None:
    global _handle, _skip_reason, _loaded, _dylib_path
    with _lock:
        _handle = None
        _skip_reason = None
        _loaded = False
        _dylib_path = None
        _symbol_cache.clear()
