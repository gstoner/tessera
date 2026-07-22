"""Shared lazy Apple GPU runtime loader.

This module is the foundation for closing the compiler/runtime
integration gap (see ``docs/status/ga_ebm.md`` § "Known
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
    "bind_registered",
    "expected_symbols",
    "read_profiling_capabilities",
    "read_dispatch_telemetry",
    "set_dispatch_telemetry_enabled",
    "clear_dispatch_telemetry",
    "APPLE_ABI",
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


# Newest C ABI symbol — used as the staleness sentinel when accepting a
# prebuilt library (env-pointed or CMake-built). If a candidate lacks it,
# it predates the current source and we fall through to a fresh build.
_SENTINEL_SYMBOL = "tessera_apple_gpu_ssm_replay_flush_dev_f32_enc"


def _prebuilt_candidate() -> Optional[ctypes.CDLL]:
    """Return a loaded handle for a prebuilt Apple GPU runtime, if one is
    available and current: ``$TESSERA_APPLE_GPU_RUNTIME_LIB`` first, then a
    CMake-built ``libTesseraAppleRuntime.{dylib,so}`` under ``build/``.

    Centralizing this here (rather than in ``runtime.py``) means there is a
    SINGLE loaded image of the runtime process-wide — both the ctypes /
    ``bind_symbol`` lane and ``runtime.py``'s MPS execution lane share it, so
    the ObjC classes (e.g. ``TesseraMlpkgPipeline``) are never defined twice
    (which previously emitted a duplicate-class warning when both lanes ran)."""
    candidates = []
    env = os.environ.get("TESSERA_APPLE_GPU_RUNTIME_LIB")
    if env:
        candidates.append(Path(env))
    repo_root = Path(__file__).resolve().parent.parent.parent
    backend = repo_root / "build/src/compiler/codegen/Tessera_Apple_Backend"
    candidates.append(backend / "libTesseraAppleRuntime.dylib")
    candidates.append(backend / "libTesseraAppleRuntime.so")
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            lib = ctypes.CDLL(str(cand))
            getattr(lib, _SENTINEL_SYMBOL)  # staleness gate
        except (OSError, AttributeError):
            continue
        return lib
    return None


def _compile_runtime() -> tuple[Optional[ctypes.CDLL], Optional[Path],
                                 Optional[str]]:
    """Compile ``apple_gpu_runtime.mm`` to a dylib + ctypes-load.

    Returns ``(handle, dylib_path, skip_reason)``. On Darwin with a
    working toolchain ``handle`` is non-None and ``skip_reason`` is
    None; otherwise ``handle`` is None and ``skip_reason`` is a short
    diagnostic string.
    """
    # Prefer a prebuilt library (env-pointed or CMake-built) so the whole
    # process shares ONE loaded runtime image. Only when none is current do we
    # compile from source below.
    prebuilt = _prebuilt_candidate()
    if prebuilt is not None:
        return prebuilt, None, None

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


# ── Apple GPU C-ABI signature registry ───────────────────────────────────────
# Single source of truth for the Apple GPU runtime C ABI: every exported symbol
# the Python side binds, mapped to its canonical ``(argtypes, restype)`` ctypes
# signature. Callers bind via :func:`bind_registered` instead of hand-writing a
# signature at each site (which then drifts from the .mm runtime). New entry
# points MUST be added here; two guards enforce it (``test_apple_gpu_abi_registry``):
#   * off-device — every literal ``bind_symbol("sym", ...)`` in the tree must
#     match this registry (catches a stale/duplicated signature on any CI run);
#   * on-device  — when the runtime dylib is loaded, every registry symbol must
#     resolve in it (catches a renamed/removed C export).
APPLE_ABI: dict[str, tuple[tuple, object]] = {
    "tessera_apple_gpu_aligned_buffer_nbytes": ((ctypes.POINTER(ctypes.c_int64), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int64),
    "tessera_apple_gpu_bmm_dev_bf16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_bmm_dev_f16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_bmm_dev_f32_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_cholesky_f32": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_complex_exp_f32": (
        (
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
        ),
        None,
    ),
    "tessera_apple_gpu_complex_mobius_f32": (
        (
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
        ),
        None,
    ),
    "tessera_apple_gpu_complex_mul_f32": (
        (
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
        ),
        None,
    ),
    "tessera_apple_gpu_complex_stereographic_f32": (
        (
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
        ),
        None,
    ),
    "tessera_apple_gpu_commit_and_wait_timeout_probe": ((ctypes.c_uint64,), ctypes.c_int32),
    "tessera_apple_gpu_conv2d_dev_bf16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_conv2d_dev_f16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_conv2d_dev_f32_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_conv2d_f16": ((ctypes.c_void_p,) * 4 + (ctypes.c_int32,) * 14, None),
    "tessera_apple_gpu_conv2d_f32": ((ctypes.c_void_p,) * 4 + (ctypes.c_int32,) * 14, None),
    "tessera_apple_gpu_dylib_load": ((ctypes.c_char_p,), ctypes.c_int32),
    "tessera_apple_gpu_dylib_serialize": ((ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p), ctypes.c_int32),
    "tessera_apple_gpu_family_integer": ((), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_dev_bf16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_dev_f16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_dev_f32_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_f32_status": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_bwd_f32_status": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_bwd_route_f32_status": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_bwd_variant_f32_status": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_bwd_variant_f16_status": ((ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_bwd_variant_bf16_status": ((ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_f16_status": ((ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_variant_f32_status": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_variant_f16_status": ((ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_variant_dev_f32_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_variant_dev_f16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_variant_dev_bf16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_cooperative_dev_f32_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_cooperative_dev_f16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_flash_attn_cooperative_dev_bf16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_layer_norm_dev_bf16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_layer_norm_dev_f16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_layer_norm_dev_f32_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_max_threadgroup_memory_length": ((), ctypes.c_int64),
    "tessera_apple_gpu_metal4_probe": ((ctypes.POINTER(ctypes.c_int32),), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_argument_table_ready": ((ctypes.c_void_p,), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_author_chain": ((ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_author_graph": ((ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_author_matmul": ((ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_author_op": ((ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_binding_count": ((ctypes.c_void_p,), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_binding_info": ((ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_compile": ((ctypes.c_char_p, ctypes.c_char_p), ctypes.c_void_p),
    "tessera_apple_gpu_mlpkg_compile_with_dims": ((ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64)), ctypes.c_void_p),
    "tessera_apple_gpu_mlpkg_destroy": ((ctypes.c_void_p,), None),
    "tessera_apple_gpu_mlpkg_dispatch": ((ctypes.c_void_p, ctypes.c_uint64), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_dtype_raw_for_tag": ((ctypes.c_int32,), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_fill_input": ((ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int64), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_fill_input_at": ((ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int64), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_first_function_name": ((ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_intermediates_heap_size": ((ctypes.c_void_p,), ctypes.c_int64),
    "tessera_apple_gpu_mlpkg_is_compiled": ((ctypes.c_void_p,), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_last_error_kind": ((), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_prepare_tensors": ((ctypes.c_void_p,), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_read_output": ((ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int64), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_read_output_at": ((ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int64), ctypes.c_int32),
    "tessera_apple_gpu_mlpkg_set_aligned_strides": ((ctypes.c_void_p, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_mpsgraph_bf16_supported": ((), ctypes.c_int32),
    "tessera_apple_gpu_mpsgraph_cache_capacity": ((), ctypes.c_int64),
    "tessera_apple_gpu_mpsgraph_cache_evictions": ((), ctypes.c_int64),
    "tessera_apple_gpu_mpsgraph_cache_size": ((), ctypes.c_int32),
    "tessera_apple_gpu_mtl4_archive_enable": ((ctypes.c_char_p,), ctypes.c_int32),
    "tessera_apple_gpu_mtl4_archive_state": ((ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_char_p, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_mtl4_matmul_sg_f32": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_mtl4_matmul2d_f16": ((ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_mtl4_matmul2d_bf16": ((ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_optimizer_f32": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)), ctypes.c_int32),
    "tessera_apple_gpu_mps_matmul_f32": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), None),
    "tessera_apple_gpu_mps_matmul_f16_status": ((ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_mps_matmul_bf16_status": ((ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_mpsgraph_bsmm_f32": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), None),
    "tessera_apple_gpu_mpsgraph_bsmm_f32_status": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_mpsgraph_bsmm_f16_status": ((ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_mpsgraph_gather_f32": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), None),
    "tessera_apple_gpu_mpsgraph_binary_f32": ((ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int64), None),
    "tessera_apple_gpu_mpsgraph_binary_f32_status": ((ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int64), ctypes.c_int32),
    "tessera_apple_gpu_mpsgraph_reduce_f32": ((ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32), None),
    "tessera_apple_gpu_mpsgraph_unary_f32": ((ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int64), None),
    "tessera_apple_gpu_mpsgraph_unary_f32_status": ((ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int64), ctypes.c_int32),
    "tessera_apple_gpu_mpsgraph_unary_f16_status": ((ctypes.c_int32, ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64), ctypes.c_int32),
    "tessera_apple_gpu_topk_f32": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_philox_uniform_f32": ((ctypes.c_uint64, ctypes.c_uint64, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_philox_normal_f32": ((ctypes.c_uint64, ctypes.c_uint64, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_philox_dropout_f32": ((ctypes.POINTER(ctypes.c_float), ctypes.c_uint64, ctypes.c_uint64, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_rmsnorm_dev_bf16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_rmsnorm_dev_f16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_rmsnorm_dev_f32_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_float), ctypes.c_int32),
    "tessera_apple_gpu_ssm_replay_decode_f32": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_ssm_replay_decode_dev_f32_enc": ((ctypes.c_void_p,) * 8 + (ctypes.c_int32,) * 6, ctypes.c_int32),
    "tessera_apple_gpu_ssm_replay_flush_dev_f32_enc": ((ctypes.c_void_p,) * 7 + (ctypes.c_int32,) * 5, ctypes.c_int32),
    "tessera_apple_gpu_ssm_block_decode_f32_status": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_ssm_block_decode_f16_status": ((ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_rope_dev_bf16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_rope_dev_f16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_rope_dev_f32_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_row_major_strides": ((ctypes.POINTER(ctypes.c_int64), ctypes.c_int32, ctypes.POINTER(ctypes.c_int64)), ctypes.c_int32),
    "tessera_apple_gpu_row_major_strides_aligned": ((ctypes.POINTER(ctypes.c_int64), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int64)), ctypes.c_int32),
    "tessera_apple_gpu_run_graph_cond_f16": ((ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.POINTER(ctypes.c_uint16)), ctypes.c_int32),
    "tessera_apple_gpu_run_graph_cond_f32": ((ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.POINTER(ctypes.c_float)), ctypes.c_int32),
    "tessera_apple_gpu_run_graph_loop_f16": ((ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.POINTER(ctypes.c_uint16)), ctypes.c_int32),
    "tessera_apple_gpu_run_graph_loop_f32": ((ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.POINTER(ctypes.c_float)), ctypes.c_int32),
    "tessera_apple_gpu_run_graph_scan_f32": ((ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)), ctypes.c_int32),
    "tessera_apple_gpu_run_graph_while_f16": ((ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.POINTER(ctypes.c_uint16)), ctypes.c_int32),
    "tessera_apple_gpu_run_graph_while_f32": ((ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.POINTER(ctypes.c_float)), ctypes.c_int32),
    "tessera_apple_gpu_scatter_f32": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int64), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_sddmm_f32": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_session_commit_count": ((), ctypes.c_int64),
    "tessera_apple_gpu_softmax_dev_bf16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_softmax_dev_f16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_softmax_dev_f32_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_solve_cholesky_f32": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_solve_lu_f32": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_spmm_csr_f32": ((ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_tri_solve_f32": ((ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_tile_simdgroup_gemm_f16": ((ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_tile_simdgroup_gemm_bf16": ((ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_tile_last_device_time_ns": ((), ctypes.c_int64),
    "tessera_apple_gpu_tile_last_counter_delta": ((), ctypes.c_int64),
    "tessera_apple_gpu_tile_counter_sampling_supported": ((), ctypes.c_int32),
    "tessera_apple_gpu_dispatch_telemetry_set_enabled": ((ctypes.c_int32,), None),
    "tessera_apple_gpu_dispatch_telemetry_enabled": ((), ctypes.c_int32),
    "tessera_apple_gpu_dispatch_telemetry_clear": ((), None),
    "tessera_apple_gpu_last_dispatch_device_time_ns": ((), ctypes.c_int64),
    "tessera_apple_gpu_last_dispatch_counter_delta": ((), ctypes.c_int64),
    "tessera_apple_gpu_last_dispatch_counter_supported": ((), ctypes.c_int32),
    "tessera_apple_gpu_last_dispatch_timing_source": ((), ctypes.c_int32),
    "tessera_apple_gpu_last_dispatch_resource_record": ((ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64)), ctypes.c_int32),
    "tessera_apple_gpu_profiling_capabilities": ((), ctypes.c_int32),
    "tessera_apple_gpu_unary_dev_bf16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_unary_dev_f16_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32), ctypes.c_int32),
    "tessera_apple_gpu_unary_dev_f32_enc": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32), ctypes.c_int32),
    "ts_dev_alloc": ((ctypes.c_int64,), ctypes.c_void_p),
    "ts_dev_download": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64), None),
    "ts_dev_free": ((ctypes.c_void_p,), None),
    "ts_dev_nbytes": ((ctypes.c_void_p,), ctypes.c_int64),
    "ts_dev_upload": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64), None),
    "ts_dev_upload_at": ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64), None),
    "ts_enc_begin": ((), ctypes.c_void_p),
    "ts_enc_commit_async": ((ctypes.c_void_p,), ctypes.c_int32),
    "ts_enc_commit_wait": ((ctypes.c_void_p,), None),
    "ts_enc_wait_destroy": ((ctypes.c_void_p,), ctypes.c_int32),
}


def bind_registered(symbol: str) -> Optional[Callable]:
    """Bind an Apple GPU C-ABI ``symbol`` using its canonical signature from
    :data:`APPLE_ABI` — the single source of truth for the ABI. Prefer this
    over :func:`bind_symbol` so call sites never hand-write ctypes signatures.

    Returns the bound function, or ``None`` when the runtime isn't available
    (same contract as :func:`bind_symbol`).

    Raises ``KeyError`` if ``symbol`` is not registered — a new C-ABI entry
    point must be added to :data:`APPLE_ABI` (the drift guard enforces this).
    """
    argtypes, restype = APPLE_ABI[symbol]
    return bind_symbol(symbol, argtypes, restype)


def expected_symbols() -> tuple[str, ...]:
    """All Apple GPU C-ABI symbols the registry declares — consumed by the
    off-device drift guard and the on-device dylib-export check."""
    return tuple(APPLE_ABI)


_DISPATCH_TIMING_SOURCES = {
    0: None,
    1: "metal_kernel_interval",
    2: "metal_command_buffer_interval",
    3: "metal4_timestamp_heap",
    # MPSGraph can rotate its root command buffer during commitAndContinue.
    # This timestamp pair is event-ordered around graph completion, so it is a
    # whole-graph GPU envelope rather than a single command-buffer interval.
    4: "metal4_mpsgraph_envelope",
}

_PROFILING_CAPABILITY_BITS = (
    (1 << 0, "pipeline_limits"),
    (1 << 1, "timestamp_counter_set"),
    (1 << 2, "statistic_counter_set"),
    (1 << 3, "stage_utilization_counter_set"),
    (1 << 4, "dispatch_boundary_sampling"),
    (1 << 5, "stage_boundary_sampling"),
    (1 << 6, "metal4_timestamp_heap"),
)


def read_profiling_capabilities() -> dict[str, object]:
    """Return exact-device public Metal profiling capabilities.

    Register count, scratch bytes, spill count, and true occupancy are named
    explicitly because the public Metal API does not expose them. They remain
    false even when pipeline-limit or timestamp evidence is available.
    """
    probe = bind_registered("tessera_apple_gpu_profiling_capabilities")
    raw = int(probe()) if probe is not None else 0
    decoded = {name: bool(raw & bit) for bit, name in _PROFILING_CAPABILITY_BITS}
    decoded.update({
        "register_count": False,
        "scratch_bytes": False,
        "spill_count": False,
        "occupancy": False,
    })
    return {"raw": raw, "capabilities": decoded}


def set_dispatch_telemetry_enabled(enabled: bool) -> bool:
    """Enable the opt-in per-dispatch timing/counter record.

    Metal 4 timestamp heaps use precise encoder-boundary samples and therefore
    carry measurable overhead. They stay disabled for normal production calls;
    benchmark/proof lanes opt in explicitly.
    """
    setter = bind_registered("tessera_apple_gpu_dispatch_telemetry_set_enabled")
    probe = bind_registered("tessera_apple_gpu_dispatch_telemetry_enabled")
    if setter is None or probe is None:
        return False
    setter(ctypes.c_int32(1 if enabled else 0))
    return bool(probe()) == bool(enabled)


def clear_dispatch_telemetry() -> None:
    clear = bind_registered("tessera_apple_gpu_dispatch_telemetry_clear")
    if clear is not None:
        clear()


def read_dispatch_telemetry() -> dict[str, object]:
    """Read the current thread's last native dispatch record.

    Missing values are returned as ``None``. In particular, a queue-only
    MPSGraph call does not inherit a prior command buffer's timing after the
    caller clears the record.
    """
    enabled = bind_registered("tessera_apple_gpu_dispatch_telemetry_enabled")
    timing = bind_registered("tessera_apple_gpu_last_dispatch_device_time_ns")
    counter = bind_registered("tessera_apple_gpu_last_dispatch_counter_delta")
    counter_supported = bind_registered(
        "tessera_apple_gpu_last_dispatch_counter_supported")
    source = bind_registered("tessera_apple_gpu_last_dispatch_timing_source")
    resource = bind_registered("tessera_apple_gpu_last_dispatch_resource_record")
    if (
        enabled is None
        or timing is None
        or counter is None
        or counter_supported is None
        or source is None
        or resource is None
    ):
        return {
            "capture_enabled": False,
            "device_time_ns": None,
            "timing_source": None,
            "counter_sampling_supported": None,
            "counter_timestamp_delta": None,
            "resources": None,
        }
    device_time_ns = int(timing())
    counter_delta = int(counter())
    counter_state = int(counter_supported())
    source_id = int(source())
    tpg_x = ctypes.c_int32(-1)
    tpg_y = ctypes.c_int32(-1)
    tpg_z = ctypes.c_int32(-1)
    execution_width = ctypes.c_int32(-1)
    max_threads = ctypes.c_int32(-1)
    static_tg_memory = ctypes.c_int64(-1)
    has_resources = bool(resource(
        ctypes.byref(tpg_x), ctypes.byref(tpg_y), ctypes.byref(tpg_z),
        ctypes.byref(execution_width), ctypes.byref(max_threads),
        ctypes.byref(static_tg_memory)))
    resources = None
    if has_resources:
        total_threads = tpg_x.value * tpg_y.value * tpg_z.value
        width = execution_width.value
        capacity = max_threads.value
        resources = {
            "threadgroup": [tpg_x.value, tpg_y.value, tpg_z.value],
            "thread_execution_width": width,
            "max_total_threads_per_threadgroup": capacity,
            "static_threadgroup_memory_bytes": static_tg_memory.value,
            # The ABI field predates dynamic threadgroup allocations. Current
            # runtimes report pipeline-static plus encoder-requested memory;
            # keep the legacy key and expose the accurate schema name too.
            "threadgroup_memory_bytes": static_tg_memory.value,
            "simdgroups_per_threadgroup": (
                (total_threads + width - 1) // width if width > 0 else None),
            "threadgroup_capacity_fraction": (
                total_threads / capacity if capacity > 0 else None),
            "occupancy": None,
            "register_count": None,
            "scratch_bytes": None,
            "spill_count": None,
        }
    return {
        "capture_enabled": bool(enabled()),
        "device_time_ns": device_time_ns if device_time_ns >= 0 else None,
        "timing_source": _DISPATCH_TIMING_SOURCES.get(source_id),
        "counter_sampling_supported": (
            bool(counter_state) if counter_state >= 0 else None),
        "counter_timestamp_delta": counter_delta if counter_delta >= 0 else None,
        "resources": resources,
    }



# Apple-sample Actions 1 + 6 (2026-05-31) — Capability snapshot helpers
# that decode the runtime's caps bitmask + archive state into a
# structured dict. Exposed so ``CompileResult.to_dict()`` (and any
# dashboard / telemetry consumer) can report "what's *really* lit up on
# this host" instead of just ``target=apple_gpu``.
#
# Capability bits match the ``tessera_apple_gpu_metal4_probe`` doc
# comment in apple_gpu_runtime.mm:
#   1  MTL4CommandQueue
#   2  MTL4CommandAllocator
#   4  MTL4Compiler
#   8  MTLTensor
#  16  MSL 4.0 library compile

_APPLE_GPU_CAP_BITS: tuple[tuple[int, str], ...] = (
    (1, "mtl4_command_queue"),
    (2, "mtl4_command_allocator"),
    (4, "mtl4_compiler"),
    (8, "mtl_tensor"),
    (16, "msl_4_0"),
)


def apple_gpu_capabilities_snapshot() -> dict[str, object]:
    """Return a structured snapshot of Apple GPU capabilities + archive
    cache state.

    Always returns a dict (never ``None``); the dict carries enough
    keys to be useful even when the runtime isn't available — the
    caller doesn't need to handle a missing snapshot specially.

    Keys:

    * ``runtime_available`` (bool) — the runtime dylib loaded.
    * ``capabilities`` (dict[str, bool]) — per-feature flags decoded
      from the C ABI capability bitmask. Always present when
      ``runtime_available``. Empty dict otherwise.
    * ``capabilities_raw`` (int) — the bitmask as returned by the C
      probe; useful for telemetry that doesn't want to enumerate the
      decoded keys.
    * ``mtl4_full`` (bool) — every cap bit is set (i.e. the FULL MTL4
      stack is usable; matches the rc=1 case of the C probe).
    * ``archive`` (dict) — MTL4Archive cache state:
       ``available`` (bool) — the archive-state probe returned
       success;  ``enabled`` (bool) — archive capture is enabled
       (compiler has a CaptureBinaries serializer attached);
       ``has_lookup`` (bool) — a prior archive was loaded as a lookup
       archive on the current compiler; ``path`` (str) — the archive
       file path (empty when unset).

    The return shape is stable — adding new bits only adds keys,
    never renames or removes.
    """
    snapshot: dict[str, object] = {
        "runtime_available": False,
        "capabilities": {},
        "capabilities_raw": 0,
        "mtl4_full": False,
        "archive": {
            "available": False,
            "enabled": False,
            "has_lookup": False,
            "path": "",
        },
    }
    handle = apple_gpu_runtime()
    if handle is None:
        return snapshot
    snapshot["runtime_available"] = True

    # Capability bitmask.
    probe = bind_symbol(
        "tessera_apple_gpu_metal4_probe",
        (ctypes.POINTER(ctypes.c_int32),),
        restype=ctypes.c_int32,
    )
    caps_raw = 0
    full = False
    if probe is not None:
        caps_out = ctypes.c_int32(0)
        full = bool(probe(ctypes.byref(caps_out)))
        caps_raw = int(caps_out.value)
    decoded = {name: bool(caps_raw & bit) for bit, name in _APPLE_GPU_CAP_BITS}
    snapshot["capabilities"] = decoded
    snapshot["capabilities_raw"] = caps_raw
    snapshot["mtl4_full"] = full

    # Archive state.
    archive_probe = bind_symbol(
        "tessera_apple_gpu_mtl4_archive_state",
        (ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_int32),
         ctypes.c_char_p,
         ctypes.c_int32),
        restype=ctypes.c_int32,
    )
    if archive_probe is not None:
        enabled = ctypes.c_int32(0)
        has_lookup = ctypes.c_int32(0)
        path_buf = ctypes.create_string_buffer(1024)
        rc = archive_probe(
            ctypes.byref(enabled),
            ctypes.byref(has_lookup),
            path_buf,
            ctypes.c_int32(1024),
        )
        snapshot["archive"] = {
            "available": bool(rc),
            "enabled": bool(enabled.value),
            "has_lookup": bool(has_lookup.value),
            "path": path_buf.value.decode("utf-8", errors="replace"),
        }
    return snapshot


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
