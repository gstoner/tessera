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


# Newest C ABI symbol — used as the staleness sentinel when accepting a
# prebuilt library (env-pointed or CMake-built). If a candidate lacks it,
# it predates the current source and we fall through to a fresh build.
_SENTINEL_SYMBOL = "tessera_apple_gpu_ppo_policy_loss_ex_f32"


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
