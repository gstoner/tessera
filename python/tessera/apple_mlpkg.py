"""Apple ``.mtlpackage`` packaged-kernel loader (PK1 of the
packaged-kernel sprint).

A Metal package (``.mtlpackage``) is the build output of Core ML Tools
/ Xcode — a directory containing a compiled MPSGraphPackage + manifest.
Tessera consumes one directly: load → compile a
``MTL4MachineLearningPipelineState`` (with reflection enabled) →
return an opaque Python handle that subsequent sprint steps (PK2-PK4)
extend with binding extraction, tensor creation, and dispatch.

PK1 scope: load + compile + lifecycle. NO execution yet — calling
``Pipeline.dispatch(...)`` raises ``NotImplementedError`` until PK4
lands.

Usage::

    from tessera.apple_mlpkg import compile_mlpackage

    pipe = compile_mlpackage(
        "/path/to/matrix-multiplication.mtlpackage",
        function_name="main",
    )
    if pipe is None:
        # macOS < 26 / non-Darwin / corrupt package / compile failure.
        # Read ``last_error_kind()`` for the diagnostic enum.
        ...
    with pipe:
        # PK2-PK4 surface lands here (bindings / dispatch).
        assert pipe.is_compiled

The handle is a context manager — exiting the ``with`` block calls
``destroy()`` so the underlying ``MTLLibrary`` + pipeline state are
ARC-released cleanly.

Skip semantics on non-Apple hosts: ``compile_mlpackage`` returns
``None`` with ``last_error_kind() == -1``. Tests should treat that
return as a graceful skip.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional

from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol


# Error enum returned by ``tessera_apple_gpu_mlpkg_last_error_kind()``.
# Mirrors the runtime's ``g_mlpkg_last_error_kind`` semantics.
ERROR_NONE = 0
ERROR_OS_UNAVAILABLE = -1
ERROR_LIBRARY_LOAD_FAILED = -2
ERROR_PIPELINE_COMPILE_FAILED = -3


def _bind_compile():
    return bind_symbol(
        "tessera_apple_gpu_mlpkg_compile",
        (ctypes.c_char_p, ctypes.c_char_p),
        restype=ctypes.c_void_p,
    )


def _bind_destroy():
    return bind_symbol(
        "tessera_apple_gpu_mlpkg_destroy",
        (ctypes.c_void_p,),
        restype=None,
    )


def _bind_is_compiled():
    return bind_symbol(
        "tessera_apple_gpu_mlpkg_is_compiled",
        (ctypes.c_void_p,),
        restype=ctypes.c_int32,
    )


def _bind_last_error_kind():
    return bind_symbol(
        "tessera_apple_gpu_mlpkg_last_error_kind",
        (),
        restype=ctypes.c_int32,
    )


def last_error_kind() -> int:
    """Return the most recent compile error code (and clear it).

    Maps to the ``ERROR_*`` constants above. Returns
    ``ERROR_OS_UNAVAILABLE`` when the runtime isn't loaded (so callers
    that probe ``last_error_kind`` to distinguish "no error happened
    yet" vs "we couldn't even reach the runtime" get the latter
    answer); otherwise returns the C-side value (0 = no error).
    """
    # Check the runtime probe through this module's namespace so test
    # monkey-patches reach the right symbol. ``bind_symbol`` consults
    # its own module's probe; we additionally pre-check here.
    if apple_gpu_runtime() is None:
        return ERROR_OS_UNAVAILABLE
    fn = _bind_last_error_kind()
    if fn is None:
        return ERROR_OS_UNAVAILABLE
    return int(fn())


class Pipeline:
    """Opaque handle wrapping a compiled ``MTL4MachineLearningPipelineState``.

    Constructed via :func:`compile_mlpackage`. Holds the underlying C
    ABI ``void*`` until ``destroy()`` (or context-manager exit) is
    called. After destruction, ``is_compiled`` returns ``False`` and
    subsequent method calls raise ``RuntimeError``.

    PK1 surface: lifecycle + ``is_compiled`` probe. PK2-PK4 will add
    ``bindings()`` / ``dispatch(...)`` / etc.
    """

    __slots__ = ("_handle", "_package_path", "_function_name")

    def __init__(self, handle: int, package_path: str, function_name: str):
        self._handle = int(handle)
        self._package_path = package_path
        self._function_name = function_name

    @property
    def is_compiled(self) -> bool:
        if not self._handle:
            return False
        fn = _bind_is_compiled()
        if fn is None:
            return False
        return bool(fn(ctypes.c_void_p(self._handle)))

    @property
    def package_path(self) -> str:
        return self._package_path

    @property
    def function_name(self) -> str:
        return self._function_name

    def dispatch(self, *args, **kwargs):
        """Reserved for PK4. Raises until end-to-end dispatch lands."""
        raise NotImplementedError(
            "dispatch() lands in PK4 of the packaged-kernel sprint; PK1 "
            "only exposes the load + compile foundation")

    def destroy(self) -> None:
        if self._handle:
            fn = _bind_destroy()
            if fn is not None:
                fn(ctypes.c_void_p(self._handle))
            self._handle = 0

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *_exc) -> None:
        self.destroy()

    def __del__(self):
        # Defensive: if the user forgot to close, release on GC. Safe
        # because destroy() is idempotent (no-op when _handle == 0).
        try:
            self.destroy()
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "compiled" if self.is_compiled else "destroyed"
        return (f"Pipeline(package={self._package_path!r}, "
                f"function={self._function_name!r}, state={state})")


def compile_mlpackage(
    path: str | Path,
    *,
    function_name: str = "main",
) -> Optional[Pipeline]:
    """Load ``path`` as a Metal package, compile the named function as
    a ``MTL4MachineLearningPipelineState`` with reflection enabled,
    and return a :class:`Pipeline` handle.

    Returns ``None`` if any step failed (OS unavailable, package load
    failed, pipeline compile failed). Call :func:`last_error_kind` to
    distinguish the cause.

    The Apple runtime is JIT-built on first use; on non-Darwin hosts
    this function returns ``None`` with
    ``last_error_kind() == ERROR_OS_UNAVAILABLE``.
    """
    if apple_gpu_runtime() is None:
        return None
    fn = _bind_compile()
    if fn is None:
        return None
    path_str = str(path)
    handle = fn(path_str.encode("utf-8"), function_name.encode("utf-8"))
    # ctypes ``c_void_p`` represents a NULL return as ``None`` OR ``0``
    # depending on platform — normalize.
    if not handle:
        return None
    return Pipeline(handle, package_path=path_str,
                    function_name=function_name)


__all__ = [
    "ERROR_NONE",
    "ERROR_OS_UNAVAILABLE",
    "ERROR_LIBRARY_LOAD_FAILED",
    "ERROR_PIPELINE_COMPILE_FAILED",
    "Pipeline",
    "compile_mlpackage",
    "last_error_kind",
]
