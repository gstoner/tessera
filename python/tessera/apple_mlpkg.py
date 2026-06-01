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
from dataclasses import dataclass
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


# ---- PK2: reflection-extraction wrappers --------------------------------

# Raw MTLTensorDataType values learned from the host runtime via the
# `tessera_apple_gpu_mlpkg_dtype_raw_for_tag` probe. We cache them so
# `TensorBinding.dtype_name` can decode without re-probing per call.
# Sentinel tags (the keys) are stable integer codes chosen by Tessera —
# they map to Apple's runtime enum values (which may shift across
# SDKs). Use ``-1`` for "unavailable" / "unknown".
_DTYPE_TAG_BY_NAME = {
    "fp32":   32,
    "fp16":   16,
    "bf16":   22,
    "int8":   8,
    "uint8":  80,
    "int16":  808,
    "uint16": 800,
    "int32":  132,
    "uint32": 232,
}

_dtype_name_by_raw: Optional[dict[int, str]] = None


def _dtype_name_for_raw(raw: int) -> str:
    """Decode an MTLTensorDataType raw enum value to a Tessera canonical
    dtype name (``"fp32"`` / ``"fp16"`` / ``"bf16"`` / ...). Falls back
    to ``"raw=<N>"`` for codes the host SDK uses but we haven't named."""
    global _dtype_name_by_raw
    if _dtype_name_by_raw is None:
        probe = bind_symbol(
            "tessera_apple_gpu_mlpkg_dtype_raw_for_tag",
            (ctypes.c_int32,),
            restype=ctypes.c_int32,
        )
        _dtype_name_by_raw = {}
        if probe is not None:
            for name, tag in _DTYPE_TAG_BY_NAME.items():
                r = int(probe(tag))
                if r != -1:
                    _dtype_name_by_raw[r] = name
    return _dtype_name_by_raw.get(raw, f"raw={raw}")


@dataclass(frozen=True)
class TensorBinding:
    """One reflection-extracted tensor binding from a packaged ML
    pipeline. Mirrors Apple's `MTLTensorBinding` protocol.

    Fields:

    * ``name`` — the binding name as declared in the Metal package
      (e.g., ``"inputA"``, ``"output"``).
    * ``buffer_index`` — the argument-table slot the kernel reads from.
      This is the value to pass to
      ``[argumentTable setResource:atBufferIndex:]`` (Apple-sample
      Pattern 2). Distinct from the binding's enumeration order.
    * ``rank`` — number of dimensions.
    * ``dims`` — extents innermost-first; ``-1`` indicates a dynamic
      dimension (sentinel from Apple's MTLTensorExtents).
    * ``dtype`` — Tessera canonical dtype name (``"fp32"`` etc.) or
      ``"raw=<N>"`` for SDK enum values we haven't named.
    * ``dtype_raw`` — the Apple ``MTLTensorDataType`` raw enum value.
    """
    name: str
    buffer_index: int
    rank: int
    dims: tuple[int, ...]
    dtype: str
    dtype_raw: int


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

    def bindings(self) -> dict[str, "TensorBinding"]:
        """PK2 — Return the reflection-extracted tensor bindings as a
        ``dict[name → TensorBinding]``.

        Raises ``RuntimeError`` if the pipeline isn't compiled (e.g.,
        already destroyed). Returns an empty dict if the underlying
        pipeline has no tensor bindings (unusual but legal).
        """
        if not self._handle:
            raise RuntimeError("Pipeline already destroyed")
        count_fn = bind_symbol(
            "tessera_apple_gpu_mlpkg_binding_count",
            (ctypes.c_void_p,), restype=ctypes.c_int32)
        info_fn = bind_symbol(
            "tessera_apple_gpu_mlpkg_binding_info",
            (ctypes.c_void_p, ctypes.c_int32,
             ctypes.c_char_p, ctypes.c_int32,
             ctypes.POINTER(ctypes.c_int32),
             ctypes.POINTER(ctypes.c_int32),
             ctypes.POINTER(ctypes.c_int64), ctypes.c_int32,
             ctypes.POINTER(ctypes.c_int32)),
            restype=ctypes.c_int32)
        if count_fn is None or info_fn is None:
            raise RuntimeError(
                "PK2 reflection symbols missing — runtime needs rebuild")
        n = int(count_fn(ctypes.c_void_p(self._handle)))
        if n < 0:
            raise RuntimeError(
                "pipeline has no reflection — compiled without "
                "MTL4ShaderReflectionBindingInfo?")
        out: dict[str, TensorBinding] = {}
        # MTL_TENSOR_MAX_RANK is 8 in Apple's headers; reserve 16 for
        # future-proofing.
        DIMS_CAP = 16
        for i in range(n):
            name_buf = ctypes.create_string_buffer(256)
            buf_idx = ctypes.c_int32(0)
            rank = ctypes.c_int32(0)
            dims_arr = (ctypes.c_int64 * DIMS_CAP)()
            dtype_raw = ctypes.c_int32(0)
            rc = info_fn(
                ctypes.c_void_p(self._handle), ctypes.c_int32(i),
                name_buf, ctypes.c_int32(256),
                ctypes.byref(buf_idx),
                ctypes.byref(rank),
                dims_arr, ctypes.c_int32(DIMS_CAP),
                ctypes.byref(dtype_raw))
            if not rc:
                continue
            name = name_buf.value.decode("utf-8", errors="replace")
            r = int(rank.value)
            dims = tuple(int(dims_arr[j]) for j in range(min(r, DIMS_CAP)))
            dtype_int = int(dtype_raw.value)
            out[name] = TensorBinding(
                name=name,
                buffer_index=int(buf_idx.value),
                rank=r,
                dims=dims,
                dtype=_dtype_name_for_raw(dtype_int),
                dtype_raw=dtype_int,
            )
        return out

    def prepare_tensors(self) -> bool:
        """PK3 — Create per-binding ``MTLTensor``s from reflected shapes
        and bind them to a fresh ``MTL4ArgumentTable``. Idempotent:
        second call returns ``True`` if already prepared. Returns
        ``False`` on any failure (dynamic dims, tensor creation
        failure, OS unavailable). Mirrors Apple's sample at
        ``MLMatrixMultiplier.m:configureWithMatrix1:`` (the lines that
        create tensors + bind them by ``binding.index``)."""
        if not self._handle:
            raise RuntimeError("Pipeline already destroyed")
        fn = bind_symbol(
            "tessera_apple_gpu_mlpkg_prepare_tensors",
            (ctypes.c_void_p,), restype=ctypes.c_int32)
        if fn is None:
            return False
        return bool(fn(ctypes.c_void_p(self._handle)))

    def argument_table_ready(self) -> bool:
        """PK3 — Has ``prepare_tensors()`` succeeded? Diagnostic
        helper; tests use it to verify the argument table was actually
        built (vs. a no-op that returned True without doing anything)."""
        if not self._handle:
            return False
        fn = bind_symbol(
            "tessera_apple_gpu_mlpkg_argument_table_ready",
            (ctypes.c_void_p,), restype=ctypes.c_int32)
        if fn is None:
            return False
        return bool(fn(ctypes.c_void_p(self._handle)))

    def fill_input(self, name: str, data: bytes) -> bool:
        """PK3 — Copy ``data`` into the input tensor named ``name``
        (``Pattern 1 / Pattern 2`` from the Apple sample: tensor data
        flows via ``replaceSliceOrigin:sliceDimensions:withBytes:strides:``).

        ``data`` length must equal ``rank-elem-count × dtype-byte-size``
        for the tensor's reflected shape; the runtime validates and
        returns ``False`` on mismatch. ``prepare_tensors()`` must have
        succeeded first."""
        if not self._handle:
            raise RuntimeError("Pipeline already destroyed")
        fn = bind_symbol(
            "tessera_apple_gpu_mlpkg_fill_input",
            (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
             ctypes.c_int64),
            restype=ctypes.c_int32)
        if fn is None:
            return False
        buf = ctypes.c_char_p(bytes(data))
        return bool(fn(ctypes.c_void_p(self._handle),
                       name.encode("utf-8"),
                       buf, ctypes.c_int64(len(data))))

    def read_output(self, name: str, byte_count: int) -> Optional[bytes]:
        """PK3 — Read tensor ``name``'s contents back to host. Returns
        ``None`` on failure (binding missing, OS unavailable, byte
        count mismatch). PK4 uses this to extract dispatch outputs;
        PK3 tests use it for fill-then-read roundtrip without GPU
        execution."""
        if not self._handle:
            raise RuntimeError("Pipeline already destroyed")
        fn = bind_symbol(
            "tessera_apple_gpu_mlpkg_read_output",
            (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
             ctypes.c_int64),
            restype=ctypes.c_int32)
        if fn is None:
            return None
        buf = (ctypes.c_char * byte_count)()
        rc = fn(ctypes.c_void_p(self._handle),
                name.encode("utf-8"),
                buf, ctypes.c_int64(byte_count))
        if not rc:
            return None
        return bytes(buf)

    def dispatch(self, timeout_ms: int = 30_000) -> bool:
        """PK4 — Run the compiled ML pipeline end-to-end on the GPU.

        Pre-condition: ``prepare_tensors()`` must have succeeded and
        ``fill_input()`` must have populated every input tensor with
        the data you want to run on. Post-condition (on True return):
        every output tensor holds the dispatch result — read via
        ``read_output(name, byte_count)``.

        ``timeout_ms`` bounds the GPU wait. Returns ``False`` on
        timeout (kernel hang / driver crash / OS unavailable). Mirrors
        Apple's sample at
        ``MLMatrixMultiplier.m::encodeAndRunModelInference`` and uses
        the audit-recommended ``intermediatesHeap`` sized from
        ``pipelineState.intermediatesHeapSize`` (Action 7 / Pattern 7).
        """
        if not self._handle:
            raise RuntimeError("Pipeline already destroyed")
        fn = bind_symbol(
            "tessera_apple_gpu_mlpkg_dispatch",
            (ctypes.c_void_p, ctypes.c_uint64),
            restype=ctypes.c_int32)
        if fn is None:
            return False
        return bool(fn(ctypes.c_void_p(self._handle),
                       ctypes.c_uint64(timeout_ms)))

    def intermediates_heap_size(self) -> int:
        """PK4 — Cached intermediates-heap size in bytes (allocated
        lazily on first dispatch from ``pipelineState.intermediatesHeapSize``).

        Returns ``-1`` if no dispatch has happened yet OR the runtime
        isn't available. Used by tests + telemetry to confirm
        audit Action 7 (pattern 7) is honored — the heap size comes
        from the pipeline state, not a magic number."""
        if not self._handle:
            return -1
        fn = bind_symbol(
            "tessera_apple_gpu_mlpkg_intermediates_heap_size",
            (ctypes.c_void_p,), restype=ctypes.c_int64)
        if fn is None:
            return -1
        return int(fn(ctypes.c_void_p(self._handle)))

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
    "TensorBinding",
    "compile_mlpackage",
    "last_error_kind",
]
