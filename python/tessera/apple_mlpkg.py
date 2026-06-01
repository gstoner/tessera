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
from typing import Iterable, Optional

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


# ---- PK6: reflection validation gate (audit Action 4) -------------------

@dataclass(frozen=True)
class ExpectedBinding:
    """Compiler-side expectation for a tensor binding — what
    Tessera's manifest entry / compile contract promises the
    ``.mtlpackage`` will expose. ``validate_bindings`` diffs these
    against the live reflection from the loaded pipeline.

    Fields:

    * ``name`` — the binding name (case-sensitive, matches Metal package).
    * ``rank`` — required dim count (``None`` to skip the rank check).
    * ``dtype`` — required Tessera dtype name (``"fp32"`` / ``"fp16"``
      / etc.; ``None`` to skip the dtype check).
    * ``buffer_index`` — required kernel-side argument-table index
      (``None`` to skip the index check). When set, this is the
      strongest contract — a package that re-orders bindings between
      builds will fail validation here.
    * ``dims`` — required exact dimensions (``None`` to skip). Useful
      only for fully-static packages; dynamic-shape packages where
      ``setInputDimensions:`` was called expose post-compile dims.
    """
    name: str
    rank: Optional[int] = None
    dtype: Optional[str] = None
    buffer_index: Optional[int] = None
    dims: Optional[tuple[int, ...]] = None


@dataclass(frozen=True)
class BindingMismatch:
    """A binding that's present in BOTH expected and actual but has a
    diverging attribute. Carries enough detail to point a developer
    at the exact field that drifted."""
    name: str
    field: str   # "rank" / "dtype" / "buffer_index" / "dims"
    expected: object
    actual: object


@dataclass(frozen=True)
class BindingValidation:
    """Result of ``validate_bindings``. Audit Action 4's "named gate":
    when ``ok`` is False, ``first_failure_reason`` names the specific
    mismatch class (``missing`` / ``extra`` / ``mismatched``) and the
    listed entries give the precise per-binding diff.

    A "soft validation" mode (the default) treats EXTRA bindings as a
    warning rather than a hard failure — a package may legitimately
    expose more bindings than the compiler cares about. Pass
    ``strict_extra=True`` to ``validate_bindings`` to fail on extras
    too (recommended for production / drift gates).
    """
    ok: bool
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    mismatched: tuple[BindingMismatch, ...]
    strict_extra: bool

    @property
    def first_failure_reason(self) -> Optional[str]:
        """The audit-Action-4 named gate. ``None`` when ``ok`` is True."""
        if self.missing:
            return f"missing bindings: {', '.join(self.missing)}"
        if self.mismatched:
            m = self.mismatched[0]
            return (f"binding {m.name!r} {m.field} mismatch: "
                    f"expected={m.expected!r} actual={m.actual!r}")
        if self.strict_extra and self.extra:
            return f"unexpected bindings: {', '.join(self.extra)}"
        return None


def validate_bindings(
    pipeline: "Pipeline",
    expected: Iterable[ExpectedBinding],
    *,
    strict_extra: bool = False,
) -> BindingValidation:
    """PK6 — Compare a compiled pipeline's actual reflection against
    a list of compiler-side ``ExpectedBinding`` declarations.

    The diff splits into three buckets:

    * ``missing`` — names in ``expected`` not present in the actual
      reflection. ALWAYS a hard failure (the compiler promised a
      binding the package doesn't have — kernels would crash at
      dispatch).
    * ``extra`` — names in the actual reflection that aren't in
      ``expected``. By default treated as a warning (``ok`` stays
      True if this is the only finding); pass ``strict_extra=True``
      to make it a hard failure (drift-gate mode).
    * ``mismatched`` — names present in both but with diverging
      ``rank`` / ``dtype`` / ``buffer_index`` / ``dims``. ALWAYS a
      hard failure. The ``BindingMismatch`` records exactly which
      field diverged + the expected/actual values.

    Mirrors the audit's "Action 4 — reflection as ABI verification"
    framing: when Tessera loads or generates an Apple ML package, it
    verifies compiler-expected bindings against reflected bindings
    BEFORE marking the artifact executable.
    """
    actual = pipeline.bindings()
    expected_list = list(expected)
    expected_by_name = {e.name: e for e in expected_list}

    missing = tuple(sorted(set(expected_by_name) - set(actual)))
    extra = tuple(sorted(set(actual) - set(expected_by_name)))
    mismatches: list[BindingMismatch] = []
    for name, exp in expected_by_name.items():
        if name not in actual:
            continue
        act = actual[name]
        if exp.rank is not None and exp.rank != act.rank:
            mismatches.append(BindingMismatch(
                name=name, field="rank", expected=exp.rank,
                actual=act.rank))
        if exp.dtype is not None and exp.dtype != act.dtype:
            mismatches.append(BindingMismatch(
                name=name, field="dtype", expected=exp.dtype,
                actual=act.dtype))
        if exp.buffer_index is not None and exp.buffer_index != act.buffer_index:
            mismatches.append(BindingMismatch(
                name=name, field="buffer_index",
                expected=exp.buffer_index, actual=act.buffer_index))
        if exp.dims is not None and tuple(exp.dims) != act.dims:
            mismatches.append(BindingMismatch(
                name=name, field="dims",
                expected=tuple(exp.dims), actual=act.dims))

    # ``ok`` is False on any hard failure (missing OR mismatched), plus
    # extras when strict_extra is True.
    ok = (not missing) and (not mismatches)
    if strict_extra and extra:
        ok = False
    return BindingValidation(
        ok=ok, missing=missing, extra=extra,
        mismatched=tuple(mismatches), strict_extra=strict_extra,
    )


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


# ---- PK7: ArgumentLayout artifact (audit Action 5) ---------------------

@dataclass(frozen=True)
class ArgumentLayoutEntry:
    """One row of the compile-time argument-layout contract for a
    packaged kernel. Mirrors Apple's ``MTL4ArgumentTable`` binding +
    enough metadata for downstream verification.

    Fields:

    * ``name`` — binding name from reflection.
    * ``buffer_index`` — kernel-side argument-table index.
    * ``kind`` — resource kind. Today always ``"tensor"`` (the only
      ``MTLBindingType`` PK1-PK6 surfaced); future PR could add
      ``"buffer"`` / ``"texture"`` once the runtime gains them.
    * ``dtype`` — canonical Tessera dtype name (``"fp32"`` /
      ``"fp16"`` / ...). Maps to ``MTLTensorDataType`` via the PK2
      decoder.
    * ``rank`` — number of dimensions.
    * ``dims`` — extents innermost-first (Apple's MTLTensorExtents
      convention; PK6's ``ExpectedBinding.dims`` uses the same form).
    * ``direction`` — ``"input"`` / ``"output"`` / ``"unknown"``,
      best-effort from the binding name prefix. Apple's reflection
      doesn't directly mark direction; the ``"input"`` / ``"output"``
      naming convention is widely used so we infer from it. Callers
      who need authoritative direction should override post-extract.
    * ``residency`` — placeholder for future per-binding residency
      hints. ``"shared"`` today (unified memory on Apple Silicon);
      future packages might declare ``"private"`` / ``"managed"``.
    """
    name: str
    buffer_index: int
    kind: str
    dtype: str
    rank: int
    dims: tuple[int, ...]
    direction: str
    residency: str


@dataclass(frozen=True)
class ArgumentLayout:
    """The full compile-time argument-layout contract for a packaged
    kernel. Audit Action 5: "emit an Apple ArgumentLayout artifact
    beside backend IR — binding name, index, resource kind,
    tensor/buffer type, dtype, rank, residency requirement."

    Carries enough metadata for:

    * **PK6 validation** — convert via :meth:`to_expected_bindings`
      and feed straight into ``validate_bindings``.
    * **Audit dashboard surface** — :meth:`to_dict` returns a
      JSON-friendly representation.
    * **Compiler-side artifact** — Tessera's manifest can attach an
      ``ArgumentLayout`` to a packaged-kernel ``BackendKernelEntry``
      as the binding contract the compiler emits.
    """
    pipeline_path: str
    function_name: str
    entries: tuple[ArgumentLayoutEntry, ...]

    def by_name(self) -> dict[str, ArgumentLayoutEntry]:
        return {e.name: e for e in self.entries}

    def inputs(self) -> tuple[ArgumentLayoutEntry, ...]:
        return tuple(e for e in self.entries if e.direction == "input")

    def outputs(self) -> tuple[ArgumentLayoutEntry, ...]:
        return tuple(e for e in self.entries if e.direction == "output")

    def to_expected_bindings(self) -> tuple["ExpectedBinding", ...]:
        """Derive a strict PK6 ``ExpectedBinding`` list from this
        layout — checks every field against the actual reflection.
        Pipe through ``validate_bindings`` to drift-check a runtime
        load against the compiler-emitted layout."""
        return tuple(
            ExpectedBinding(
                name=e.name, rank=e.rank, dtype=e.dtype,
                buffer_index=e.buffer_index, dims=e.dims,
            )
            for e in self.entries
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "pipeline_path": self.pipeline_path,
            "function_name": self.function_name,
            "entries": [
                {
                    "name": e.name,
                    "buffer_index": e.buffer_index,
                    "kind": e.kind,
                    "dtype": e.dtype,
                    "rank": e.rank,
                    "dims": list(e.dims),
                    "direction": e.direction,
                    "residency": e.residency,
                }
                for e in self.entries
            ],
        }


def _infer_direction(name: str) -> str:
    """Best-effort direction inference from binding name. Apple's
    sample uses ``"input"`` / ``"output"`` prefixes; many Core ML
    packages follow the same convention. Returns ``"unknown"`` for
    anything else (caller can override post-extract)."""
    lower = name.lower()
    if "output" in lower:
        return "output"
    if "input" in lower:
        return "input"
    return "unknown"


def extract_argument_layout(pipeline: "Pipeline") -> ArgumentLayout:
    """PK7 — Build an ``ArgumentLayout`` from a compiled pipeline's
    reflection. The layout becomes the compiler-side artifact the
    audit asked for (Action 5).

    Pipeline must be compiled (``is_compiled`` True). Raises
    ``RuntimeError`` on a destroyed handle (same lifecycle contract
    as ``bindings()``).
    """
    bindings = pipeline.bindings()
    entries = tuple(
        ArgumentLayoutEntry(
            name=b.name,
            buffer_index=b.buffer_index,
            kind="tensor",       # only kind today
            dtype=b.dtype,
            rank=b.rank,
            dims=b.dims,
            direction=_infer_direction(b.name),
            residency="shared",  # Apple Silicon default
        )
        # Sort by buffer_index for stable round-trip (reflection order
        # IS stable per PK2 tests, but sorting makes the artifact
        # easier to diff in dashboards).
        for b in sorted(bindings.values(), key=lambda x: x.buffer_index)
    )
    return ArgumentLayout(
        pipeline_path=pipeline.package_path,
        function_name=pipeline.function_name,
        entries=entries,
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
    input_dimensions: Optional[dict[int, tuple[int, ...]]] = None,
) -> Optional[Pipeline]:
    """Load ``path`` as a Metal package, compile the named function as
    a ``MTL4MachineLearningPipelineState`` with reflection enabled,
    and return a :class:`Pipeline` handle.

    Returns ``None`` if any step failed (OS unavailable, package load
    failed, pipeline compile failed). Call :func:`last_error_kind` to
    distinguish the cause.

    ``input_dimensions`` (PK1.5, 2026-05-31) is an optional
    ``dict[buffer_index → dims]`` mapping. When the Metal package has
    dynamic-shape inputs (e.g., Apple's sample matrix-multiplication
    package), the descriptor needs concrete shapes via
    ``setInputDimensions:atBufferIndex:`` BEFORE compile. Without
    them the pipeline build fails with "Unsupported Ops or shapes for
    MLEncoder". The buffer_index keys come from
    ``Pipeline.bindings()[name].buffer_index`` — but you'd typically
    have a separate library-reflection pass to discover them; for
    Apple's sample matmul package the convention is well known
    (bindings ``inputA``/``inputB`` at sequential indices, dims
    innermost-first).

    The Apple runtime is JIT-built on first use; on non-Darwin hosts
    this function returns ``None`` with
    ``last_error_kind() == ERROR_OS_UNAVAILABLE``.
    """
    if apple_gpu_runtime() is None:
        return None
    path_str = str(path)
    # Without input_dimensions, use the simpler PK1 entry point.
    if not input_dimensions:
        fn = _bind_compile()
        if fn is None:
            return None
        handle = fn(path_str.encode("utf-8"), function_name.encode("utf-8"))
        if not handle:
            return None
        return Pipeline(handle, package_path=path_str,
                        function_name=function_name)
    # With input_dimensions: pack into the C ABI's flat-array form.
    fn = bind_symbol(
        "tessera_apple_gpu_mlpkg_compile_with_dims",
        (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32,
         ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_int64)),
        restype=ctypes.c_void_p,
    )
    if fn is None:
        return None
    n = len(input_dimensions)
    buf_idx_arr = (ctypes.c_int32 * n)(*input_dimensions.keys())
    ranks_arr = (ctypes.c_int32 * n)(
        *(len(d) for d in input_dimensions.values()))
    flat: list[int] = []
    for dims in input_dimensions.values():
        flat.extend(dims)
    dims_arr = (ctypes.c_int64 * len(flat))(*flat)
    handle = fn(
        path_str.encode("utf-8"), function_name.encode("utf-8"),
        ctypes.c_int32(n), buf_idx_arr, ranks_arr, dims_arr)
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
    "ExpectedBinding",
    "BindingMismatch",
    "BindingValidation",
    "ArgumentLayoutEntry",
    "ArgumentLayout",
    "compile_mlpackage",
    "extract_argument_layout",
    "last_error_kind",
    "validate_bindings",
]
