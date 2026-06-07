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
from typing import Any, Iterable, Optional

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


# ---- Audit Action 2: AppleTensorBindingSpec (compiler-emitted) ---------
#
# Distinction vs PK7's ``ArgumentLayoutEntry`` / ``ArgumentLayout``:
#
# * ``ArgumentLayout*`` is RUNTIME-EXTRACTED from a compiled pipeline's
#   reflection — "what Apple's pipeline actually exposed."
# * ``AppleTensorBindingSpec`` / ``AppleKernelBindingSpec`` is
#   COMPILER-EMITTED — "what the Tessera compiler EXPECTS each binding
#   to look like, declared in advance."
#
# Both shapes round-trip through one another and through
# ``ExpectedBinding`` (PK6 validation):
#
#   compiler          spec.to_expected_bindings()        ┐
#   spec ─────────────────────────────────────────────► ExpectedBinding ─► validate_bindings()
#         spec.to_argument_layout(path, fn) │
#                                           ▼
#                                  ArgumentLayout (runtime form)
#
#   runtime extract ─ AppleKernelBindingSpec.from_argument_layout(layout)
#                                           ▼
#                                  AppleKernelBindingSpec (compiler form)
#
# Wildcard support: ``dims`` accepts ``None`` per axis to mean "any
# value matches." When ``to_expected_bindings()`` sees a wildcard, it
# drops the ``dims`` constraint entirely (PK6 then skips the check) —
# the compiler still pins ``buffer_index`` / ``dtype`` / ``rank``.
# Useful for dynamic-shape kernels where the host calls
# ``setInputDimensions:`` post-load.

@dataclass(frozen=True)
class AppleTensorBindingSpec:
    """Compiler-side declarative spec for ONE tensor binding of an
    Apple packaged kernel. The compiler emits this BEFORE the runtime
    ever loads the ``.mtlpackage``; the runtime extract
    (``ArgumentLayoutEntry``) is then diff-checked against it.

    Fields:

    * ``name`` — binding name (case-sensitive, must match the
      package's reflection).
    * ``buffer_index`` — kernel-side argument-table index. Pinned by
      the compiler — drift here is the audit's headline concern (a
      package that re-orders bindings between builds silently
      misroutes data).
    * ``kind`` — resource kind. ``"tensor"`` today (the only kind
      Apple's reflection surfaces); future packages might expose
      ``"buffer"`` / ``"texture"``.
    * ``dtype`` — canonical Tessera dtype name (``"fp32"`` / ``"fp16"``
      / ...).
    * ``rank`` — number of dimensions.
    * ``dims`` — per-axis extent. ``None`` per axis = wildcard "any
      value" (used for dynamic-shape kernels). Apple's MTLTensorExtents
      convention is innermost-first; this spec follows the same.
    * ``direction`` — ``"input"`` / ``"output"`` / ``"unknown"``.
      Apple's reflection doesn't mark direction, so the compiler
      should set this explicitly when known.
    * ``residency`` — placeholder for future per-binding residency
      hints. ``"shared"`` today (unified memory on Apple Silicon);
      future packages might declare ``"private"`` / ``"managed"``.
    """
    name: str
    buffer_index: int
    kind: str
    dtype: str
    rank: int
    dims: tuple[Optional[int], ...]
    direction: str
    residency: str = "shared"

    def __post_init__(self) -> None:
        # Structural invariants — keep the compiler honest at
        # construction time so a corrupted spec can't propagate
        # silently to a runtime drift gate.
        if self.rank != len(self.dims):
            raise ValueError(
                f"AppleTensorBindingSpec {self.name!r}: rank={self.rank} "
                f"but len(dims)={len(self.dims)} — must match")
        if self.buffer_index < 0:
            raise ValueError(
                f"AppleTensorBindingSpec {self.name!r}: buffer_index="
                f"{self.buffer_index} must be non-negative")
        if self.kind not in ("tensor", "buffer", "texture"):
            raise ValueError(
                f"AppleTensorBindingSpec {self.name!r}: kind={self.kind!r} "
                f"not in {{'tensor', 'buffer', 'texture'}}")
        if self.direction not in ("input", "output", "unknown"):
            raise ValueError(
                f"AppleTensorBindingSpec {self.name!r}: direction="
                f"{self.direction!r} not in {{'input', 'output', 'unknown'}}")
        for i, d in enumerate(self.dims):
            if d is not None and d <= 0:
                raise ValueError(
                    f"AppleTensorBindingSpec {self.name!r}: dims[{i}]={d} "
                    f"must be positive or None (wildcard)")

    @property
    def has_wildcard_dims(self) -> bool:
        """True if at least one axis is a wildcard. Wildcard specs
        won't pin the ``dims`` field in ``to_expected_bindings`` —
        the runtime drift gate falls back to checking ``rank``."""
        return any(d is None for d in self.dims)

    def concrete_dims(self) -> Optional[tuple[int, ...]]:
        """The dims as a concrete tuple iff no axis is a wildcard;
        else ``None``. Useful for round-tripping into PK7's
        ``ArgumentLayoutEntry`` (which doesn't carry wildcards)."""
        if self.has_wildcard_dims:
            return None
        return tuple(int(d) for d in self.dims)  # type: ignore[arg-type]

    def to_expected_binding(self) -> ExpectedBinding:
        """Convert to a PK6 ``ExpectedBinding`` for runtime validation.
        Wildcard dims become "skip the check" — the rest of the spec
        is still strictly pinned (buffer_index / dtype / rank)."""
        return ExpectedBinding(
            name=self.name,
            rank=self.rank,
            dtype=self.dtype,
            buffer_index=self.buffer_index,
            dims=self.concrete_dims(),  # None ↔ skip
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "buffer_index": self.buffer_index,
            "kind": self.kind,
            "dtype": self.dtype,
            "rank": self.rank,
            # ``None`` survives JSON as ``null``; consumers can detect
            # wildcards by checking for ``null`` entries.
            "dims": list(self.dims),
            "direction": self.direction,
            "residency": self.residency,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "AppleTensorBindingSpec":
        dims_raw = d["dims"]
        if not isinstance(dims_raw, (list, tuple)):
            raise ValueError(f"dims must be list/tuple, got {type(dims_raw)}")
        dims = tuple((None if x is None else int(x)) for x in dims_raw)
        # ``d`` is ``dict[str, object]`` (a parsed-JSON record); the
        # numeric fields are int-able at runtime but ``int(object)``
        # fails mypy's overload check. Route them through an ``Any``
        # alias — the same effect isinstance-narrowing gives the
        # ``dims`` elements above.
        da: Any = d
        return cls(
            name=str(d["name"]),
            buffer_index=int(da["buffer_index"]),
            kind=str(d.get("kind", "tensor")),
            dtype=str(d["dtype"]),
            rank=int(da["rank"]),
            dims=dims,
            direction=str(d.get("direction", "unknown")),
            residency=str(d.get("residency", "shared")),
        )

    @classmethod
    def from_argument_layout_entry(
        cls, entry: ArgumentLayoutEntry
    ) -> "AppleTensorBindingSpec":
        """Derive a spec from a runtime-extracted ``ArgumentLayoutEntry``.
        The result has fully-concrete dims (no wildcards) since the
        runtime extract reports actual shapes."""
        return cls(
            name=entry.name,
            buffer_index=entry.buffer_index,
            kind=entry.kind,
            dtype=entry.dtype,
            rank=entry.rank,
            dims=tuple(entry.dims),
            direction=entry.direction,
            residency=entry.residency,
        )

    def to_argument_layout_entry(self) -> ArgumentLayoutEntry:
        """Convert to a PK7 ``ArgumentLayoutEntry``. Raises if the
        spec has wildcard dims — the runtime form requires concrete
        extents."""
        concrete = self.concrete_dims()
        if concrete is None:
            raise ValueError(
                f"AppleTensorBindingSpec {self.name!r}: cannot convert to "
                f"ArgumentLayoutEntry while ``dims`` has wildcards "
                f"({self.dims!r}) — supply concrete dims first.")
        return ArgumentLayoutEntry(
            name=self.name,
            buffer_index=self.buffer_index,
            kind=self.kind,
            dtype=self.dtype,
            rank=self.rank,
            dims=concrete,
            direction=self.direction,
            residency=self.residency,
        )


@dataclass(frozen=True)
class AppleKernelBindingSpec:
    """The full compiler-emitted binding contract for ONE Apple
    packaged kernel. Audit Action 2: "emit an Apple tensor-binding
    spec at the IR layer that runtime drift gates validate against."

    Compose with ``BackendKernelEntry(status='packaged', ...)`` so the
    manifest carries the binding contract beside the package path. The
    runtime can then:

    1. Load the package via ``compile_mlpackage()``.
    2. Extract the runtime layout via ``extract_argument_layout()``.
    3. Diff via ``validate_bindings(pipe, spec.to_expected_bindings())``.
    4. Either proceed (clean diff) or fail loud (named gate).

    Fields:

    * ``function_name`` — the entry-point function the compiler
      compiled against. Must match the runtime's ``compile_mlpackage``
      ``function_name`` argument.
    * ``package_path`` — repository-relative or absolute path to the
      ``.mtlpackage``. Stored for traceability; the runtime resolves
      it via ``BackendKernelEntry.packaged_pipeline_path``.
    * ``entries`` — one ``AppleTensorBindingSpec`` per declared
      binding. Sorted by ``buffer_index`` at construction.
    """
    function_name: str
    package_path: str
    entries: tuple[AppleTensorBindingSpec, ...]

    def __post_init__(self) -> None:
        # Stable order = deterministic round-trip / diff.
        sorted_entries = tuple(
            sorted(self.entries, key=lambda e: e.buffer_index))
        if sorted_entries != self.entries:
            object.__setattr__(self, "entries", sorted_entries)
        # Catch buffer_index collisions early — two bindings can't
        # share an argument-table slot.
        seen: dict[int, str] = {}
        for e in self.entries:
            if e.buffer_index in seen:
                raise ValueError(
                    f"AppleKernelBindingSpec {self.function_name!r}: "
                    f"buffer_index {e.buffer_index} reused by both "
                    f"{seen[e.buffer_index]!r} and {e.name!r}")
            seen[e.buffer_index] = e.name
        names_seen: set[str] = set()
        for e in self.entries:
            if e.name in names_seen:
                raise ValueError(
                    f"AppleKernelBindingSpec {self.function_name!r}: "
                    f"binding name {e.name!r} duplicated")
            names_seen.add(e.name)

    def by_name(self) -> dict[str, AppleTensorBindingSpec]:
        return {e.name: e for e in self.entries}

    def inputs(self) -> tuple[AppleTensorBindingSpec, ...]:
        return tuple(e for e in self.entries if e.direction == "input")

    def outputs(self) -> tuple[AppleTensorBindingSpec, ...]:
        return tuple(e for e in self.entries if e.direction == "output")

    @property
    def has_wildcard_dims(self) -> bool:
        return any(e.has_wildcard_dims for e in self.entries)

    def to_expected_bindings(self) -> tuple[ExpectedBinding, ...]:
        """One PK6 ``ExpectedBinding`` per entry. Feed straight into
        ``validate_bindings(pipe, spec.to_expected_bindings())`` to
        diff a runtime load against the compiler contract."""
        return tuple(e.to_expected_binding() for e in self.entries)

    def to_argument_layout(self) -> "ArgumentLayout":
        """Convert to a PK7 ``ArgumentLayout``. Raises if any entry
        has wildcard dims — the runtime form requires concrete shapes.
        For wildcard specs, only ``to_expected_bindings()`` is
        meaningful (the runtime extract supplies the concrete dims)."""
        return ArgumentLayout(
            pipeline_path=self.package_path,
            function_name=self.function_name,
            entries=tuple(e.to_argument_layout_entry() for e in self.entries),
        )

    @classmethod
    def from_argument_layout(
        cls, layout: "ArgumentLayout"
    ) -> "AppleKernelBindingSpec":
        """Derive a compiler-side spec from a runtime extract. Useful
        for: (a) bootstrapping a manifest entry from a known-good
        package, (b) golden-file tests that snapshot the runtime
        layout into a compiler-side artifact."""
        return cls(
            function_name=layout.function_name,
            package_path=layout.pipeline_path,
            entries=tuple(
                AppleTensorBindingSpec.from_argument_layout_entry(e)
                for e in layout.entries),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "function_name": self.function_name,
            "package_path": self.package_path,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "AppleKernelBindingSpec":
        entries_raw = d["entries"]
        if not isinstance(entries_raw, (list, tuple)):
            raise ValueError(
                f"entries must be list/tuple, got {type(entries_raw)}")
        entries = tuple(
            AppleTensorBindingSpec.from_dict(e)
            for e in entries_raw)
        return cls(
            function_name=str(d["function_name"]),
            package_path=str(d["package_path"]),
            entries=entries,
        )

    def validate_against(
        self, pipeline: "Pipeline", *, strict_extra: bool = False
    ) -> BindingValidation:
        """Convenience: diff this compiler spec against a live runtime
        pipeline's reflection. Equivalent to::

            validate_bindings(pipeline, self.to_expected_bindings(),
                              strict_extra=strict_extra)
        """
        return validate_bindings(
            pipeline, self.to_expected_bindings(),
            strict_extra=strict_extra)


def packaged_ml_available() -> bool:
    """PK audit-fix P1 (2026-05-31) — Is Metal 4 packaged ML actually
    executable on this host?

    The dylib can BUILD on macOS < 26 (clang accepts ``@available``
    guarded code) but `MTL4MachineLearningPipelineDescriptor` and
    friends only become callable on macOS 26+. Tests that gated on
    just ``apple_gpu_runtime() is not None`` proceeded into the
    compile path on older macOS and got
    ``compile_mlpackage() → None`` with
    ``last_error_kind() == ERROR_OS_UNAVAILABLE``, which surfaces as
    a test failure rather than a clean skip.

    This helper does the right capability check: consults the
    runtime's MTL4 capability probe and reports True iff every
    required MTL4 capability is available. Use this in test skip
    guards instead of ``apple_gpu_runtime() is None``::

        if not packaged_ml_available():
            pytest.skip("Metal 4 packaged ML not available on this host")

    Returns False on non-Darwin / macOS < 26 / runtime that can't
    create an MTL4 compiler.
    """
    handle = apple_gpu_runtime()
    if handle is None:
        return False
    probe = bind_symbol(
        "tessera_apple_gpu_metal4_probe",
        (ctypes.POINTER(ctypes.c_int32),),
        restype=ctypes.c_int32,
    )
    if probe is None:
        return False
    caps_out = ctypes.c_int32(0)
    # The probe returns 1 iff ALL MTL4 capability bits are set
    # (queue + allocator + compiler + tensor + MSL 4.0). That's
    # exactly the surface packaged-ML needs.
    return bool(probe(ctypes.byref(caps_out)))


def packaged_ml_skip_reason() -> Optional[str]:
    """Return a human-readable reason packaged ML isn't available, or
    ``None`` when it is. Useful for pytest.skip messages so the
    failing surface is named, not just "skipped"."""
    handle = apple_gpu_runtime()
    if handle is None:
        return "Apple GPU runtime dylib not buildable on this host"
    if not packaged_ml_available():
        return ("Metal 4 packaged ML not available — host SDK builds the "
                "dylib but MTL4 capabilities probe returns false (likely "
                "macOS < 26)")
    return None


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

    def set_aligned_strides(self, enabled: bool) -> bool:
        """Phase 2 stride-alignment wire-up (2026-06-01) — opt in to
        Apple's stride-alignment rules at the ``MTLTensorDescriptor``
        level. When enabled, ``prepare_tensors`` sets ``td.strides``
        explicitly via the aligned helper (64-byte for ML-usage
        byte+ dtypes, 128-byte for sub-byte dtypes); Metal allocates
        storage accordingly. Default off (Metal's implicit strides).

        Must be called BEFORE ``prepare_tensors`` to take effect.
        Returns True on success, False if the runtime is unavailable.
        """
        if not self._handle:
            raise RuntimeError("Pipeline already destroyed")
        fn = bind_symbol(
            "tessera_apple_gpu_mlpkg_set_aligned_strides",
            (ctypes.c_void_p, ctypes.c_int32), restype=ctypes.c_int32)
        if fn is None:
            return False
        return bool(fn(ctypes.c_void_p(self._handle),
                       ctypes.c_int32(1 if enabled else 0)))

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

    def fill_input_at(self, index: int, data: bytes) -> bool:
        """PK8 — Copy ``data`` into the input tensor at kernel-side binding
        ``index``. The positional addressing mode for packages whose
        bindings carry no names — MPSGraph-authored packages (vs the
        CoreML-origin Apple sample) expose *unnamed* bindings, so
        :meth:`fill_input` by name can't disambiguate them. Indices come
        from the per-binding ``buffer_index`` reflection."""
        if not self._handle:
            raise RuntimeError("Pipeline already destroyed")
        fn = bind_symbol(
            "tessera_apple_gpu_mlpkg_fill_input_at",
            (ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p,
             ctypes.c_int64),
            restype=ctypes.c_int32)
        if fn is None:
            return False
        buf = ctypes.c_char_p(bytes(data))
        return bool(fn(ctypes.c_void_p(self._handle), ctypes.c_int32(index),
                       buf, ctypes.c_int64(len(data))))

    def read_output_at(self, index: int, byte_count: int) -> Optional[bytes]:
        """PK8 — Read the tensor at kernel-side binding ``index`` back to
        host. Positional counterpart to :meth:`read_output` for unnamed
        (MPSGraph-authored) packages."""
        if not self._handle:
            raise RuntimeError("Pipeline already destroyed")
        fn = bind_symbol(
            "tessera_apple_gpu_mlpkg_read_output_at",
            (ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p,
             ctypes.c_int64),
            restype=ctypes.c_int32)
        if fn is None:
            return None
        buf = (ctypes.c_char * byte_count)()
        rc = fn(ctypes.c_void_p(self._handle), ctypes.c_int32(index),
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


# ---- PK8: author a production .mtlpackage from the MPSGraph lane --------


def first_function_name(path: str | Path) -> Optional[str]:
    """Return the entry-point function name an ``.mtlpackage`` exposes.

    MPSGraph names the serialized function itself, so a Tessera-authored
    package's entry point is not necessarily ``"main"``. This loads the
    package's ``MTLLibrary`` and returns its first function name, so
    callers can feed it to :func:`compile_mlpackage`. Returns ``None``
    when the runtime is unavailable or the package exposes no functions.
    """
    if apple_gpu_runtime() is None:
        return None
    fn = bind_symbol(
        "tessera_apple_gpu_mlpkg_first_function_name",
        (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32),
        restype=ctypes.c_int32,
    )
    if fn is None:
        return None
    buf = ctypes.create_string_buffer(256)
    ok = int(fn(str(path).encode("utf-8"), buf, ctypes.c_int32(256)))
    if ok != 1:
        return None
    return buf.value.decode("utf-8") or None


def author_matmul_package(
    out_path: str | Path, m: int, k: int, n: int
) -> bool:
    """Author a production ``.mtlpackage`` for ``C[m,n] = A[m,k] @ B[k,n]``
    (fp32) from Tessera's MPSGraph lane.

    Builds an ``MPSGraph`` matmul, compiles it to an ``MPSGraphExecutable``,
    serializes that to ``<out_path>/library.mpsgraphpackage`` via
    ``serializeToMPSGraphPackageAtURL:``, and writes the ``manifest.json``
    MLLibrary wrapper — producing a ``.mtlpackage`` directory that
    :func:`compile_mlpackage` (PK1) can load and the PK1-PK7 lifecycle can
    dispatch. No coremltools, no DXIL: this rides the same MPSGraph
    primitive the runtime already builds for its MPSGraph-lane ops.

    Returns ``True`` on success. Returns ``False`` when the Apple runtime
    is unavailable (non-Darwin / pre-macOS-14) or any authoring step fails;
    no exception is raised so callers can skip cleanly.
    """
    if apple_gpu_runtime() is None:
        return False
    fn = bind_symbol(
        "tessera_apple_gpu_mlpkg_author_matmul",
        (ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32),
        restype=ctypes.c_int32,
    )
    if fn is None:
        return False
    rc = int(fn(str(out_path).encode("utf-8"),
                ctypes.c_int32(m), ctypes.c_int32(k), ctypes.c_int32(n)))
    return rc == 1


# Ops the generalized authoring path supports, grouped by binding arity.
# (Kept in Python so callers/tests can introspect without the runtime.)
AUTHOR_OP_UNARY = (
    "relu", "sigmoid", "tanh", "softplus", "silu", "gelu",
    "exp", "log", "sqrt", "rsqrt", "neg", "abs",
)
AUTHOR_OP_ROWOP = ("softmax", "log_softmax", "rmsnorm", "layer_norm")
AUTHOR_OP_BINARY = ("add", "sub", "mul", "div", "max", "min", "silu_mul")
AUTHOR_OPS = AUTHOR_OP_UNARY + AUTHOR_OP_ROWOP + AUTHOR_OP_BINARY


def author_op_package(
    out_path: str | Path,
    op: str,
    rows: int,
    cols: int,
    *,
    eps: float = 1e-5,
    weighted: bool = False,
) -> bool:
    """Author a production ``.mtlpackage`` for an MPSGraph-lane op over a
    ``[rows, cols]`` fp32 input — the generalized PK8 path.

    ``op`` is one of :data:`AUTHOR_OPS`:

    * **unary** (1 input): ``relu`` ``sigmoid`` ``tanh`` ``softplus``
      ``silu`` ``gelu`` ``exp`` ``log`` ``sqrt`` ``rsqrt`` ``neg`` ``abs``
    * **rowop** (1 input, over the last axis): ``softmax`` ``log_softmax``
    * **norm** (1 input + optional ``gamma``/``beta`` when ``weighted``):
      ``rmsnorm`` (gamma), ``layer_norm`` (gamma + beta)
    * **binary** (2 inputs): ``add`` ``sub`` ``mul`` ``div`` ``max`` ``min``
      ``silu_mul``

    The authored graph reuses the *same* builders the runtime dispatches at
    execution time, so the packaged kernel is numerically identical to the
    live MPSGraph path. ``eps`` applies to the norms; ``weighted`` adds the
    gamma/beta inputs. Bindings are positional (use ``fill_input_at`` /
    ``read_output_at``): inputs first, output last.

    Returns ``True`` on success; ``False`` when the runtime is unavailable
    or ``op`` is not recognized.
    """
    if apple_gpu_runtime() is None:
        return False
    fn = bind_symbol(
        "tessera_apple_gpu_mlpkg_author_op",
        (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_float, ctypes.c_int32),
        restype=ctypes.c_int32,
    )
    if fn is None:
        return False
    rc = int(fn(str(out_path).encode("utf-8"), op.encode("utf-8"),
                ctypes.c_int32(rows), ctypes.c_int32(cols),
                ctypes.c_float(eps), ctypes.c_int32(1 if weighted else 0)))
    return rc == 1


# Fused chains the multi-op authoring path supports. Each maps to a
# dims-vector contract (see ``author_chain_package``).
AUTHOR_CHAINS = ("matmul_softmax", "matmul_softmax_matmul", "rmsnorm_matmul")


def author_chain_package(
    out_path: str | Path,
    chain: str,
    dims: "tuple[int, ...] | list[int]",
    *,
    eps: float = 1e-5,
) -> bool:
    """Author a *fused multi-op* ``.mtlpackage`` — a whole chain composed into
    one serialized MPSGraph executable (one GPU dispatch). The packaged
    equivalent of the runtime's fused MSL kernels.

    ``chain`` is one of :data:`AUTHOR_CHAINS`; ``dims`` is the per-chain shape
    vector:

    * ``matmul_softmax`` — ``[M, K, N]``: ``A[M,K], B[K,N] → softmax(A@B)[M,N]``
    * ``matmul_softmax_matmul`` — ``[M, K, N, P]``:
      ``A, B, C → (softmax(A@B) @ C)[M,P]`` (the attention block)
    * ``rmsnorm_matmul`` — ``[M, K, N]``:
      ``x[M,K], gamma[K], W[K,N] → (rmsnorm(x)*gamma) @ W [M,N]``

    Bindings are positional (inputs in the order above, output last); use
    ``fill_input_at`` / ``read_output_at``. Returns ``True`` on success;
    ``False`` when the runtime is unavailable, ``chain`` is unknown, or
    ``dims`` has the wrong arity.
    """
    if apple_gpu_runtime() is None:
        return False
    fn = bind_symbol(
        "tessera_apple_gpu_mlpkg_author_chain",
        (ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32),
         ctypes.c_int32, ctypes.c_float),
        restype=ctypes.c_int32,
    )
    if fn is None:
        return False
    n = len(dims)
    arr = (ctypes.c_int32 * n)(*dims)
    rc = int(fn(str(out_path).encode("utf-8"), chain.encode("utf-8"),
                arr, ctypes.c_int32(n), ctypes.c_float(eps)))
    return rc == 1


# Opcodes — mirror the switch in tessera_apple_gpu_mlpkg_author_graph
# (apple_gpu_runtime.mm). Keep in sync if either side changes.
GRAPH_OP: dict[str, int] = {
    "matmul": 0,
    "add": 1, "sub": 2, "mul": 3, "div": 4,
    "softmax": 10, "rmsnorm": 11, "layer_norm": 12,
    "relu": 20, "sigmoid": 21, "tanh": 22, "silu": 23, "gelu": 24,
}


def author_graph_package(
    out_path: str | Path,
    arg_shapes: "list[tuple[int, ...]]",
    ops: "list[dict]",
    output_id: int,
    *,
    bf16: bool = False,
) -> bool:
    """Author an ARBITRARY straight-line op graph as ONE serialized MPSGraph
    package — the whole graph becomes a single executable, so it runs as ONE
    Metal dispatch (MPSGraph fuses globally). This is the "graph as one fused
    unit" authoring path (PK8c).

    * ``arg_shapes`` — per-input shape ``(rows, cols)``; ``cols <= 0`` (or a
      1-tuple) declares a rank-1 vector of length ``rows``.
    * ``ops`` — straight-line op list; each is a dict
      ``{"op": <name in GRAPH_OP>, "in0": <tensor id>, "in1": <id or -1>,
      "transpose_a": bool, "transpose_b": bool, "eps": float}``.
    * Tensor ids: ``0..len(arg_shapes)-1`` are inputs; op ``j`` produces id
      ``len(arg_shapes)+j``. ``output_id`` is the single result.

    Positional bindings (inputs at ``0..``, output last); drive with
    ``fill_input_at`` / ``read_output_at`` then ``read_output_at(len(args), ...)``.
    When ``bf16=True`` the boundary tensors are bf16 (inputs/output) while internal
    compute stays f32 (the ABI f32-accumulate policy) — fill/read bf16 bytes.
    Returns ``True`` on success; ``False`` when the runtime is unavailable or the
    graph is malformed (unknown op / bad tensor id).
    """
    if apple_gpu_runtime() is None:
        return False
    fn = bind_symbol(
        "tessera_apple_gpu_mlpkg_author_graph",
        (ctypes.c_char_p, ctypes.c_int32,
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.c_int32,
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32),
        restype=ctypes.c_int32,
    )
    if fn is None:
        return False
    na = len(arg_shapes)
    rows = (ctypes.c_int32 * na)(*[int(s[0]) for s in arg_shapes])
    cols = (ctypes.c_int32 * na)(
        *[int(s[1]) if len(s) > 1 else 0 for s in arg_shapes])
    no = len(ops)
    codes = (ctypes.c_int32 * no)()
    in0 = (ctypes.c_int32 * no)()
    in1 = (ctypes.c_int32 * no)()
    iattr = (ctypes.c_int32 * no)()
    fattr = (ctypes.c_float * no)()
    for j, o in enumerate(ops):
        if o["op"] not in GRAPH_OP:
            return False
        codes[j] = GRAPH_OP[o["op"]]
        in0[j] = int(o["in0"])
        in1[j] = int(o.get("in1", -1))
        bits = (1 if o.get("transpose_a") else 0) | (2 if o.get("transpose_b") else 0)
        iattr[j] = bits
        fattr[j] = float(o.get("eps", 1e-5))
    rc = int(fn(str(out_path).encode("utf-8"), ctypes.c_int32(na), rows, cols,
                ctypes.c_int32(no), codes, in0, in1, iattr, fattr,
                ctypes.c_int32(int(output_id)), ctypes.c_int32(1 if bf16 else 0)))
    return rc == 1


def run_graph_loop_f32(
    arg_arrays: "list",
    arg_shapes: "list[tuple[int, ...]]",
    carry_arg_index: int,
    trip: int,
    body_ops: "list[dict]",
    body_out_id: int,
    out_shape: "tuple[int, ...]",
):
    """Run a BOUNDED for-loop as ONE MPSGraph ``forLoop``, executed directly
    (PK8d / Phase-G G-A): ``for _ in range(trip): carry = body(carry, args)`` →
    the final carry. The package/MLEncoder path rejects control-flow ops, so the
    loop graph is built + run + read in one call (like ``cf_scan``).

    Body tensor ids: ``0..len(args)-1`` = args, ``len(args)`` = the carry,
    ``len(args)+1+j`` = body op ``j`` (op dicts as in :func:`author_graph_package`).
    ``carry_arg_index`` is the arg initializing the carry; ``body_out_id`` is the
    next-carry tensor. f32. Returns the final carry as an ``np.float32`` array of
    ``out_shape``, or ``None`` if the runtime is unavailable / the loop is
    malformed (caller falls back to host)."""
    import numpy as np

    if apple_gpu_runtime() is None:
        return None
    fn = bind_symbol(
        "tessera_apple_gpu_run_graph_loop_f32",
        (ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
         ctypes.POINTER(ctypes.c_float)),
        restype=ctypes.c_int32,
    )
    if fn is None:
        return None
    na = len(arg_shapes)
    cargs = [np.ascontiguousarray(np.asarray(a, dtype=np.float32)) for a in arg_arrays]
    ptrs = (ctypes.c_void_p * na)(
        *[ctypes.cast(a.ctypes.data, ctypes.c_void_p) for a in cargs])
    rows = (ctypes.c_int32 * na)(*[int(s[0]) for s in arg_shapes])
    cols = (ctypes.c_int32 * na)(
        *[int(s[1]) if len(s) > 1 else 0 for s in arg_shapes])
    nb = len(body_ops)
    codes = (ctypes.c_int32 * nb)()
    in0 = (ctypes.c_int32 * nb)()
    in1 = (ctypes.c_int32 * nb)()
    iattr = (ctypes.c_int32 * nb)()
    fattr = (ctypes.c_float * nb)()
    for j, o in enumerate(body_ops):
        if o["op"] not in GRAPH_OP:
            return None
        codes[j] = GRAPH_OP[o["op"]]
        in0[j] = int(o["in0"])
        in1[j] = int(o.get("in1", -1))
        iattr[j] = (1 if o.get("transpose_a") else 0) | (2 if o.get("transpose_b") else 0)
        fattr[j] = float(o.get("eps", 1e-5))
    out = np.zeros(out_shape, dtype=np.float32)
    rc = int(fn(ctypes.c_int32(na), ptrs, rows, cols,
                ctypes.c_int32(int(carry_arg_index)), ctypes.c_int32(int(trip)),
                ctypes.c_int32(nb), codes, in0, in1, iattr, fattr,
                ctypes.c_int32(int(body_out_id)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
    return out if rc == 1 else None


def _flatten_ops(ops: "list[dict]"):
    """Flatten an op-dict list to the 5 ctypes arrays the C ABI expects, or
    ``None`` if any op name is unknown."""
    nb = len(ops)
    codes = (ctypes.c_int32 * nb)()
    in0 = (ctypes.c_int32 * nb)()
    in1 = (ctypes.c_int32 * nb)()
    iattr = (ctypes.c_int32 * nb)()
    fattr = (ctypes.c_float * nb)()
    for j, o in enumerate(ops):
        if o["op"] not in GRAPH_OP:
            return None
        codes[j] = GRAPH_OP[o["op"]]
        in0[j] = int(o["in0"])
        in1[j] = int(o.get("in1", -1))
        iattr[j] = (1 if o.get("transpose_a") else 0) | (2 if o.get("transpose_b") else 0)
        fattr[j] = float(o.get("eps", 1e-5))
    return codes, in0, in1, iattr, fattr, nb


def run_graph_cond_f32(
    arg_arrays: "list",
    arg_shapes: "list[tuple[int, ...]]",
    flag_arg_index: int,
    then_ops: "list[dict]",
    then_out_id: int,
    else_ops: "list[dict]",
    else_out_id: int,
    out_shape: "tuple[int, ...]",
):
    """Run a ``cond`` (divergent if/else) as ONE MPSGraph ``if``, executed
    directly (PK8e / Phase-G G-A.2). The predicate is ``args[flag_arg_index] > 0``;
    only the taken branch executes. Each branch is a straight-line op-list (op
    dicts as in :func:`author_graph_package`) over the args — tensor ids per
    branch: ``0..len(args)-1`` = args, ``len(args)+j`` = branch op ``j``; the
    ``*_out_id`` is that branch's result. Both branches must produce ``out_shape``.
    Returns the result as an ``np.float32`` array, or ``None`` if the runtime is
    unavailable / a branch is malformed (caller falls back to host)."""
    import numpy as np

    if apple_gpu_runtime() is None:
        return None
    fn = bind_symbol(
        "tessera_apple_gpu_run_graph_cond_f32",
        (ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.c_int32,
         ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
         ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
         ctypes.POINTER(ctypes.c_float)),
        restype=ctypes.c_int32,
    )
    if fn is None:
        return None
    tflat = _flatten_ops(then_ops)
    eflat = _flatten_ops(else_ops)
    if tflat is None or eflat is None:
        return None
    tc, ti0, ti1, tia, tfa, tnb = tflat
    ec, ei0, ei1, eia, efa, enb = eflat
    na = len(arg_shapes)
    cargs = [np.ascontiguousarray(np.asarray(a, dtype=np.float32)) for a in arg_arrays]
    ptrs = (ctypes.c_void_p * na)(
        *[ctypes.cast(a.ctypes.data, ctypes.c_void_p) for a in cargs])
    rows = (ctypes.c_int32 * na)(*[int(s[0]) for s in arg_shapes])
    cols = (ctypes.c_int32 * na)(
        *[int(s[1]) if len(s) > 1 else 0 for s in arg_shapes])
    out = np.zeros(out_shape, dtype=np.float32)
    rc = int(fn(ctypes.c_int32(na), ptrs, rows, cols, ctypes.c_int32(int(flag_arg_index)),
                ctypes.c_int32(tnb), tc, ti0, ti1, tia, tfa, ctypes.c_int32(int(then_out_id)),
                ctypes.c_int32(enb), ec, ei0, ei1, eia, efa, ctypes.c_int32(int(else_out_id)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
    return out if rc == 1 else None


def run_graph_while_f32(
    arg_arrays: "list",
    arg_shapes: "list[tuple[int, ...]]",
    carry_arg_index: int,
    max_iters: int,
    body_ops: "list[dict]",
    body_out_id: int,
    cond_ops: "list[dict]",
    cond_out_id: int,
    out_shape: "tuple[int, ...]",
):
    """Run a BOUNDED ``while`` as ONE MPSGraph ``forLoop`` with select-masking
    (PK8f / Phase-G G-A.3): ``for i in range(max_iters): pred = cond(carry)>0;
    carry = pred ? body(carry) : carry`` → the final carry. MPSGraph's native
    ``while`` is unstable, so a max-iter-capped while lowers to forLoop+select.
    Body and cond are straight-line op-lists over args + carry (ids:
    ``0..len(args)-1`` args, ``len(args)`` carry, ``len(args)+1+j`` op ``j``);
    ``cond_out_id`` is the predicate source. Returns the final carry as an
    ``np.float32`` array, or ``None`` (runtime unavailable / malformed)."""
    import numpy as np

    if apple_gpu_runtime() is None:
        return None
    fn = bind_symbol(
        "tessera_apple_gpu_run_graph_while_f32",
        (ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.c_int32, ctypes.c_int32,
         ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
         ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
         ctypes.POINTER(ctypes.c_float)),
        restype=ctypes.c_int32,
    )
    if fn is None:
        return None
    bflat = _flatten_ops(body_ops)
    cflat = _flatten_ops(cond_ops)
    if bflat is None or cflat is None:
        return None
    bc, bi0, bi1, bia, bfa, bnb = bflat
    cc, ci0, ci1, cia, cfa, cnb = cflat
    na = len(arg_shapes)
    cargs = [np.ascontiguousarray(np.asarray(a, dtype=np.float32)) for a in arg_arrays]
    ptrs = (ctypes.c_void_p * na)(
        *[ctypes.cast(a.ctypes.data, ctypes.c_void_p) for a in cargs])
    rows = (ctypes.c_int32 * na)(*[int(s[0]) for s in arg_shapes])
    cols = (ctypes.c_int32 * na)(
        *[int(s[1]) if len(s) > 1 else 0 for s in arg_shapes])
    out = np.zeros(out_shape, dtype=np.float32)
    rc = int(fn(ctypes.c_int32(na), ptrs, rows, cols,
                ctypes.c_int32(int(carry_arg_index)), ctypes.c_int32(int(max_iters)),
                ctypes.c_int32(bnb), bc, bi0, bi1, bia, bfa, ctypes.c_int32(int(body_out_id)),
                ctypes.c_int32(cnb), cc, ci0, ci1, cia, cfa, ctypes.c_int32(int(cond_out_id)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
    return out if rc == 1 else None


# ── Phase-H H2 — native f16 control flow ──────────────────────────────────────
# MPSGraph supports f16 natively (bf16 has no MPSGraph type → it stays host-upcast
# to f32). These mirror the f32 wrappers exactly but build/run in f16 over the
# f16-bit ABI: args + the result are ``np.float16`` (the runtime takes
# ``uint16_t*`` for the result). The body-op packing (`_flatten_ops`) is
# dtype-independent.
def run_graph_loop_f16(
    arg_arrays: "list", arg_shapes: "list[tuple[int, ...]]", carry_arg_index: int,
    trip: int, body_ops: "list[dict]", body_out_id: int,
    out_shape: "tuple[int, ...]",
):
    """f16 counterpart of :func:`run_graph_loop_f32` (native f16 MPSGraph loop).
    Returns the final carry as an ``np.float16`` array, or ``None``."""
    import numpy as np

    if apple_gpu_runtime() is None:
        return None
    fn = bind_symbol(
        "tessera_apple_gpu_run_graph_loop_f16",
        (ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
         ctypes.POINTER(ctypes.c_uint16)),
        restype=ctypes.c_int32,
    )
    if fn is None:
        return None
    flat = _flatten_ops(body_ops)
    if flat is None:
        return None
    codes, in0, in1, iattr, fattr, nb = flat
    na = len(arg_shapes)
    cargs = [np.ascontiguousarray(np.asarray(a, dtype=np.float16)) for a in arg_arrays]
    ptrs = (ctypes.c_void_p * na)(
        *[ctypes.cast(a.ctypes.data, ctypes.c_void_p) for a in cargs])
    rows = (ctypes.c_int32 * na)(*[int(s[0]) for s in arg_shapes])
    cols = (ctypes.c_int32 * na)(
        *[int(s[1]) if len(s) > 1 else 0 for s in arg_shapes])
    out = np.zeros(out_shape, dtype=np.float16)
    rc = int(fn(ctypes.c_int32(na), ptrs, rows, cols,
                ctypes.c_int32(int(carry_arg_index)), ctypes.c_int32(int(trip)),
                ctypes.c_int32(nb), codes, in0, in1, iattr, fattr,
                ctypes.c_int32(int(body_out_id)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))))
    return out if rc == 1 else None


def run_graph_cond_f16(
    arg_arrays: "list", arg_shapes: "list[tuple[int, ...]]", flag_arg_index: int,
    then_ops: "list[dict]", then_out_id: int, else_ops: "list[dict]",
    else_out_id: int, out_shape: "tuple[int, ...]",
):
    """f16 counterpart of :func:`run_graph_cond_f32`. Returns ``np.float16``/None."""
    import numpy as np

    if apple_gpu_runtime() is None:
        return None
    fn = bind_symbol(
        "tessera_apple_gpu_run_graph_cond_f16",
        (ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.c_int32,
         ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
         ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
         ctypes.POINTER(ctypes.c_uint16)),
        restype=ctypes.c_int32,
    )
    if fn is None:
        return None
    tflat, eflat = _flatten_ops(then_ops), _flatten_ops(else_ops)
    if tflat is None or eflat is None:
        return None
    tc, ti0, ti1, tia, tfa, tnb = tflat
    ec, ei0, ei1, eia, efa, enb = eflat
    na = len(arg_shapes)
    cargs = [np.ascontiguousarray(np.asarray(a, dtype=np.float16)) for a in arg_arrays]
    ptrs = (ctypes.c_void_p * na)(
        *[ctypes.cast(a.ctypes.data, ctypes.c_void_p) for a in cargs])
    rows = (ctypes.c_int32 * na)(*[int(s[0]) for s in arg_shapes])
    cols = (ctypes.c_int32 * na)(
        *[int(s[1]) if len(s) > 1 else 0 for s in arg_shapes])
    out = np.zeros(out_shape, dtype=np.float16)
    rc = int(fn(ctypes.c_int32(na), ptrs, rows, cols, ctypes.c_int32(int(flag_arg_index)),
                ctypes.c_int32(tnb), tc, ti0, ti1, tia, tfa, ctypes.c_int32(int(then_out_id)),
                ctypes.c_int32(enb), ec, ei0, ei1, eia, efa, ctypes.c_int32(int(else_out_id)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))))
    return out if rc == 1 else None


def run_graph_while_f16(
    arg_arrays: "list", arg_shapes: "list[tuple[int, ...]]", carry_arg_index: int,
    max_iters: int, body_ops: "list[dict]", body_out_id: int,
    cond_ops: "list[dict]", cond_out_id: int, out_shape: "tuple[int, ...]",
):
    """f16 counterpart of :func:`run_graph_while_f32`. Returns ``np.float16``/None."""
    import numpy as np

    if apple_gpu_runtime() is None:
        return None
    fn = bind_symbol(
        "tessera_apple_gpu_run_graph_while_f16",
        (ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.c_int32, ctypes.c_int32,
         ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
         ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
         ctypes.POINTER(ctypes.c_uint16)),
        restype=ctypes.c_int32,
    )
    if fn is None:
        return None
    bflat, cflat = _flatten_ops(body_ops), _flatten_ops(cond_ops)
    if bflat is None or cflat is None:
        return None
    bc, bi0, bi1, bia, bfa, bnb = bflat
    cc, ci0, ci1, cia, cfa, cnb = cflat
    na = len(arg_shapes)
    cargs = [np.ascontiguousarray(np.asarray(a, dtype=np.float16)) for a in arg_arrays]
    ptrs = (ctypes.c_void_p * na)(
        *[ctypes.cast(a.ctypes.data, ctypes.c_void_p) for a in cargs])
    rows = (ctypes.c_int32 * na)(*[int(s[0]) for s in arg_shapes])
    cols = (ctypes.c_int32 * na)(
        *[int(s[1]) if len(s) > 1 else 0 for s in arg_shapes])
    out = np.zeros(out_shape, dtype=np.float16)
    rc = int(fn(ctypes.c_int32(na), ptrs, rows, cols,
                ctypes.c_int32(int(carry_arg_index)), ctypes.c_int32(int(max_iters)),
                ctypes.c_int32(bnb), bc, bi0, bi1, bia, bfa, ctypes.c_int32(int(body_out_id)),
                ctypes.c_int32(cnb), cc, ci0, ci1, cia, cfa, ctypes.c_int32(int(cond_out_id)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))))
    return out if rc == 1 else None


def run_graph_scan_f32(
    arg_arrays: "list", arg_shapes: "list[tuple[int, ...]]", carry_arg_index: int,
    xs_array, trip: int, x_shape: "tuple[int, ...]", body_ops: "list[dict]",
    carry_out_id: int, y_out_id: int, carry_shape: "tuple[int, ...]",
    y_shape: "tuple[int, ...]",
):
    """Run a fused ``scan`` (Phase-H H3) as ONE MPSGraph ``forLoop`` carrying
    ``[carry, ys]``: per step ``x_t = xs[index]``;
    ``(carry, y_t) = body(args, carry, x_t)``; ``ys[index] = y_t``. Returns
    ``(final_carry, ys)`` as ``np.float32`` arrays (``ys`` is ``(trip, *y_shape)``),
    or ``None`` (runtime unavailable / malformed). Body tensor ids:
    ``0..len(args)-1`` = args (consts; carry init = ``args[carry_arg_index]``),
    ``len(args)`` = carry, ``len(args)+1`` = ``x_t``, ``len(args)+2+j`` = body op
    ``j``. ``xs``/``ys`` are ``(trip, *2D-inner)``; consts/carry are rank<=2."""
    import numpy as np

    if apple_gpu_runtime() is None:
        return None
    fn = bind_symbol(
        "tessera_apple_gpu_run_graph_scan_f32",
        (ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
         ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)),
        restype=ctypes.c_int32,
    )
    if fn is None:
        return None
    flat = _flatten_ops(body_ops)
    if flat is None:
        return None
    codes, in0, in1, iattr, fattr, nb = flat
    na = len(arg_shapes)
    cargs = [np.ascontiguousarray(np.asarray(a, dtype=np.float32)) for a in arg_arrays]
    ptrs = (ctypes.c_void_p * na)(
        *[ctypes.cast(a.ctypes.data, ctypes.c_void_p) for a in cargs])
    rows = (ctypes.c_int32 * na)(*[int(s[0]) for s in arg_shapes])
    cols = (ctypes.c_int32 * na)(
        *[int(s[1]) if len(s) > 1 else 0 for s in arg_shapes])
    xs = np.ascontiguousarray(np.asarray(xs_array, dtype=np.float32))
    x_rows = int(x_shape[0])
    x_cols = int(x_shape[1]) if len(x_shape) > 1 else 0
    c_rows = int(carry_shape[0])
    c_cols = int(carry_shape[1]) if len(carry_shape) > 1 else 0
    y_rows = int(y_shape[0])
    y_cols = int(y_shape[1]) if len(y_shape) > 1 else 0
    ys_zeros = np.zeros((int(trip), *y_shape), dtype=np.float32)
    out_carry = np.zeros(carry_shape, dtype=np.float32)
    out_ys = np.zeros((int(trip), *y_shape), dtype=np.float32)
    rc = int(fn(
        ctypes.c_int32(na), ptrs, rows, cols, ctypes.c_int32(int(carry_arg_index)),
        ctypes.cast(xs.ctypes.data, ctypes.c_void_p), ctypes.c_int32(int(trip)),
        ctypes.c_int32(x_rows), ctypes.c_int32(x_cols),
        ctypes.c_int32(nb), codes, in0, in1, iattr, fattr,
        ctypes.c_int32(int(carry_out_id)), ctypes.c_int32(int(y_out_id)),
        ctypes.c_int32(c_rows), ctypes.c_int32(c_cols),
        ctypes.c_int32(y_rows), ctypes.c_int32(y_cols),
        ctypes.cast(ys_zeros.ctypes.data, ctypes.c_void_p),
        out_carry.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_ys.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
    return (out_carry, out_ys) if rc == 1 else None


def execute_control_loop_mlir(mlir_text: str, arg_arrays: "list"):
    """MLIR-driven execution (Phase-G G-B.2): given a lowered module containing a
    `tessera_apple.gpu.control_loop` op (with the serialized body op-list payload),
    read the op's attributes and dispatch via its recorded runtime `symbol`
    (`tessera_apple_gpu_run_graph_loop_f32`). This is what makes the IR path
    *execute*, not just lit-check. Returns the final carry as an ``np.float32``
    array, or ``None`` if the op/payload is absent or malformed."""
    import re

    import numpy as np

    if "tessera_apple.gpu.control_loop" not in mlir_text:
        return None
    seg = mlir_text[mlir_text.index("tessera_apple.gpu.control_loop"):]
    try:
        attrs = seg[seg.index("{"):seg.index("}") + 1]
    except ValueError:
        return None

    def i32arr(name):
        m = re.search(name + r"\s*=\s*array<i32(?::\s*([^>]*))?>", attrs)
        if m is None:
            return None
        b = (m.group(1) or "").strip()
        return [int(x) for x in b.split(",")] if b else []

    def f32arr(name):
        m = re.search(name + r"\s*=\s*array<f32(?::\s*([^>]*))?>", attrs)
        if m is None:
            return None
        b = (m.group(1) or "").strip()
        return [float(x) for x in b.split(",")] if b else []

    def i64(name):
        m = re.search(name + r"\s*=\s*(-?\d+)\s*:\s*i64", attrs)
        return int(m.group(1)) if m else None

    def s(name):
        m = re.search(name + r'\s*=\s*"([^"]*)"', attrs)
        return m.group(1) if m else None

    opcodes = i32arr("body_opcodes")
    in0, in1, iattr, fattr = (i32arr("body_in0"), i32arr("body_in1"),
                              i32arr("body_iattr"), f32arr("body_fattr"))
    body_out_id = i64("body_out_id")
    carry_arg_index = i64("carry_arg_index")
    start, stop, step = i64("start"), i64("stop"), i64("step")
    symbol = s("symbol")
    if (opcodes is None or in0 is None or body_out_id is None
            or carry_arg_index is None or start is None or stop is None
            or step is None or step == 0
            or symbol != "tessera_apple_gpu_run_graph_loop_f32"):
        return None

    rev = {v: k for k, v in GRAPH_OP.items()}
    body_ops: list = []
    for j, code in enumerate(opcodes):
        name = rev.get(code)
        if name is None:
            return None
        e: dict = {"op": name, "in0": in0[j]}
        if name == "matmul":
            e["in1"] = (in1 or [])[j]
            e["transpose_a"] = bool((iattr or [])[j] & 1)
            e["transpose_b"] = bool((iattr or [])[j] & 2)
        elif name in ("add", "sub", "mul", "div"):
            e["in1"] = (in1 or [])[j]
        elif name in ("rmsnorm", "layer_norm"):
            e["eps"] = (fattr or [1e-5])[j]
        body_ops.append(e)

    trip = (stop - start) // step
    arg_shapes = [tuple(np.asarray(a).shape) for a in arg_arrays]
    out_shape = tuple(np.asarray(arg_arrays[carry_arg_index]).shape)
    return run_graph_loop_f32(arg_arrays, arg_shapes, carry_arg_index, trip,
                              body_ops, body_out_id, out_shape)


def execute_control_if_mlir(mlir_text: str, arg_arrays: "list"):
    """MLIR-driven execution (Phase-G close-out C): given a lowered module with a
    `tessera_apple.gpu.control_if` op (carrying the then/else op-list payload +
    `out_shape`), read the op's attributes and dispatch via its recorded runtime
    `symbol` (`tessera_apple_gpu_run_graph_cond_f32`). Returns the selected
    branch result as an ``np.float32`` array, or ``None`` if the op/payload is
    absent or malformed."""
    import re

    import numpy as np

    if "tessera_apple.gpu.control_if" not in mlir_text:
        return None
    seg = mlir_text[mlir_text.index("tessera_apple.gpu.control_if"):]
    try:
        attrs = seg[seg.index("{"):seg.index("}") + 1]
    except ValueError:
        return None

    def i32arr(name):
        m = re.search(name + r"\s*=\s*array<i32(?::\s*([^>]*))?>", attrs)
        if m is None:
            return None
        b = (m.group(1) or "").strip()
        return [int(x) for x in b.split(",")] if b else []

    def f32arr(name):
        m = re.search(name + r"\s*=\s*array<f32(?::\s*([^>]*))?>", attrs)
        if m is None:
            return None
        b = (m.group(1) or "").strip()
        return [float(x) for x in b.split(",")] if b else []

    def i64arr(name):
        m = re.search(name + r"\s*=\s*array<i64(?::\s*([^>]*))?>", attrs)
        if m is None:
            return None
        b = (m.group(1) or "").strip()
        return [int(x) for x in b.split(",")] if b else []

    def i64(name):
        m = re.search(name + r"\s*=\s*(-?\d+)\s*:\s*i64", attrs)
        return int(m.group(1)) if m else None

    def s(name):
        m = re.search(name + r'\s*=\s*"([^"]*)"', attrs)
        return m.group(1) if m else None

    rev = {v: k for k, v in GRAPH_OP.items()}

    def branch(prefix):
        opcodes = i32arr(prefix + "_opcodes")
        in0 = i32arr(prefix + "_in0")
        in1, iattr, fattr = (i32arr(prefix + "_in1"), i32arr(prefix + "_iattr"),
                             f32arr(prefix + "_fattr"))
        out_id = i64(prefix + "_out_id")
        if opcodes is None or in0 is None or out_id is None:
            return None, None
        ops: list = []
        for j, code in enumerate(opcodes):
            name = rev.get(code)
            if name is None:
                return None, None
            e: dict = {"op": name, "in0": in0[j]}
            if name == "matmul":
                e["in1"] = (in1 or [])[j]
                e["transpose_a"] = bool((iattr or [])[j] & 1)
                e["transpose_b"] = bool((iattr or [])[j] & 2)
            elif name in ("add", "sub", "mul", "div"):
                e["in1"] = (in1 or [])[j]
            elif name in ("rmsnorm", "layer_norm"):
                e["eps"] = (fattr or [1e-5])[j]
            ops.append(e)
        return ops, out_id

    flag_arg_index = i64("flag_arg_index")
    out_shape = i64arr("out_shape")
    symbol = s("symbol")
    then_ops, then_out_id = branch("then")
    else_ops, else_out_id = branch("else")
    if (flag_arg_index is None or out_shape is None or then_ops is None
            or else_ops is None
            or symbol != "tessera_apple_gpu_run_graph_cond_f32"):
        return None

    arg_shapes = [tuple(np.asarray(a).shape) for a in arg_arrays]
    return run_graph_cond_f32(arg_arrays, arg_shapes, flag_arg_index,
                              then_ops, then_out_id, else_ops, else_out_id,
                              tuple(out_shape))


def execute_control_while_mlir(mlir_text: str, arg_arrays: "list"):
    """MLIR-driven execution (Phase-G close-out D): given a lowered module with a
    `tessera_apple.gpu.control_while` op (carrying the body+cond op-list payload),
    read the op's attributes and dispatch via its recorded runtime `symbol`
    (`tessera_apple_gpu_run_graph_while_f32`). Returns the final carry as an
    ``np.float32`` array, or ``None`` if the op/payload is absent or malformed."""
    import re

    import numpy as np

    if "tessera_apple.gpu.control_while" not in mlir_text:
        return None
    seg = mlir_text[mlir_text.index("tessera_apple.gpu.control_while"):]
    try:
        attrs = seg[seg.index("{"):seg.index("}") + 1]
    except ValueError:
        return None

    def i32arr(name):
        m = re.search(name + r"\s*=\s*array<i32(?::\s*([^>]*))?>", attrs)
        if m is None:
            return None
        b = (m.group(1) or "").strip()
        return [int(x) for x in b.split(",")] if b else []

    def f32arr(name):
        m = re.search(name + r"\s*=\s*array<f32(?::\s*([^>]*))?>", attrs)
        if m is None:
            return None
        b = (m.group(1) or "").strip()
        return [float(x) for x in b.split(",")] if b else []

    def i64(name):
        m = re.search(name + r"\s*=\s*(-?\d+)\s*:\s*i64", attrs)
        return int(m.group(1)) if m else None

    def s(name):
        m = re.search(name + r'\s*=\s*"([^"]*)"', attrs)
        return m.group(1) if m else None

    rev = {v: k for k, v in GRAPH_OP.items()}

    def branch(prefix):
        opcodes = i32arr(prefix + "_opcodes")
        in0 = i32arr(prefix + "_in0")
        in1, iattr, fattr = (i32arr(prefix + "_in1"), i32arr(prefix + "_iattr"),
                             f32arr(prefix + "_fattr"))
        out_id = i64(prefix + "_out_id")
        if opcodes is None or in0 is None or out_id is None:
            return None, None
        ops: list = []
        for j, code in enumerate(opcodes):
            name = rev.get(code)
            if name is None:
                return None, None
            e: dict = {"op": name, "in0": in0[j]}
            if name == "matmul":
                e["in1"] = (in1 or [])[j]
                e["transpose_a"] = bool((iattr or [])[j] & 1)
                e["transpose_b"] = bool((iattr or [])[j] & 2)
            elif name in ("add", "sub", "mul", "div"):
                e["in1"] = (in1 or [])[j]
            elif name in ("rmsnorm", "layer_norm"):
                e["eps"] = (fattr or [1e-5])[j]
            ops.append(e)
        return ops, out_id

    carry_arg_index = i64("carry_arg_index")
    max_iters = i64("max_iters")
    symbol = s("symbol")
    body_ops, body_out_id = branch("body")
    cond_ops, cond_out_id = branch("cond")
    if (carry_arg_index is None or max_iters is None or body_ops is None
            or cond_ops is None
            or symbol != "tessera_apple_gpu_run_graph_while_f32"):
        return None

    arg_shapes = [tuple(np.asarray(a).shape) for a in arg_arrays]
    carry_shape = tuple(np.asarray(arg_arrays[carry_arg_index]).shape)
    return run_graph_while_f32(arg_arrays, arg_shapes, carry_arg_index, max_iters,
                               body_ops, body_out_id, cond_ops, cond_out_id,
                               carry_shape)


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
    "AppleTensorBindingSpec",
    "AppleKernelBindingSpec",
    "compile_mlpackage",
    "author_matmul_package",
    "author_op_package",
    "author_chain_package",
    "author_graph_package",
    "run_graph_loop_f32",
    "run_graph_cond_f32",
    "run_graph_while_f32",
    "run_graph_loop_f16",
    "run_graph_cond_f16",
    "run_graph_while_f16",
    "run_graph_scan_f32",
    "execute_control_loop_mlir",
    "execute_control_if_mlir",
    "execute_control_while_mlir",
    "GRAPH_OP",
    "AUTHOR_CHAINS",
    "AUTHOR_OPS",
    "AUTHOR_OP_UNARY",
    "AUTHOR_OP_ROWOP",
    "AUTHOR_OP_BINARY",
    "first_function_name",
    "extract_argument_layout",
    "last_error_kind",
    "packaged_ml_available",
    "packaged_ml_skip_reason",
    "validate_bindings",
]
