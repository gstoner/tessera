"""Target-agnostic kernel-emitter protocol (COMPILER_REFACTOR_PLAN Workstream B2).

B1 split ``fusion.py`` into an arch-agnostic middle-end (``fusion_core``) and the
Apple MSL emitter (``apple_msl``). B2 lifts the emitter behind a *plugin protocol*
so a non-Apple backend (x86 clang, NVIDIA PTX, ROCm AMDGCN — Workstream C) reuses
the whole synthesizer by implementing one interface instead of forking the
synthesis code.

Three pieces, all target-independent:

* :class:`SpecPolicy` — the **specialization policy** ``static | bucket |
  dynamic`` (dynamic-shapes decision, 2026-07-02). ``bucket`` compiles one kernel
  per shape-bucket (seq-len / batch / KV-len) dispatched by runtime shape;
  ``dynamic`` (runtime-arg + guards) is the endgame. First impls emit ``bucket``;
  the *interface* is dynamic-ready now so ``dynamic`` drops in with no API break.
* :class:`KernelSource` — an emitted kernel: source text + entry-point name +
  language tag + the policy it was specialized under.
* :class:`KernelEmitter` — the plugin ABC. Each backend registers one
  target-bound emitter; :func:`emit_kernel` is the ``emit(region, target, spec)``
  entry point named in the plan, backed by the registry.

Apple MSL is the reference implementation (``emit.apple_msl.AppleMSLEmitter``),
relocated behind this interface — not rewritten.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # region types are only needed for hints — no runtime import.
    pass


class SpecPolicy(Enum):
    """How a kernel is specialized over shape (dynamic-shapes decision).

    ``STATIC`` — every dim is a compile-time constant (one kernel per exact
    shape). ``BUCKET`` — dims are quantized into buckets (seq-len / batch /
    KV-len); one kernel serves a bucket of runtime shapes, dispatched by shape.
    ``DYNAMIC`` — dims are runtime kernel args with in-kernel guards (one kernel
    serves all shapes). First impls emit ``BUCKET``; ``DYNAMIC`` is defined so the
    interface is stable when the guarded emitter lands.
    """

    STATIC = "static"
    BUCKET = "bucket"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class KernelSource:
    """One emitted kernel, ready for the backend's compile step.

    ``source`` is the kernel text (MSL / PTX / AMDGCN / C-LLVM); ``entry`` is the
    entry-point symbol the runtime launches; ``lang`` tags the source dialect so
    the compile step picks the right toolchain; ``spec`` records the policy it was
    specialized under; ``shape_key`` is the bucket key (``None`` for ``STATIC``).
    """

    source: str
    entry: str
    lang: str
    spec: SpecPolicy = SpecPolicy.STATIC
    shape_key: tuple[Any, ...] | None = None


def _dim_bucket(n: int) -> int:
    """Coarsen a dimension to its next power-of-two bucket, so one specialization
    generalizes across nearby shapes instead of memorizing every exact size.
    Matches ``emit.apple_msl._shape_bucket`` so the emitter's ``shape_key`` and
    the autotune corpus key agree on bucket boundaries."""
    b = 1
    while b < n:
        b *= 2
    return b


def bucket_key(
    dims: tuple[int, ...] | None,
    spec: SpecPolicy,
    *,
    dim_names: tuple[str, ...] | None = None,
) -> tuple[Any, ...] | None:
    """Compute the shape specialization key for ``dims`` under ``spec`` — the
    ``(op, shape-bucket, dtype, target)`` arbiter/cache keys on this (D1).

    * ``STATIC`` — the exact dims (one kernel per exact shape).
    * ``BUCKET`` — each dim coarsened to its power-of-two bucket, so one kernel
      serves a bucket of runtime shapes (seq-len / batch / KV-len).
    * ``DYNAMIC`` — the *symbolic* identity (``dim_names``), not the values: one
      kernel serves all shapes, so its key is shape-independent.

    For ``STATIC``/``BUCKET``, ``dims is None`` yields ``None`` (shape-anonymous —
    a fully static kernel whose shape is baked into the source needs no key). For
    ``DYNAMIC`` the key is the *symbolic* identity and is independent of concrete
    dims, so it is returned even when ``dims`` is omitted (an AOT symbolic kernel
    emitted without example dims) — otherwise distinct dynamic kernels would
    collapse to one anonymous key and the cache/arbiter could reuse the wrong
    entry."""
    if spec is SpecPolicy.DYNAMIC:
        return tuple(dim_names) if dim_names else ()
    if dims is None:
        return None
    if spec is SpecPolicy.STATIC:
        return tuple(dims)
    if spec is SpecPolicy.BUCKET:
        return tuple(_dim_bucket(d) for d in dims)
    raise ValueError(f"unknown SpecPolicy {spec!r}")  # pragma: no cover


class KernelEmitter(ABC):
    """A per-target kernel emitter plugin.

    An emitter is *bound* to one ``target`` (``"apple_gpu"``, ``"x86"``,
    ``"nvidia"``, ``"rocm"``) and one source ``lang``. It turns an arch-agnostic
    fused ``region`` (from ``fusion_core``) into a :class:`KernelSource`. This is
    the seam Workstream C backends implement; Apple MSL is the reference impl.

    The ``spec`` parameter is the specialization policy — present from day one so
    a later ``DYNAMIC`` implementation is not an API break. An emitter that does
    not yet support a requested policy must raise :class:`EmitError`, never
    silently emit a different specialization (Decision #21: no silent fallthrough).
    """

    #: Backend identity, e.g. ``"apple_gpu"``. Subclasses set this.
    target: str = ""
    #: Source dialect this emitter produces, e.g. ``"msl"``. Subclasses set this.
    lang: str = ""

    @abstractmethod
    def can_emit(self, region: Any) -> bool:
        """Whether this emitter can lower ``region`` (by region type/shape)."""

    @abstractmethod
    def emit(
        self,
        region: Any,
        *,
        spec: SpecPolicy = SpecPolicy.BUCKET,
        dtype: str = "f32",
        dims: tuple[int, ...] | None = None,
    ) -> KernelSource:
        """Emit a :class:`KernelSource` for ``region`` under ``spec``.

        ``dims`` is the concrete shape this call specializes for; the emitter
        records the corresponding :func:`bucket_key` in ``KernelSource.shape_key``
        (``None`` leaves it shape-anonymous). Raise :class:`EmitError` if the
        region or policy is unsupported — never return a kernel specialized
        differently than requested.
        """


class EmitError(RuntimeError):
    """A backend cannot emit the requested region/policy — names both so the
    caller can fall back or report (Decision #21: no silent no-op)."""


# --- registry: the plan's ``KernelEmitter.emit(region, target, spec)`` entry ---
_EMITTERS: dict[str, KernelEmitter] = {}


def register_emitter(emitter: KernelEmitter) -> None:
    """Register a target-bound emitter. Re-registering a target replaces it (the
    hook Workstream C uses to plug x86/NVIDIA/ROCm alongside Apple)."""
    if not emitter.target:
        raise ValueError("KernelEmitter.target must be a non-empty backend id")
    _EMITTERS[emitter.target] = emitter


def get_emitter(target: str) -> KernelEmitter:
    """Look up the emitter for ``target`` or raise a clear diagnostic.

    The built-in Apple reference emitter registers only as an import side effect
    of ``emit.apple_msl``. So a caller that reaches this registry through the
    public API (``emit_kernel``/``get_emitter``) without first importing the
    facade or ``apple_msl`` would otherwise see an empty registry. Bootstrap the
    Apple backend on demand so the advertised reference target is available
    regardless of import order (Workstream C backends register the same way)."""
    if target not in _EMITTERS and target in METAL_TARGETS:
        import tessera.compiler.emit.apple_msl  # noqa: F401 — self-registers
    try:
        return _EMITTERS[target]
    except KeyError:
        known = ", ".join(sorted(_EMITTERS)) or "<none registered>"
        raise EmitError(
            f"no KernelEmitter registered for target {target!r}; known: {known}"
        ) from None


def emit_kernel(
    region: Any,
    target: str,
    spec: SpecPolicy = SpecPolicy.BUCKET,
    *,
    dtype: str = "f32",
    dims: tuple[int, ...] | None = None,
) -> KernelSource:
    """The plan's ``KernelEmitter.emit(region, target, spec)`` — dispatch ``region``
    to the emitter registered for ``target``. ``dims`` (concrete shape) is threaded
    so the returned :class:`KernelSource` records its :func:`bucket_key`."""
    return get_emitter(target).emit(region, spec=spec, dtype=dtype, dims=dims)


#: Target aliases that mean "Apple GPU / Metal Shading Language" — shared by the
#: op-level ``emit(target)`` methods in ``fusion_core`` so a snippet request for
#: any of these resolves to the MSL body.
METAL_TARGETS = frozenset({"metal", "msl", "apple", "apple_gpu"})


# --- runner: the execute half of the seam (B2b) ------------------------------
#
# A KernelEmitter produces *source*; a KernelRunner *executes* a synthesized
# region on probe inputs and reports whether a real device kernel ran. The F4
# oracles in fusion_core (``verify_synthesized_*``) need a runner to produce the
# "actual" they compare against the numpy reference. B1 hard-wired that to a lazy
# ``import apple_msl``; B2b injects it through this registry so every backend
# wires the same oracle through its own runner (Decision #21 diagnostics; the
# per-backend explicit-injection into ``verify_*`` is Workstream B3).


class RunnerError(RuntimeError):
    """No KernelRunner is available to execute a synthesized region — names the
    gap so the caller can fall back or report (Decision #21: no silent no-op)."""


#: Execution tags that mean *no real device kernel ran* — the runner fell back to
#: the numpy reference. The F4 oracle trusts these by construction (there is
#: nothing device-emitted to distrust); ANY OTHER tag means a real kernel ran and
#: the oracle compares it to the numpy reference. A new backend therefore returns
#: its own real-execution tag (e.g. ``"x86_native"`` / ``"rocm_hip"`` / ``"cuda"``)
#: to get gated, and one of these when it declines — it does NOT need to pretend
#: to be Metal. This is what makes the F4 gate backend-agnostic (B3).
REFERENCE_EXECUTIONS = frozenset({"reference", "fallback"})


class KernelRunner(ABC):
    """Executes a synthesized fused region on probe inputs.

    Each method returns ``(output, execution)`` where ``execution`` is a backend
    tag: a real-execution tag (``"metal_runtime"``, ``"x86_native"``, …) when a
    real device kernel ran, or a tag in :data:`REFERENCE_EXECUTIONS`
    (``"reference"`` / ``"fallback"``) when it fell back to numpy. The F4 oracle
    compares to the numpy reference iff a real kernel ran, and trusts the
    reference otherwise. Signatures accept ``*args, **kwargs`` so a backend's
    ``run_*`` gaining an optional knob is not an interface break; the documented
    positional args are what the oracles pass.
    """

    #: Backend identity, e.g. ``"apple_gpu"``. Subclasses set this.
    target: str = ""

    #: Precision budget: the largest absolute error this backend's *correct*
    #: kernel may show vs the f32 numpy reference, for the F4 oracle to treat as
    #: "right" rather than "buggy". ``None`` = use the oracle's default (an f32 /
    #: exact backend, e.g. Apple, x86). A half-precision lead backend (ROCm f16
    #: WMMA / flash-attn) sets a looser value so f16 rounding is not misread as a
    #: miscompile — while an O(1) miscompile is still caught. This is the simplest
    #: slice of the accuracy-budgeted arbiter (Decision #28 / plan D2); the oracle
    #: takes ``max(caller_atol, accuracy_atol)``.
    accuracy_atol: float | None = None

    @abstractmethod
    def run_fused_region(self, region: Any, *args: Any, **kwargs: Any) -> tuple[Any, str]:
        """Run a matmul-epilogue region on ``(A, B, bias=None, ...)``."""

    @abstractmethod
    def run_fused_attention(self, region: Any, *args: Any, **kwargs: Any) -> tuple[Any, str]:
        """Run an attention block on ``(Q, K, V)``."""

    @abstractmethod
    def run_gated_matmul_region(self, region: Any, *args: Any, **kwargs: Any) -> tuple[Any, str]:
        """Run a gated-matmul region on ``(A, Wg, Wu)``."""

    @abstractmethod
    def run_pointwise_graph(self, region: Any, *args: Any, **kwargs: Any) -> tuple[Any, str]:
        """Run a pointwise-DAG region on ``(arrays)`` (a list of probes)."""


_RUNNERS: dict[str, KernelRunner] = {}
_DEFAULT_RUNNER_TARGET: str | None = None


def register_runner(runner: KernelRunner, *, default: bool | None = None) -> None:
    """Register a target-bound runner. The first registered runner becomes the
    active default; pass ``default=True`` to make a later one active (the hook the
    arbiter/Workstream C use to swap execution backends)."""
    global _DEFAULT_RUNNER_TARGET
    if not runner.target:
        raise ValueError("KernelRunner.target must be a non-empty backend id")
    _RUNNERS[runner.target] = runner
    if default or (_DEFAULT_RUNNER_TARGET is None and default is not False):
        _DEFAULT_RUNNER_TARGET = runner.target


def get_runner(target: str) -> KernelRunner:
    """Look up the runner for ``target`` or raise a clear diagnostic."""
    try:
        return _RUNNERS[target]
    except KeyError:
        known = ", ".join(sorted(_RUNNERS)) or "<none registered>"
        raise RunnerError(
            f"no KernelRunner registered for target {target!r}; known: {known}"
        ) from None


def active_runner() -> KernelRunner | None:
    """The default runner (first registered / explicitly defaulted), or ``None``
    if nothing has registered yet."""
    if _DEFAULT_RUNNER_TARGET is None:
        return None
    return _RUNNERS.get(_DEFAULT_RUNNER_TARGET)
