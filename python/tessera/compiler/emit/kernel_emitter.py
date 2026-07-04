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
        self, region: Any, *, spec: SpecPolicy = SpecPolicy.BUCKET, dtype: str = "f32"
    ) -> KernelSource:
        """Emit a :class:`KernelSource` for ``region`` under ``spec``.

        Raise :class:`EmitError` if the region or policy is unsupported — never
        return a kernel specialized differently than requested.
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
    """Look up the emitter for ``target`` or raise a clear diagnostic."""
    try:
        return _EMITTERS[target]
    except KeyError:
        known = ", ".join(sorted(_EMITTERS)) or "<none registered>"
        raise EmitError(
            f"no KernelEmitter registered for target {target!r}; known: {known}"
        ) from None


def emit_kernel(
    region: Any, target: str, spec: SpecPolicy = SpecPolicy.BUCKET, *, dtype: str = "f32"
) -> KernelSource:
    """The plan's ``KernelEmitter.emit(region, target, spec)`` — dispatch ``region``
    to the emitter registered for ``target``."""
    return get_emitter(target).emit(region, spec=spec, dtype=dtype)


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


class KernelRunner(ABC):
    """Executes a synthesized fused region on probe inputs.

    Each method returns ``(output, execution)`` where ``execution`` is a backend
    tag (``"metal_runtime"`` when a real device kernel ran, ``"reference"`` /
    ``"fallback"`` when it did not). The oracle trusts the numpy reference unless
    a real kernel ran and *diverged*. Signatures accept ``*args, **kwargs`` so a
    backend's ``run_*`` gaining an optional knob is not an interface break; the
    documented positional args are what the oracles pass.
    """

    #: Backend identity, e.g. ``"apple_gpu"``. Subclasses set this.
    target: str = ""

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
