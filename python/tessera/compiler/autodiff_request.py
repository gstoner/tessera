"""The public differentiation *request* — Phase 1 of the autodiff unification.

Phase 1 makes differentiation an explicit compiler request rather than an
implicit Python-tape behavior (see
``docs/audit/compiler/AUTODIFF_UNIFICATION_PLAN.md`` §4 Phase 1).

Python owns validation, source spans, and user-facing diagnostics; it emits a
canonical ``tessera.autodiff = "reverse"|"forward"`` intent plus stable
``wrt_indices`` into Graph IR. The C++ reverse or forward pass owns the program
transform. Python does **not** reimplement either transform here.

The mode-neutral differentiation provenance facet distinguishes IR-transformed /
artifact-only / native-executable. The legacy backward facet remains a
reverse-only compatibility alias. Native reverse resolution is sourced from
exact-target device proof; forward/JVP remains IR-transformed until separately
proven target rows exist.
"""

from __future__ import annotations

import enum
import inspect
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

# Reuse the canonical autodiff error so front-end diagnostics are one type.
from ..autodiff.tape import TesseraAutodiffError

#: Public modes. ``jvp`` is an ergonomic spelling of the compiler's canonical
#: ``forward`` intent, not a third transformation mode.
SUPPORTED_MODES: frozenset[str] = frozenset({"forward", "jvp", "reverse"})
_MODE_ALIASES = {"jvp": "forward"}

# Production-side mirror of the ODS TangentInterface set used for decoration-
# time provenance. The compiler remains authoritative; test_autodiff_ledger.py
# requires exact equality with TangentInterface.cpp so this selector cannot
# silently drift or over-claim an IR-transformed program.
COMPILER_FORWARD_FAMILIES: frozenset[str] = frozenset(
    {
        "add", "all_gather", "all_reduce", "all_to_all", "broadcast", "dct",
        "depth_attn", "dropout", "es_low_rank_correction", "expand", "fft", "flatten",
        "ifft", "irfft", "layer_norm", "matmul", "mul", "permute", "reduce",
        "reduce_scatter", "reshape", "rfft", "rmsnorm", "sigmoid", "softmax", "squeeze",
        "spectral_conv", "spectral_filter", "stft", "istft", "stop_gradient",
        "sub", "tanh", "transpose", "unsqueeze", "view",
    }
)


class DifferentiationStatus(str, enum.Enum):
    """The compiler answer for the requested differentiation mode."""

    NOT_REQUESTED = "not_requested"       # no autodiff= on this @jit
    UNSUPPORTED = "unsupported"           # requested but cannot be honored
    IR_TRANSFORMED = "ir_transformed"     # compiler emitted the requested derivative IR
    ARTIFACT_ONLY = "artifact_only"       # backward artifacts exist but do not execute
    NATIVE_EXECUTABLE = "native_executable"  # backward runs natively (Phase 4+)


@dataclass(frozen=True)
class DifferentiationRequest:
    """A validated, decoration-time differentiation request."""

    mode: str
    wrt: tuple[str, ...]
    wrt_indices: tuple[int, ...]
    native_required: bool = False

    def module_intent_attrs(self) -> dict[str, str]:
        """The Graph IR *module* attributes carrying this intent, as raw MLIR
        attribute strings (values already quoted/bracketed for emission)."""
        wrt_list = ", ".join(f'"{name}"' for name in self.wrt)
        return {
            "tessera.autodiff": f'"{self.mode}"',
            "tessera.autodiff.wrt": f"[{wrt_list}]",
            "tessera.autodiff.wrt_indices": (
                "[" + ", ".join(str(index) for index in self.wrt_indices) + "]"
            ),
        }


@dataclass(frozen=True)
class DifferentiationProvenance:
    """The typed mode-neutral differentiation facet on a compile answer.

    ``reason`` is empty when ``status`` is a satisfied state; otherwise it leads
    with a stable, source-oriented explanation (Decision #21 — never a silent
    no-op)."""

    status: DifferentiationStatus
    mode: Optional[str] = None
    wrt: tuple[str, ...] = ()
    native_required: bool = False
    reason: str = ""

    @property
    def requested(self) -> bool:
        return self.status is not DifferentiationStatus.NOT_REQUESTED

    @property
    def native(self) -> bool:
        return self.status is DifferentiationStatus.NATIVE_EXECUTABLE


#: The sentinel for "no differentiation requested".
NOT_REQUESTED = DifferentiationProvenance(DifferentiationStatus.NOT_REQUESTED)

# Compatibility aliases for callers that consume the established reverse-only
# names. New code should use the mode-neutral spellings above.
BackwardStatus = DifferentiationStatus
BackwardProvenance = DifferentiationProvenance


def _signature_params(fn: Callable) -> tuple[str, ...]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C funcs — no introspectable sig
        return ()
    return tuple(
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY)
    )


def build_request(
    fn: Callable,
    *,
    autodiff: Optional[str],
    wrt: Optional[Sequence[str]],
    native_required: bool = False,
) -> Optional[DifferentiationRequest]:
    """Validate an ``@jit(autodiff=..., wrt=...)`` request at decoration time.

    Returns ``None`` when no differentiation was requested. Raises
    ``TesseraAutodiffError`` with a clear message for an unsupported mode or a
    ``wrt`` name that is not a parameter of ``fn``.
    """
    if autodiff is None:
        if wrt is not None:
            raise TesseraAutodiffError(
                f"@jit(wrt={tuple(wrt)!r}) was given for {fn.__name__!r} without "
                f"autodiff=... — `wrt` only means something when a "
                f"differentiation mode is requested. Pass "
                f"autodiff=\"reverse\" or drop `wrt`."
            )
        return None

    mode = str(autodiff).lower()
    if mode not in SUPPORTED_MODES:
        raise TesseraAutodiffError(
            f"@jit(autodiff={autodiff!r}) on {fn.__name__!r}: unknown "
            f"differentiation mode. Supported: {sorted(SUPPORTED_MODES)}."
        )

    mode = _MODE_ALIASES.get(mode, mode)
    params = _signature_params(fn)
    if wrt is None:
        # Differentiate w.r.t. every parameter by default.
        resolved_wrt = params
        if not resolved_wrt:
            raise TesseraAutodiffError(
                f"@jit(autodiff={autodiff!r}) on {fn.__name__!r}: no "
                f"differentiable parameters found and no explicit `wrt=` given."
            )
    else:
        resolved_wrt = tuple(str(name) for name in wrt)
        if not resolved_wrt:
            raise TesseraAutodiffError(
                f"@jit(autodiff={autodiff!r}, wrt=()) on {fn.__name__!r}: "
                f"`wrt` is empty — name at least one parameter to differentiate."
            )
        if params:  # only validate when we could introspect the signature
            unknown = [name for name in resolved_wrt if name not in params]
            if unknown:
                raise TesseraAutodiffError(
                    f"@jit(autodiff={autodiff!r}, wrt={resolved_wrt!r}) on "
                    f"{fn.__name__!r}: {unknown!r} not in the function's "
                    f"parameters {list(params)!r}."
                )
        if len(set(resolved_wrt)) != len(resolved_wrt):
            raise TesseraAutodiffError(
                f"@jit(autodiff={autodiff!r}, wrt={resolved_wrt!r}) on "
                f"{fn.__name__!r}: `wrt` contains duplicate parameter names."
            )

    if not params:
        raise TesseraAutodiffError(
            f"@jit(autodiff={autodiff!r}) on {fn.__name__!r}: the function "
            "signature is not introspectable, so stable `wrt_indices` cannot "
            "be derived. Wrap it in a Python function with named parameters."
        )

    wrt_indices = tuple(params.index(name) for name in resolved_wrt) if params else ()
    return DifferentiationRequest(
        mode=mode,
        wrt=resolved_wrt,
        wrt_indices=wrt_indices,
        native_required=bool(native_required),
    )


def resolve_backward_provenance(
    request: Optional[DifferentiationRequest],
    *,
    has_native_backward: bool = False,
    target: Optional[str] = None,
    op_families: Sequence[str] = (),
) -> BackwardProvenance:
    """Resolve the backward facet for a request against today's reality.

    ``has_native_backward`` is the hook Phase 4 flips per (op-family, target)
    once a real backward launch ABI exists.

    Phase 4 (A3): when ``op_families`` + ``target`` are given, the hook is
    **sourced from the runtime execution matrix** — a program's backward is
    native iff *every* differentiable component op has a native (device-proven)
    backward launch on ``target`` (``execution_matrix.has_native_backward``). So
    ``@jit(autodiff="reverse", target="rocm")`` over a covered family (e.g.
    ``flash_attn`` on gfx1151) resolves to ``NATIVE_EXECUTABLE`` and a
    ``native_required`` request is honored; everything else still resolves
    honestly (``IR_TRANSFORMED`` / ``UNSUPPORTED``), never a false native claim.
    An explicit ``has_native_backward=True`` still forces native (test hook).
    """
    if request is None:
        return NOT_REQUESTED
    if request.mode != "reverse":
        raise ValueError(
            "resolve_backward_provenance requires a reverse-mode request; "
            "use resolve_differentiation_provenance for forward/JVP"
        )

    if not has_native_backward and op_families and target is not None:
        # Lazy import — execution_matrix is light and does not import this module.
        from tessera.compiler import execution_matrix as _em
        has_native_backward = all(
            _em.has_native_backward(fam, target) for fam in op_families
        )

    def _prov(status: BackwardStatus, reason: str = "") -> BackwardProvenance:
        return BackwardProvenance(
            status=status,
            mode=request.mode,
            wrt=request.wrt,
            native_required=request.native_required,
            reason=reason,
        )

    if has_native_backward:
        return _prov(BackwardStatus.NATIVE_EXECUTABLE)

    if request.native_required:
        where = f" on target {target!r}" if target else ""
        return _prov(
            BackwardStatus.UNSUPPORTED,
            reason=(
                f"native_required=True but no native backward execution path "
                f"exists{where} with exact-target device verification. "
                f"See autodiff_connection_ledger.md for the live proof axes. "
                f"Drop native_required to accept the IR-transformed path."
            ),
        )

    return _prov(
        BackwardStatus.IR_TRANSFORMED,
        reason=(
            "intent emitted (`tessera.autodiff`); the AutodiffPass builds "
            "backward IR, but it does not execute natively yet (Phase 3/4)."
        ),
    )


def resolve_differentiation_provenance(
    request: Optional[DifferentiationRequest],
    *,
    has_native_differentiation: bool = False,
    target: Optional[str] = None,
    op_families: Sequence[str] = (),
) -> DifferentiationProvenance:
    """Resolve either reverse or forward/JVP provenance without conflating it.

    Reverse mode delegates to the established exact-target backward matrix.
    Forward mode currently proves a compiler Graph-IR transform, not a native
    target package. Its native hook is explicit until target execution-matrix
    rows gain a JVP phase.
    """
    if request is None:
        return NOT_REQUESTED
    if request.mode == "reverse":
        return resolve_backward_provenance(
            request,
            has_native_backward=has_native_differentiation,
            target=target,
            op_families=op_families,
        )

    def _prov(
        status: DifferentiationStatus, reason: str = ""
    ) -> DifferentiationProvenance:
        return DifferentiationProvenance(
            status=status,
            mode=request.mode,
            wrt=request.wrt,
            native_required=request.native_required,
            reason=reason,
        )

    unsupported = tuple(
        family for family in op_families
        if family.removeprefix("tessera.") not in COMPILER_FORWARD_FAMILIES
    )
    if unsupported:
        return _prov(
            DifferentiationStatus.UNSUPPORTED,
            reason=(
                "compiler forward mode has no TangentInterface for active "
                f"families {unsupported!r}; the request fails closed."
            ),
        )
    if has_native_differentiation:
        return _prov(DifferentiationStatus.NATIVE_EXECUTABLE)
    if request.native_required:
        where = f" on target {target!r}" if target else ""
        return _prov(
            DifferentiationStatus.UNSUPPORTED,
            reason=(
                "native_required=True but no native JVP execution path exists"
                f"{where} with exact-target numerical proof. The compiler "
                "Graph-IR JVP is available; drop native_required to accept it."
            ),
        )
    return _prov(
        DifferentiationStatus.IR_TRANSFORMED,
        reason=(
            "intent emitted (`tessera.autodiff = forward`); "
            "--tessera-autodiff-forward emits the paired JVP Graph function, "
            "but no native target JVP package is proven yet."
        ),
    )
