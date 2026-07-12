"""The public differentiation *request* — Phase 1 of the autodiff unification.

Phase 1 makes differentiation an explicit compiler request rather than an
implicit Python-tape behavior (see
``docs/audit/compiler/AUTODIFF_UNIFICATION_PLAN.md`` §4 Phase 1).

Python owns validation, source spans, and user-facing diagnostics; it emits the
``tessera.autodiff = "reverse"`` *intent* into the Graph IR module (the attribute
the C++ ``--tessera-autodiff`` pass keys on — verified end-to-end by
``tests/tessera-ir/phase_f4/autodiff_pass_smoke.mlir``). It does **not**
reimplement backward execution here.

The *backward provenance facet* mirrors the forward compile answer: it
distinguishes IR-transformed / artifact-only / native-executable, so a caller can
tell whether gradients merely have IR, or actually execute natively. Native
resolution is sourced from exact-target device proof in the runtime execution
matrix; an executable row alone is insufficient. The generated autodiff ledger
shows the live per-family evidence.
"""

from __future__ import annotations

import enum
import inspect
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

# Reuse the canonical autodiff error so front-end diagnostics are one type.
from ..autodiff.tape import TesseraAutodiffError

#: Differentiation modes accepted today. Reverse-mode first (plan Phase 1);
#: forward/JVP is a deliberate later slice and is rejected with a clear
#: diagnostic rather than silently ignored.
SUPPORTED_MODES: frozenset[str] = frozenset({"reverse"})
_PLANNED_MODES: frozenset[str] = frozenset({"forward", "jvp"})


class BackwardStatus(str, enum.Enum):
    """The backward compile answer, mirroring the forward executability facet."""

    NOT_REQUESTED = "not_requested"       # no autodiff= on this @jit
    UNSUPPORTED = "unsupported"           # requested but cannot be honored
    IR_TRANSFORMED = "ir_transformed"     # intent emitted; AutodiffPass builds backward IR
    ARTIFACT_ONLY = "artifact_only"       # backward artifacts exist but do not execute
    NATIVE_EXECUTABLE = "native_executable"  # backward runs natively (Phase 4+)


@dataclass(frozen=True)
class DifferentiationRequest:
    """A validated, decoration-time differentiation request."""

    mode: str
    wrt: tuple[str, ...]
    native_required: bool = False

    def module_intent_attrs(self) -> dict[str, str]:
        """The Graph IR *module* attributes carrying this intent, as raw MLIR
        attribute strings (values already quoted/bracketed for emission)."""
        wrt_list = ", ".join(f'"{name}"' for name in self.wrt)
        return {
            "tessera.autodiff": f'"{self.mode}"',
            "tessera.autodiff.wrt": f"[{wrt_list}]",
        }


@dataclass(frozen=True)
class BackwardProvenance:
    """The typed backward facet attached to a compile answer.

    ``reason`` is empty when ``status`` is a satisfied state; otherwise it leads
    with a stable, source-oriented explanation (Decision #21 — never a silent
    no-op)."""

    status: BackwardStatus
    mode: Optional[str] = None
    wrt: tuple[str, ...] = ()
    native_required: bool = False
    reason: str = ""

    @property
    def requested(self) -> bool:
        return self.status is not BackwardStatus.NOT_REQUESTED

    @property
    def native(self) -> bool:
        return self.status is BackwardStatus.NATIVE_EXECUTABLE


#: The sentinel for "no differentiation requested".
NOT_REQUESTED = BackwardProvenance(BackwardStatus.NOT_REQUESTED)


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
        if mode in _PLANNED_MODES:
            raise TesseraAutodiffError(
                f"@jit(autodiff={autodiff!r}) on {fn.__name__!r}: forward/JVP "
                f"differentiation is planned but not in this slice — Phase 1 is "
                f"reverse-mode only. Use autodiff=\"reverse\"."
            )
        raise TesseraAutodiffError(
            f"@jit(autodiff={autodiff!r}) on {fn.__name__!r}: unknown "
            f"differentiation mode. Supported: {sorted(SUPPORTED_MODES)}."
        )

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

    return DifferentiationRequest(
        mode=mode, wrt=resolved_wrt, native_required=bool(native_required)
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
