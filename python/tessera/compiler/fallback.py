"""Stable fallback-reason taxonomy — M3 deliverable.

Until M3, fallback reasons were free-text strings scattered across the
runtime, the canonical drivers, and the compile reports.  That made
"why did this run on numpy instead of MSL?" hard to assert on in tests
and impossible to diff between runs.

M3 centralizes the taxonomy:

  - :class:`FallbackReason` — frozen enum of stable codes.
  - :func:`message_for` — canonical human-readable text per code.
  - :class:`TesseraNativeRequiredError` — raised when the user passed
    ``native_required=True`` and a fallback would otherwise fire.

The :class:`tessera.compiler.compile_report.CompileReport.fallback_reason`
field accepts either ``None`` (native dispatch fired) or a
:class:`FallbackReason` value (which serializes to its ``.value`` for
JSON round-trips).  Free-form strings are still permitted for
backwards compatibility but tests can now assert against the enum
codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FallbackReason(str, Enum):
    """Canonical, JSON-friendly codes describing why a compile-report
    program did NOT run on its native path."""

    #: Host OS isn't macOS — Apple GPU runtime isn't available.
    NON_DARWIN_HOST = "non_darwin_host"

    #: macOS host but Metal initialization failed (no device, no queue,
    #: or shim couldn't be built).
    APPLE_GPU_RUNTIME_UNAVAILABLE = "apple_gpu_runtime_unavailable"

    #: Manifest has no fused / reference entry for ``(op, target)``.
    MANIFEST_MISS = "manifest_miss"

    #: Manifest entry exists but the dtype isn't covered.
    DTYPE_NOT_COVERED = "dtype_not_covered"

    #: Manifest entry exists but the shape envelope rejects the call
    #: (e.g., N > 8192 for the tiled matmul_softmax kernel).
    SHAPE_OUT_OF_ENVELOPE = "shape_out_of_envelope"

    #: Target capability registry says ``planned`` / ``artifact_only``
    #: rather than ``ready``.
    CAPABILITY_NOT_READY = "capability_not_ready"

    #: User explicitly disabled the native path (e.g., for A/B
    #: correctness checks).
    OPT_OUT = "opt_out"

    #: Reference path forced because compile-time gating chose it.
    REFERENCE_FORCED = "reference_forced"

    def message(self) -> str:
        """Canonical human-readable message for this reason."""
        return message_for(self)


_MESSAGES: dict[FallbackReason, str] = {
    FallbackReason.NON_DARWIN_HOST: (
        "non-Darwin host: Apple GPU runtime is not loadable; numpy "
        "reference path used"
    ),
    FallbackReason.APPLE_GPU_RUNTIME_UNAVAILABLE: (
        "Apple GPU runtime unavailable on this Darwin host (Metal "
        "init failed); numpy reference path used"
    ),
    FallbackReason.MANIFEST_MISS: (
        "backend manifest has no entry for this (op, target) pair; "
        "numpy reference path used"
    ),
    FallbackReason.DTYPE_NOT_COVERED: (
        "backend manifest entry does not cover the requested dtype; "
        "numpy reference path used"
    ),
    FallbackReason.SHAPE_OUT_OF_ENVELOPE: (
        "input shape falls outside the fused kernel's documented "
        "envelope; numpy reference path used"
    ),
    FallbackReason.CAPABILITY_NOT_READY: (
        "target capability registry reports this op is not ready "
        "on the requested target; numpy reference path used"
    ),
    FallbackReason.OPT_OUT: (
        "caller explicitly opted out of the native path"
    ),
    FallbackReason.REFERENCE_FORCED: (
        "reference path was forced by compile-time gating"
    ),
}


def message_for(reason: FallbackReason) -> str:
    """Canonical message for a fallback code.  ``KeyError`` is
    impossible by construction (every enum member is mapped)."""
    return _MESSAGES[reason]


class TesseraNativeRequiredError(RuntimeError):
    """Raised when ``native_required=True`` was set and the compiler
    would otherwise fall back to the reference path.

    Carries the stable :class:`FallbackReason` so callers can match
    on the code instead of the (mutable) human-readable text::

        try:
            f(x)
        except TesseraNativeRequiredError as exc:
            if exc.reason == FallbackReason.NON_DARWIN_HOST:
                pytest.skip("Apple GPU only")
            raise
    """

    def __init__(
        self,
        reason: FallbackReason,
        *,
        target: str = "",
        op_name: str = "",
        detail: str = "",
    ) -> None:
        self.reason = reason
        self.target = target
        self.op_name = op_name
        self.detail = detail
        prefix = f"native_required=True: "
        suffix = ""
        if target or op_name:
            scope = "/".join(s for s in (target, op_name) if s)
            suffix = f" [scope={scope}]"
        body = detail or reason.message()
        super().__init__(f"{prefix}{body}{suffix} [code={reason.value}]")


@dataclass(frozen=True)
class FallbackDecision:
    """Output of :func:`classify_fallback` — pairs a reason with the
    decision whether to surface it as an error or a degraded path."""
    reason: FallbackReason
    native_required: bool

    def raise_if_required(self, *, target: str = "", op_name: str = "") -> None:
        if self.native_required:
            raise TesseraNativeRequiredError(
                self.reason, target=target, op_name=op_name,
            )


def classify_host(
    *, is_darwin: bool, runtime_available: bool,
) -> FallbackReason | None:
    """Classify the host's ability to run native paths.

    Returns:
        ``None`` when the host can run native code,
        :attr:`FallbackReason.NON_DARWIN_HOST` on non-macOS,
        :attr:`FallbackReason.APPLE_GPU_RUNTIME_UNAVAILABLE` on
        Darwin where the Metal runtime failed to load.
    """
    if not is_darwin:
        return FallbackReason.NON_DARWIN_HOST
    if not runtime_available:
        return FallbackReason.APPLE_GPU_RUNTIME_UNAVAILABLE
    return None


__all__ = [
    "FallbackReason",
    "TesseraNativeRequiredError",
    "FallbackDecision",
    "classify_host",
    "message_for",
]
