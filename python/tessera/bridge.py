"""``tessera.bridge`` — explicit boundary ops between value kinds.

M2 mandates that mixed-kind boundaries (tensor ↔ multivector ↔
complex) are crossed through **explicit ops**, not through private
attribute reads.  This module is the join point.

Decision #15a (locked in `ga_scope_lock.md` Q2): the value kind is
a distinct entity, not a tensor attribute.  ``tessera.bridge``
exposes the kind via a function (:func:`value_kind_of`) rather
than an attribute, so the structural shape of the API enforces the
decision — any code that needs to ask "what kind is this?" goes
through the bridge.

Ops shipped today:

  - :func:`multivector_to_tensor` — Cl(p,q,r) Multivector → numpy
    coefficients (shape ``(..., 2^(p+q+r))``).  Equivalent to
    reading ``mv._coefficients`` but explicit + auditable.
  - :func:`tensor_to_multivector` — inverse: numpy coefficients +
    algebra → Multivector.
  - :func:`complex_to_tensor` / :func:`tensor_to_complex` —
    same pattern for the M7 ``ComplexScalar`` sibling kind.
  - :func:`value_kind_of` — classify an arbitrary value into
    ``"tensor"`` / ``"multivector"`` / ``"complex"`` / ``"unknown"``.

Future ops (gated on Phase G / H): GPU-side boundary kernels that
avoid the host roundtrip when the producer + consumer ops both
ship fused kernels on the same target.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Re-exported value-kind constants — these match
# :mod:`tessera.compiler.compile_report`'s normative set.
VALUE_KIND_TENSOR = "tensor"
VALUE_KIND_MULTIVECTOR = "multivector"
VALUE_KIND_COMPLEX = "complex"
VALUE_KIND_UNKNOWN = "unknown"


class TesseraValueKindError(TypeError):
    """Raised when a value kind doesn't match what the bridge op
    requires (e.g., feeding a numpy array into
    :func:`multivector_to_tensor`).

    Carries the offending source span when known so downstream
    callers can surface it in diagnostics.
    """

    def __init__(
        self,
        message: str,
        *,
        expected_kind: str = "",
        actual_kind: str = "",
        source_span: tuple[int, int] | None = None,
    ) -> None:
        self.expected_kind = expected_kind
        self.actual_kind = actual_kind
        self.source_span = source_span
        prefix = "[M2_VALUE_KIND_MISMATCH] "
        scope = ""
        if expected_kind or actual_kind:
            scope = (
                f" (expected={expected_kind!r}, actual={actual_kind!r})"
            )
        loc = ""
        if source_span is not None:
            loc = f" at line {source_span[0]}, col {source_span[1]}"
        super().__init__(f"{prefix}{message}{scope}{loc}")


# ─────────────────────────────────────────────────────────────────────────────
# Multivector ↔ tensor boundary
# ─────────────────────────────────────────────────────────────────────────────

def multivector_to_tensor(mv: Any) -> np.ndarray:
    """Extract the coefficients of a :class:`tessera.ga.Multivector`
    as a numpy array.

    This is the explicit form of the otherwise-private
    ``mv._coefficients`` read — Decision #15a forbids the private
    read because it hides the kind boundary.
    """
    from tessera.ga import Multivector
    if not isinstance(mv, Multivector):
        raise TesseraValueKindError(
            "multivector_to_tensor requires a Multivector",
            expected_kind=VALUE_KIND_MULTIVECTOR,
            actual_kind=value_kind_of(mv),
        )
    return np.asarray(mv._coefficients).copy()


def tensor_to_multivector(coefficients: Any, algebra: Any) -> Any:
    """Wrap a numpy coefficient array as a Multivector under the
    given Clifford algebra.

    The reverse boundary op.  Validates that ``coefficients`` has
    the right trailing-axis length for the algebra (``2^(p+q+r)``).
    """
    from tessera.ga import Multivector
    arr = np.asarray(coefficients)
    expected = 2 ** (algebra.p + algebra.q + algebra.r)
    if arr.shape[-1] != expected:
        raise TesseraValueKindError(
            f"tensor_to_multivector: coefficients trailing axis "
            f"{arr.shape[-1]} != 2^(p+q+r) = {expected} for algebra "
            f"{algebra}",
        )
    return Multivector(arr.astype(np.float32, copy=False), algebra)


# ─────────────────────────────────────────────────────────────────────────────
# ComplexScalar ↔ tensor boundary (M7 surface)
# ─────────────────────────────────────────────────────────────────────────────

def complex_to_tensor(z: Any) -> np.ndarray:
    """Pack a :class:`tessera.complex.ComplexScalar` into a numpy
    array with a trailing axis of length 2 (``[re, im]``)."""
    from tessera.complex import ComplexScalar
    if not isinstance(z, ComplexScalar):
        raise TesseraValueKindError(
            "complex_to_tensor requires a ComplexScalar",
            expected_kind=VALUE_KIND_COMPLEX,
            actual_kind=value_kind_of(z),
        )
    return np.stack([z.re, z.im], axis=-1)


def tensor_to_complex(arr: Any) -> Any:
    """Inverse of :func:`complex_to_tensor` — trailing axis of
    length 2 → :class:`ComplexScalar`."""
    from tessera.complex import ComplexScalar
    arr = np.asarray(arr)
    if arr.shape[-1] != 2:
        raise TesseraValueKindError(
            f"tensor_to_complex: trailing axis must be 2 (real, imag); "
            f"got {arr.shape[-1]}",
        )
    return ComplexScalar(arr[..., 0].copy(), arr[..., 1].copy())


# ─────────────────────────────────────────────────────────────────────────────
# Kind classifier — Decision #15a enforcement point
# ─────────────────────────────────────────────────────────────────────────────

def value_kind_of(x: Any) -> str:
    """Classify a value into one of the canonical value-kind
    strings.  ``"unknown"`` for anything we don't recognize.

    This is intentionally a **function**, not an attribute read on
    ``x`` — per Decision #15a, value kind is a distinct sibling
    concept, not a property hanging off a tensor.  Code that
    branches on kind must call this function so the boundary is
    auditable.
    """
    # Multivector — GA sibling kind.
    try:
        from tessera.ga import Multivector
        if isinstance(x, Multivector):
            return VALUE_KIND_MULTIVECTOR
    except ImportError:  # pragma: no cover — GA always present
        pass
    # ComplexScalar — M7 sibling kind.
    try:
        from tessera.complex import ComplexScalar
        if isinstance(x, ComplexScalar):
            return VALUE_KIND_COMPLEX
    except ImportError:  # pragma: no cover
        pass
    # Anything numpy-shaped is tensor.  Plain python scalars too.
    if isinstance(x, np.ndarray):
        return VALUE_KIND_TENSOR
    if isinstance(x, (int, float, np.number)):
        return VALUE_KIND_TENSOR
    return VALUE_KIND_UNKNOWN


def assert_value_kind(
    x: Any,
    *expected: str,
    op_name: str = "",
    source_span: tuple[int, int] | None = None,
) -> None:
    """Raise :class:`TesseraValueKindError` if ``value_kind_of(x)``
    isn't in ``expected``.

    Use at the call sites of GA / complex ops that require a
    specific sibling kind.  ``op_name`` shows up in the error
    message so the user sees which op rejected which value.
    """
    actual = value_kind_of(x)
    if actual not in expected:
        op_prefix = f"{op_name}: " if op_name else ""
        raise TesseraValueKindError(
            f"{op_prefix}value kind mismatch",
            expected_kind="|".join(expected),
            actual_kind=actual,
            source_span=source_span,
        )


def check_call_kinds(
    args: tuple[Any, ...],
    expected: tuple[str, ...] | str,
    *,
    op_name: str,
    arg_names: tuple[str, ...] = (),
    source_span: tuple[int, int] | None = None,
) -> None:
    """Validate the value kinds of a call's positional arguments.

    Iterates over ``args`` and raises
    :class:`TesseraValueKindError` at the first kind mismatch.
    The exception carries the offending argument's position +
    name (when available) so the diagnostic is actionable.

    Parameters
    ----------
    args
        The positional arguments to the call.
    expected
        Either a single kind string (all args must match) or a
        tuple of per-arg kinds.  For per-arg kinds, ``len(expected)``
        must equal ``len(args)``.
    op_name
        Human-readable op name for the diagnostic.
    arg_names
        Optional positional argument names — same length as
        ``args`` when provided.  Used in the error text.
    source_span
        Optional ``(line, col)`` of the call site.
    """
    if isinstance(expected, str):
        per_arg = tuple(expected for _ in args)
    else:
        if len(expected) != len(args):
            raise ValueError(
                f"check_call_kinds: expected length {len(expected)} "
                f"≠ args length {len(args)}"
            )
        per_arg = tuple(expected)
    for i, (arg, want) in enumerate(zip(args, per_arg)):
        actual = value_kind_of(arg)
        if actual != want:
            name = arg_names[i] if i < len(arg_names) else f"arg[{i}]"
            raise TesseraValueKindError(
                f"{op_name}: argument {name} has wrong value kind",
                expected_kind=want,
                actual_kind=actual,
                source_span=source_span,
            )


__all__ = [
    "VALUE_KIND_TENSOR",
    "VALUE_KIND_MULTIVECTOR",
    "VALUE_KIND_COMPLEX",
    "VALUE_KIND_UNKNOWN",
    "TesseraValueKindError",
    "multivector_to_tensor",
    "tensor_to_multivector",
    "complex_to_tensor",
    "tensor_to_complex",
    "value_kind_of",
    "assert_value_kind",
    "check_call_kinds",
]
