"""M7 follow-up — symbolic Cauchy-Riemann verifier.

Numerical CR (in :mod:`tessera.complex`) probes a function at
random points and finite-differences.  This module ships the
**symbolic** counterpart: lower a Python complex function to an
IR (re-using the :mod:`tessera.compiler.ast_ir` core), walk the
IR, and classify each op as holomorphic or non-holomorphic.

The structural theorem the verifier uses:

  * A composition of holomorphic complex functions is holomorphic.
  * If any op in the function's IR is non-holomorphic (e.g.,
    ``complex_conjugate``, ``complex_abs``), the whole function
    is non-holomorphic.

Unlike the numerical verifier, this catches non-analyticity at
**compile time** — no probing, no false-pass risk near a
flat-but-non-analytic point.

Architecture (P3 reviewer correction satisfied):

  * The whitelist below is **type-specialized** to complex ops —
    not the M6 energy whitelist.  This is the right shape: M6's
    template (AST→IR + per-op rule table) is reused, but the
    rule table is rewritten for the complex/conformal surface.
  * Reuse is **structural** (the AST→IR machinery), not
    **type-specific** (the rules).  The plan flagged this
    explicitly.

Public API:

  * :func:`lower_complex_function(fn) -> ComplexIRProgram`
  * :func:`is_holomorphic(ir_or_fn) -> HolomorphicReport`
  * :func:`analytic_symbolic` — decorator that raises
    :class:`NotHolomorphicError` at decoration time when the
    function's IR contains a non-holomorphic op.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from . import ast_ir as _ast_ir


# ─────────────────────────────────────────────────────────────────────────────
# Whitelist + holomorphicity classification.
#
# Every op here is recognized by :func:`lower_complex_function`.
# The bool flags it as "holomorphic in z" (HOL) or not.
# ─────────────────────────────────────────────────────────────────────────────

_COMPLEX_ATTR_TO_OP_NAME: Mapping[str, str] = {
    # Holomorphic complex ops.
    "complex_mul":   "complex_mul",
    "complex_exp":   "complex_exp",
    "complex_div":   "complex_div",
    "mobius":        "mobius",
    # NON-holomorphic — these break analyticity.
    "complex_conjugate": "complex_conjugate",
    "complex_abs":       "complex_abs",
    # Real-valued / structural — also non-holomorphic in the
    # complex sense (output isn't a complex function of z).
    "stereographic":           "stereographic",
    "stereographic_inverse":   "stereographic_inverse",
    "conformal_jacobian":      "conformal_jacobian",
    "laplacian_2d":            "laplacian_2d",
}


HOLOMORPHIC_OPS: frozenset[str] = frozenset({
    "complex_mul",
    "complex_exp",
    "complex_div",
    "mobius",
})

NON_HOLOMORPHIC_OPS: frozenset[str] = frozenset({
    "complex_conjugate",
    "complex_abs",
    "stereographic",
    "stereographic_inverse",
    "conformal_jacobian",
    "laplacian_2d",
})


# ─────────────────────────────────────────────────────────────────────────────
# Error + report types
# ─────────────────────────────────────────────────────────────────────────────

class ComplexJitError(Exception):
    """Raised by :func:`lower_complex_function` when the AST falls
    outside the v1 grammar (non-complex callees, unsupported
    expression shapes, etc.)."""


class NotHolomorphicError(ComplexJitError):
    """Raised by :func:`analytic_symbolic` when the IR contains a
    non-holomorphic op.  Names the offending op + line/col.

    Carries the stable :class:`ConstrainedDiagnosticCode.COMPLEX_NON_HOLOMORPHIC`
    code so callers can match on it without parsing the message.
    """

    def __init__(self, op_name: str, python_attr: str,
                 fn_name: str = "") -> None:
        self.op_name = op_name
        self.python_attr = python_attr
        self.fn_name = fn_name
        msg = (
            f"@analytic_symbolic({fn_name}): IR contains "
            f"non-holomorphic op {op_name!r} "
            f"(complex.{python_attr}).  A composition of complex "
            "ops is holomorphic only when every op is."
        )
        super().__init__(msg)

    def to_diagnostic(self):
        """Lift this exception into a unified
        :class:`tessera.compiler.diagnostics.Diagnostic`."""

        from .diagnostics import (
            ConstrainedDiagnosticCode,
            Diagnostic,
        )

        return Diagnostic.from_constrained(
            code=ConstrainedDiagnosticCode.COMPLEX_NON_HOLOMORPHIC.value,
            message=str(self),
            lane="complex_jit",
            detail={
                "op_name": self.op_name,
                "python_attr": self.python_attr,
                "fn_name": self.fn_name,
            },
        )


@dataclass(frozen=True)
class HolomorphicViolation:
    op_name: str
    python_attr: str
    result_name: str

    def format(self) -> str:
        return (
            f"non-holomorphic op {self.op_name!r} (complex.{self.python_attr}) "
            f"binds to SSA ref {self.result_name}"
        )


@dataclass(frozen=True)
class HolomorphicReport:
    holomorphic: bool
    violations: tuple[HolomorphicViolation, ...]

    def __bool__(self) -> bool:
        return self.holomorphic

    def format(self) -> str:
        if self.holomorphic:
            return "holomorphic (every IR op is in HOLOMORPHIC_OPS)"
        lines = ["non-holomorphic — IR contains:"]
        for v in self.violations:
            lines.append(f"  - {v.format()}")
        return "\n".join(lines)


# Thin type alias mirroring the M6 pattern: a CompileReport-shaped
# IR specific to complex functions.
@dataclass(frozen=True)
class ComplexIRProgram:
    arg_names: tuple[str, ...]
    ops: tuple[_ast_ir.IROpCall, ...]
    return_ref: str

    def text(self) -> str:
        lines = [f"complex_ir({', '.join(self.arg_names)}):"]
        for c in self.ops:
            lines.append(
                f"  {c.result_name} = {c.op_name}({', '.join(c.operand_refs)})"
                + (f"  # complex.{c.python_attr}" if c.python_attr else "")
            )
        lines.append(f"  return {self.return_ref}")
        return "\n".join(lines)

    def to_graph_ir_view(
        self,
        *,
        function_name: str = "complex_fn",
    ) -> "Any":
        """Project this program into a Graph IR module for audit /
        explain / normalization consumption.

        Phase B (2026-05-20).  The op names in the constrained IR
        are already canonical (``mobius``, ``stereographic``,
        ``complex_mul``, ``complex_exp``) — the backend aliases
        (``complex_mobius``, ``complex_stereographic``) live only
        in :mod:`tessera.compiler.backend_manifest` and are applied
        downstream by the audit walker.

        Lane stamping: ``view.functions[0].lane = "complex_jit"``.

        Verification facts: ``{"holomorphic"}`` **only** when every
        op in this program is in :data:`HOLOMORPHIC_OPS`.  A
        program with even one non-holomorphic op (``complex_abs``,
        ``complex_conjugate``, ``stereographic``, etc.) projects
        without the holomorphic fact — passes that depend on it
        won't fire.

        Contract spec: ``docs/spec/COMPILER_REFERENCE.md``
        § "Constrained-lane Graph IR views".
        """

        from ._view_helpers import build_graph_ir_view

        facts: frozenset[str] = frozenset()
        if all(c.op_name in HOLOMORPHIC_OPS for c in self.ops):
            facts = frozenset({"holomorphic"})
        return build_graph_ir_view(
            function_name=function_name,
            arg_names=self.arg_names,
            ops=self.ops,
            return_ref=self.return_ref,
            lane="complex_jit",
            verification_facts=facts,
            value_kind="complex",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Lowering
# ─────────────────────────────────────────────────────────────────────────────

_COMPLEX_LOWERING_CONFIG = _ast_ir.LoweringConfig(
    namespace="complex",
    attr_to_op_name=_COMPLEX_ATTR_TO_OP_NAME,
    error_prefix="complex_jit",
    error_class=ComplexJitError,
)


def lower_complex_function(fn: Callable[..., Any]) -> ComplexIRProgram:
    """Lower a complex-only Python function to a
    :class:`ComplexIRProgram` via the shared ``ast_ir`` core.

    The function must read ``complex.<op>(...)`` (where
    ``<op>`` is in :data:`_COMPLEX_ATTR_TO_OP_NAME`); any other
    call shape raises :class:`ComplexJitError`.
    """
    generic = _ast_ir.lower_function(fn, _COMPLEX_LOWERING_CONFIG)
    return ComplexIRProgram(
        arg_names=generic.arg_names,
        ops=generic.ops,
        return_ref=generic.return_ref,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Symbolic Cauchy-Riemann verifier
# ─────────────────────────────────────────────────────────────────────────────

def is_holomorphic(ir_or_fn: Any) -> HolomorphicReport:
    """Symbolically classify whether a complex function is
    holomorphic by walking its IR.

    Returns a :class:`HolomorphicReport` whose ``holomorphic``
    field is ``True`` iff every op in the function's IR is in
    :data:`HOLOMORPHIC_OPS`.  Otherwise the report lists each
    non-holomorphic op + its IR position.
    """
    if isinstance(ir_or_fn, ComplexIRProgram):
        ir = ir_or_fn
    elif callable(ir_or_fn):
        ir = lower_complex_function(ir_or_fn)
    else:
        raise TypeError(
            f"is_holomorphic: expected ComplexIRProgram or callable, "
            f"got {type(ir_or_fn).__name__}"
        )
    violations: list[HolomorphicViolation] = []
    for op in ir.ops:
        if op.op_name not in HOLOMORPHIC_OPS:
            violations.append(HolomorphicViolation(
                op_name=op.op_name,
                python_attr=op.python_attr,
                result_name=op.result_name,
            ))
    return HolomorphicReport(
        holomorphic=not violations,
        violations=tuple(violations),
    )


def analytic_symbolic(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator: lower the complex function at decoration time
    and raise :class:`NotHolomorphicError` if its IR contains any
    non-holomorphic op.

    Unlike the numerical ``@tessera.complex.analytic`` decorator
    (which probes at runtime), this one is **compile-time** and
    **exact**: it surfaces non-analyticity from a single op
    inspection, not from finite-difference residuals.
    """
    fn_name = getattr(fn, "__qualname__", "<anon>")
    ir = lower_complex_function(fn)
    report = is_holomorphic(ir)
    if not report.holomorphic:
        first = report.violations[0]
        raise NotHolomorphicError(
            first.op_name, first.python_attr, fn_name=fn_name,
        )
    # Verified holomorphic — return the function unchanged.
    fn.__tessera_analytic_symbolic__ = True  # type: ignore[attr-defined]
    return fn


# Public alias.  The M7 milestone doc + audit rows reference
# ``@complex_jit`` as the canonical decorator name (parallel to
# ``@clifford_jit`` for the GA family).  The implementation lives
# under :func:`analytic_symbolic` for clarity in the decoration-time
# logic; the alias gives the public surface the name users actually
# type.  Both names point at the same function and are kept in
# ``__all__`` so static analyzers see both.
complex_jit = analytic_symbolic


__all__ = [
    "ComplexJitError",
    "ComplexIRProgram",
    "HolomorphicReport",
    "HolomorphicViolation",
    "NotHolomorphicError",
    "HOLOMORPHIC_OPS",
    "NON_HOLOMORPHIC_OPS",
    "lower_complex_function",
    "is_holomorphic",
    "analytic_symbolic",
    "complex_jit",
]
