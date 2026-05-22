"""``@energy_jit`` — restricted energy-function AST → EnergyIRProgram.

M6 Step 2 deliverable.  Defines the *shape* of the energy lowering
surface — the whitelist, the IR type, and the decoration-time
validator — without yet generating fused MSL kernels or closed-form
VJPs (those are M6 Steps 3 + 4).

Why this is small: M6 explicitly reuses the shared
:mod:`tessera.compiler.ast_ir` core that ``@clifford_jit`` is
already built on.  Adding an energy lane is a whitelist + a config
+ an :class:`EnergyIRProgram` alias, not a new compiler.

Surface (M6 Step 2 scope):

  Quadratic / bilinear:
    ``energy.quadratic(y, W)``         — ``y^T W y``
    ``energy.bilinear(y, x, W)``       — ``y^T W x``
    ``energy.inner(y, x)``             — ``y · x``

  Polynomial / norms:
    ``energy.polynomial(y, coefs)``    — Σ_k coefs[k] · y^k (low degree)
    ``energy.norm(y)``                 — ‖y‖₂
    ``energy.norm_sq(y)``              — ‖y‖₂²

  Activations:
    ``energy.relu(x)`` / ``tanh`` / ``sigmoid`` / ``gelu`` / ``softplus``

  Small MLP heads:
    ``energy.linear(y, W, b)``         — y @ W + b
    ``energy.mlp_head(y, W1, b1, W2, b2)``
                                        — linear → activation → linear

  Aggregation:
    ``energy.reduce_sum(y)``           — scalar reduction for the
                                          final energy value

Out of scope this step (deferred to M6 Steps 3/4):
  - Closed-form VJP generation w.r.t. ``y``.
  - MSL codegen.
  - On-device Philox RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from tessera.compiler import ast_ir as _ast_ir


class EnergyJitError(Exception):
    """Raised when an ``@energy_jit`` function violates the v1
    contract (op not in whitelist, unsupported AST shape, etc.).

    Carries an optional stable :class:`ConstrainedDiagnosticCode`
    so callers can match against the code rather than the message.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
    ) -> None:
        self.code = code
        super().__init__(message)

    def to_diagnostic(self):
        """Lift this exception into a unified
        :class:`tessera.compiler.diagnostics.Diagnostic`."""

        from .diagnostics import (
            ConstrainedDiagnosticCode,
            Diagnostic,
        )

        code = self.code or ConstrainedDiagnosticCode.ENERGY_FORBIDDEN_OP.value
        return Diagnostic.from_constrained(
            code=code,
            message=str(self),
            lane="energy_jit",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Energy whitelist — public Python attr → canonical IR op name.
# ─────────────────────────────────────────────────────────────────────────────

_ENERGY_ATTR_TO_OP_NAME: dict[str, str] = {
    # Bilinear / quadratic
    "quadratic":   "energy_quadratic",
    "bilinear":    "energy_bilinear",
    "inner":       "energy_inner",
    # Polynomial / norms
    "polynomial":  "energy_polynomial",
    "norm":        "energy_norm",
    "norm_sq":     "energy_norm_sq",
    # Activations
    "relu":        "energy_relu",
    "tanh":        "energy_tanh",
    "sigmoid":     "energy_sigmoid",
    "gelu":        "energy_gelu",
    "softplus":    "energy_softplus",
    # Small dense heads
    "linear":      "energy_linear",
    "mlp_head":    "energy_mlp_head",
    # Aggregation
    "reduce_sum":  "energy_reduce_sum",
}


# ─────────────────────────────────────────────────────────────────────────────
# IR types — thin renames of the shared ast_ir core for callers that
# want energy-flavored type names.
# ─────────────────────────────────────────────────────────────────────────────

EnergyIROpCall = _ast_ir.IROpCall


@dataclass(frozen=True)
class EnergyIRProgram:
    """A lowered energy function.

    Wraps :class:`tessera.compiler.ast_ir.IRProgram` with an
    energy-flavored name, ``text()`` prefix (``energy_ir(...)``), and
    field accessors that match the Clifford program's shape so
    downstream consumers can treat both uniformly.
    """
    arg_names: tuple[str, ...]
    ops: tuple[EnergyIROpCall, ...]
    return_ref: str

    def as_metadata(self) -> dict[str, Any]:
        return {
            "namespace": "energy",
            "arg_names": list(self.arg_names),
            "ops": [
                {"op": c.op_name, "operands": list(c.operand_refs),
                 "result": c.result_name, "python_attr": c.python_attr}
                for c in self.ops
            ],
            "return_ref": self.return_ref,
        }

    def text(self) -> str:
        lines = [f"energy_ir({', '.join(self.arg_names)}):"]
        for c in self.ops:
            lines.append(
                f"  {c.result_name} = {c.op_name}({', '.join(c.operand_refs)})"
                + (f"  # energy.{c.python_attr}" if c.python_attr else "")
            )
        lines.append(f"  return {self.return_ref}")
        return "\n".join(lines)

    def to_graph_ir_view(
        self,
        *,
        function_name: str = "energy_fn",
    ) -> "Any":
        """Project this program into a Graph IR module for audit /
        explain / normalization consumption.

        Phase B (2026-05-20).  Op names are already canonical
        (``energy_quadratic``, ``energy_softmax``, etc.).

        Lane stamping: ``view.functions[0].lane = "energy_jit"``.
        Verification facts: ``{"energy_whitelisted"}``.

        Contract spec: ``docs/spec/COMPILER_REFERENCE.md``
        § "Constrained-lane Graph IR views".
        """

        from ._view_helpers import build_graph_ir_view

        return build_graph_ir_view(
            function_name=function_name,
            arg_names=self.arg_names,
            ops=self.ops,
            return_ref=self.return_ref,
            lane="energy_jit",
            verification_facts=frozenset({"energy_whitelisted"}),
            value_kind="energy",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Lowering
# ─────────────────────────────────────────────────────────────────────────────

_ENERGY_LOWERING_CONFIG = _ast_ir.LoweringConfig(
    namespace="energy",
    attr_to_op_name=_ENERGY_ATTR_TO_OP_NAME,
    error_prefix="energy_jit",
    error_class=EnergyJitError,
)


def lower_energy_function(fn: Callable[..., Any]) -> EnergyIRProgram:
    """Lower an energy function to :class:`EnergyIRProgram`.

    The function must have the shape::

        def E(y, *params):
            \"\"\"optional docstring\"\"\"
            x = energy.<op>(...)
            ...
            return energy.<op>(...)

    Anything else raises :class:`EnergyJitError`.
    """
    generic = _ast_ir.lower_function(fn, _ENERGY_LOWERING_CONFIG)
    return EnergyIRProgram(
        arg_names=generic.arg_names,
        ops=generic.ops,
        return_ref=generic.return_ref,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Decorator surface — v1 returns a callable carrying the lowered IR.
# Execution (M6 Step 3) and MSL codegen (M6 Step 4) are deferred.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EnergyCompiledArtifact:
    """The lowered, validated energy function — what
    ``@energy_jit`` carries on the returned callable."""
    source_name: str
    target: str
    dtype: str
    ir: EnergyIRProgram


class EnergyCompiledCallable:
    """Decorated energy-function wrapper.

    M6 Step 2 keeps the callable simple: it carries the IR + the
    original Python function, and delegates execution to the Python
    function (NOT a generated kernel) until M6 Steps 3 + 4 land.
    Calling the instance therefore matches the user's source code
    semantics; the IR is purely for compile-time inspection and
    future codegen.
    """

    def __init__(
        self, fn: Callable[..., Any], artifact: EnergyCompiledArtifact,
    ) -> None:
        self._fn = fn
        self.artifact = artifact
        # Skip functools.update_wrapper — under Python 3.14 setting
        # __module__ on a custom class instance is blocked, and the
        # important wrapper attributes (__name__, __qualname__) are
        # set explicitly below.  This mirrors what callers of
        # functools.wraps need at the test boundary.
        self.__name__ = getattr(fn, "__name__", "<energy_compiled>")
        self.__qualname__ = getattr(fn, "__qualname__", self.__name__)
        self.__doc__ = getattr(fn, "__doc__", None)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)

    # ─────────────────────────────────────────────────────────────────
    # M6 Slice 1 (2026-05-22) — gradient + refine integration.
    #
    # Until now, users decorating an energy function with @energy_jit
    # got back a callable carrying only the forward IR.  To compute
    # ∂E/∂y they had to manually call ``make_gradient_program(fn)`` —
    # a friction point that meant the @energy_jit lane could not be
    # the everyday truth source for energy + gradient + refinement.
    #
    # These four accessors close the gap:
    #
    #   .gradient_program  : lazily built EnergyGradientProgram,
    #                         cached on the instance.  Build cost paid
    #                         once; per-call work is forward + reverse
    #                         + cotangent accumulation only.
    #   .grad_y(env)       : ∂E/∂y for an env binding y + params.
    #   .refine(y0, T, η)  : T-step gradient descent (loop alias for
    #                         the energy_grad.refine() function with
    #                         the cached program).
    #   .fused_report()    : inspectable dict describing the energy
    #                         IR + paired VJP chain — what an MSL
    #                         fused energy+grad kernel would compile
    #                         from when M6 Step 4 lands.
    # ─────────────────────────────────────────────────────────────────

    @property
    def gradient_program(self) -> "Any":  # EnergyGradientProgram forward-ref
        """Lazily built + cached :class:`EnergyGradientProgram` for
        this artifact's IR.  Build happens once per instance; reuse
        across many ``grad_y`` / ``refine`` calls.

        Lazy + thread-unsafe by design: the build is fast and idempotent;
        the first thread to access wins, subsequent threads see the
        cached instance.  If thread-safety becomes important, wrap in
        an :class:`RLock` here (no other caller of this property needs
        to change).
        """
        cached = getattr(self, "_gradient_program", None)
        if cached is None:
            from .energy_grad import make_gradient_program
            cached = make_gradient_program(self.artifact.ir)
            self._gradient_program = cached
        return cached

    def grad_y(self, env: "Mapping[str, Any]") -> Any:
        """Reverse-mode autodiff: ∂E/∂y where ``y`` is the first
        positional argument in the energy function's signature.

        ``env`` binds every named arg by its source-code name (so for
        ``def E(y, W): ...`` you pass ``{"y": ..., "W": ...}``).
        Returns a numpy array shaped like ``env[y_name]``.
        """
        return self.gradient_program.grad_y(env)

    def refine(
        self,
        y0: Any,
        *,
        T: int,
        eta: float,
        params: "Mapping[str, Any] | None" = None,
    ) -> Any:
        """``T`` steps of gradient descent against the gradient program:

            y ← y − η · grad_y(y, *params)

        The gradient program is built once (lazy cache) and reused
        across every step — mirroring the invariant the future fused
        MSL kernel will satisfy: bind once, loop T times.

        Returns the refined ``y`` as a numpy array.
        """
        from .energy_grad import refine as _refine
        return _refine(y0, self.gradient_program, T=T, eta=eta, params=params)

    def fused_report(self) -> dict[str, Any]:
        """Inspectable description of the energy IR + paired VJP
        chain.  This is what an MSL fused energy+gradient kernel
        would compile from when M6 Step 4 lands — exposing it now
        lets external tooling (status dashboards, autotuners, design
        reviews) reason about the fused shape without waiting for
        codegen.

        Returns a dict matching the CompileReport-style envelope so
        consumers can serialise + diff uniformly.  Fields:

          forward_ops     : list of {op_name, operand_refs, result_name}
                            from the EnergyIRProgram.
          gradient_chain  : list of {op_name, has_vjp, arity} — the
                            backward pass the gradient program will
                            run; empty if any op lacks a VJP (in
                            which case the build raises and this
                            method surfaces the failure mode in the
                            ``errors`` field).
          fusion_class    : 'forward_only' | 'forward_and_grad'.  M6
                            Step 4 will lower 'forward_and_grad' to a
                            single MSL dispatch; Step 3 lowers
                            forward_only.
          arg_names       : tuple, mirrors ir.arg_names.
          return_ref      : str, mirrors ir.return_ref.
          errors          : tuple of strings — empty on success.
        """
        from .energy_vjp import has_vjp
        # Inline import to avoid a hard dep on the lookup table here.
        from .energy_grad import _OP_ARITY

        forward_ops = [
            {
                "op_name": op.op_name,
                "operand_refs": list(op.operand_refs),
                "result_name": op.result_name,
            }
            for op in self.artifact.ir.ops
        ]

        gradient_chain: list[dict[str, Any]] = []
        errors: list[str] = []
        all_have_vjp = True
        for op in self.artifact.ir.ops:
            has = has_vjp(op.op_name)
            arity = _OP_ARITY.get(op.op_name, 0)
            gradient_chain.append({
                "op_name": op.op_name,
                "has_vjp": bool(has),
                "arity": arity,
            })
            if not has:
                all_have_vjp = False
                errors.append(
                    f"no VJP registered for {op.op_name!r}; "
                    "fused energy+gradient kernel cannot lower this op"
                )

        fusion_class = "forward_and_grad" if all_have_vjp else "forward_only"

        return {
            "program_id": self.artifact.source_name,
            "source": f"@energy_jit({self.artifact.source_name})",
            "target": self.artifact.target,
            "dtype": self.artifact.dtype,
            "arg_names": list(self.artifact.ir.arg_names),
            "return_ref": self.artifact.ir.return_ref,
            "forward_ops": forward_ops,
            "gradient_chain": gradient_chain,
            "fusion_class": fusion_class,
            "errors": errors,
        }

    # ─────────────────────────────────────────────────────────────────
    # Slice 4 (2026-05-22) — CompileReport canonical schema.
    #
    # Every JIT frontend (@tessera.jit, textual, @clifford_jit,
    # @energy_jit) must expose a uniform CompileReport accessor so the
    # M5 no-silent-native rule + the M1 stability gate apply to every
    # shipped lane.  energy_jit funnels through the same AST-constrained
    # template as clifford_jit, so it reports as FRONTEND_CLIFFORD_JIT
    # with the source disambiguating "energy_jit" vs other constrained
    # lanes.  Value kind is `tensor` (energy outputs are scalar/tensor,
    # never multivector).
    # ─────────────────────────────────────────────────────────────────

    def compile_report(self) -> "Any":  # CompileReport forward-ref
        """Synthesize a :class:`CompileReport` from this callable's
        artifact.  Same shape as ``CliffordCompiledCallable.compile_report``
        so cross-lane consumers (CI, audit, status dashboards) can
        treat both uniformly."""
        from . import compile_report as _cr  # local to avoid cycle
        ir = self.artifact.ir
        ir_text = ir.text() if ir is not None else ""
        ir_hashes = (
            {"graph_ir": _cr.hash_ir_text(ir_text)} if ir_text else {}
        )
        return _cr.CompileReport(
            program_id=self.artifact.source_name,
            source=f"@energy_jit({self.artifact.source_name})",
            frontend=_cr.FRONTEND_CLIFFORD_JIT,
            value_kind=_cr.VALUE_KIND_TENSOR,
            target=self.artifact.target,
            ir_hashes=ir_hashes,
            target_decision={
                self.artifact.target: f"energy_jit({self.artifact.dtype})"
            },
        )


def energy_jit(
    *, target: str = "apple_gpu", dtype: str = "f32",
) -> Callable[[Callable[..., Any]], EnergyCompiledCallable]:
    """Decorator: lower a restricted energy function to
    :class:`EnergyIRProgram` at decoration time.

    M6 Step 2 (2026-05-18): IR + validation only.  Native MSL
    codegen + closed-form VJP generation land in Steps 3 + 4.

    .. code-block:: python

        from tessera import energy

        @energy_jit(target="apple_gpu")
        def E(y):
            q   = energy.norm_sq(y)
            r   = energy.relu(y)
            out = energy.inner(r, y)
            return energy.reduce_sum(out)

        E.artifact.ir.text()   # human-readable IR
        E.artifact.ir.as_metadata()  # JSON-serializable form
    """
    if dtype != "f32":
        from .diagnostics import ConstrainedDiagnosticCode as _Code
        raise EnergyJitError(
            f"energy_jit v1 only supports dtype='f32', got {dtype!r}",
            code=_Code.ENERGY_UNSUPPORTED_DTYPE.value,
        )

    def decorator(fn: Callable[..., Any]) -> EnergyCompiledCallable:
        ir = lower_energy_function(fn)
        artifact = EnergyCompiledArtifact(
            source_name=fn.__qualname__,
            target=target,
            dtype=dtype,
            ir=ir,
        )
        return EnergyCompiledCallable(fn, artifact)

    return decorator


__all__ = [
    "EnergyJitError",
    "EnergyIRProgram",
    "EnergyIROpCall",
    "EnergyCompiledArtifact",
    "EnergyCompiledCallable",
    "energy_jit",
    "lower_energy_function",
]
