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
from typing import Any, Callable

from tessera.compiler import ast_ir as _ast_ir


class EnergyJitError(Exception):
    """Raised when an ``@energy_jit`` function violates the v1
    contract (op not in whitelist, unsupported AST shape, etc.)."""


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
        raise EnergyJitError(
            f"energy_jit v1 only supports dtype='f32', got {dtype!r}")

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
