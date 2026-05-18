"""M6 Step 3 — gradient-program builder + T-step refinement.

The :mod:`energy_vjp` module ships closed-form VJPs for every
whitelisted energy op.  M6 Step 3's remaining work is to compose
those rules into a per-program gradient pass and a refinement
loop that **builds the gradient program once** and reuses it
across T steps.

Why that property matters:

  * On Apple GPU, the future fused MSL kernel for T-step
    refinement will allocate buffers and bind symbols once at
    the start of the dispatch, then loop T iterations inside
    the same kernel — no per-step rebind / re-upload.
  * The Python reference here preserves the same invariant:
    :class:`EnergyGradientProgram` is built once from an
    :class:`EnergyIRProgram`, and :func:`refine` reuses it across
    every step.  Tests verify the build call count.

The acceptance criterion M6 Step 3 ships against:

  * Each whitelisted op gets a working ``grad_y`` via reverse-
    mode AD over the closed-form VJP table.
  * The composed gradient matches finite differences.
  * The T-step refinement converges on a known minimum.
  * The build-once / reuse-per-step contract is auditable
    (a test inspects the gradient program's call count).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import numpy as np

from .. import energy as _energy
from .energy_jit import EnergyIRProgram, lower_energy_function
from .energy_vjp import has_vjp, vjp_for


# ─────────────────────────────────────────────────────────────────────────────
# Op → forward callable.  Mirrors :data:`energy_jit._ENERGY_ATTR_TO_OP_NAME`
# but keyed on the **canonical IR op name** so the executor doesn't have
# to round-trip through python attrs.
# ─────────────────────────────────────────────────────────────────────────────

_FORWARD_BY_OP_NAME: Mapping[str, Callable[..., np.ndarray]] = {
    "energy_quadratic":   _energy.quadratic,
    "energy_bilinear":    _energy.bilinear,
    "energy_inner":       _energy.inner,
    "energy_polynomial":  _energy.polynomial,
    "energy_norm":        _energy.norm,
    "energy_norm_sq":     _energy.norm_sq,
    "energy_relu":        _energy.relu,
    "energy_tanh":        _energy.tanh,
    "energy_sigmoid":     _energy.sigmoid,
    "energy_gelu":        _energy.gelu,
    "energy_softplus":    _energy.softplus,
    "energy_linear":      _energy.linear,
    "energy_mlp_head":    _energy.mlp_head,
    "energy_reduce_sum":  _energy.reduce_sum,
}


# ─────────────────────────────────────────────────────────────────────────────
# Op-arity table — how many positional ``np.ndarray``-shape inputs each
# op consumes BEFORE the ``out_grad`` cotangent in its VJP signature.
# The builder needs this to slice the operand list correctly.
#
# Note: ``polynomial`` takes (y, coefs) where ``coefs`` is a Python
# sequence, not an ndarray.  We treat it specially.
# ─────────────────────────────────────────────────────────────────────────────

_OP_ARITY: Mapping[str, int] = {
    "energy_quadratic":   2,
    "energy_bilinear":    3,
    "energy_inner":       2,
    "energy_polynomial":  2,
    "energy_norm":        1,
    "energy_norm_sq":     1,
    "energy_relu":        1,
    "energy_tanh":        1,
    "energy_sigmoid":     1,
    "energy_gelu":        1,
    "energy_softplus":    1,
    "energy_linear":      3,
    "energy_mlp_head":    5,
    "energy_reduce_sum":  1,
}


class EnergyGradientError(Exception):
    """Raised by :class:`EnergyGradientProgram` when the IR or env
    fails a pre-condition (unknown op, missing operand, etc.)."""


# ─────────────────────────────────────────────────────────────────────────────
# Built program
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnergyGradientProgram:
    """A forward-only :class:`EnergyIRProgram` paired with the
    machinery to compute the gradient w.r.t. its first argument.

    The program is **built once** via :func:`make_gradient_program`
    and reused across many evaluations.  Per-call work is the
    forward pass + reverse pass + minor cotangent accumulation —
    no IR re-analysis, no allocation table re-build.

    Public API:

      * :meth:`evaluate(env) -> energy`  — forward only.
      * :meth:`grad_y(env) -> ndarray`   — ∂E/∂y as a numpy array,
        same shape as ``env[arg_names[0]]``.
      * :attr:`build_call_count` — increments on creation; tests
        verify ``refine`` doesn't rebuild per step.
    """
    ir: EnergyIRProgram
    # Resolved per-op metadata cached at build time.
    _op_arity_cache: tuple[int, ...] = ()
    build_call_count: int = field(default=0)

    def __post_init__(self) -> None:
        # Build-time validation: every IR op must have a VJP AND a
        # known forward callable AND a known arity.  Anything else
        # is a programmer error.
        arities: list[int] = []
        for op in self.ir.ops:
            if op.op_name not in _FORWARD_BY_OP_NAME:
                raise EnergyGradientError(
                    f"unknown forward op {op.op_name!r} in energy IR; "
                    f"valid ops: {sorted(_FORWARD_BY_OP_NAME)}"
                )
            if not has_vjp(op.op_name):
                raise EnergyGradientError(
                    f"no closed-form VJP for {op.op_name!r}; cannot "
                    "build a gradient program over it"
                )
            arities.append(_OP_ARITY[op.op_name])
        object.__setattr__(self, "_op_arity_cache", tuple(arities))
        # Track that build was called.
        object.__setattr__(self, "build_call_count", self.build_call_count + 1)

    # ── forward ────────────────────────────────────────────────

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        """Forward pass.  Returns the energy value at the IR's
        ``return_ref``.  Does not compute gradients."""
        values = self._forward(env)
        return values[self.ir.return_ref]

    # ── gradient w.r.t. first argument ──────────────────────────

    def grad_y(self, env: Mapping[str, Any]) -> np.ndarray:
        """Reverse-mode autodiff to produce ``∂E/∂y``.

        The "y" here is the first argument in
        ``self.ir.arg_names`` — by convention every
        ``@energy_jit``-style function takes ``y`` first and
        params after.

        Implementation:

          * Forward pass populates a name → value map.
          * Backward pass walks ops in reverse, accumulating
            cotangents at each operand ref via the VJP table.
          * Return the cotangent at the first arg.
        """
        if not self.ir.arg_names:
            raise EnergyGradientError(
                "energy IR has no arguments; cannot compute grad_y"
            )
        y_name = self.ir.arg_names[0]
        values = self._forward(env)
        # Cotangent map keyed on operand ref.  ``+`` accumulates
        # contributions from multiple ops sharing the same operand.
        cotangents: dict[str, Any] = {}
        # Seed: ``∂E/∂return = 1`` (scalar energy).
        cotangents[self.ir.return_ref] = self._unit_cotangent(
            values[self.ir.return_ref],
        )
        # Backward pass.
        for op, arity in zip(reversed(self.ir.ops), reversed(self._op_arity_cache)):
            grad_out = cotangents.get(op.result_name)
            if grad_out is None:
                # Op result not consumed by anything that contributes
                # to the energy; safe to skip.
                continue
            # Resolve the op's positional inputs.
            inputs = self._resolve_inputs(op, values, arity)
            vjp = vjp_for(op.op_name)
            # The VJP returns one cotangent per input, positional order.
            input_grads = vjp(*inputs, grad_out)
            # Accumulate into the cotangent map.  Only the FIRST
            # ``arity`` operand refs are ndarray-shaped; trailing
            # operand refs (e.g., the ``coefs`` sequence for
            # ``polynomial``) get a cotangent we ignore.
            for ref, igrad in zip(op.operand_refs[:arity], input_grads[:arity]):
                if ref.startswith("#"):
                    continue  # inline literal — no gradient to record
                if ref in cotangents:
                    cotangents[ref] = cotangents[ref] + np.asarray(igrad)
                else:
                    cotangents[ref] = np.asarray(igrad)
        if y_name not in cotangents:
            # Function didn't depend on y → gradient is zero of y's shape.
            return np.zeros_like(np.asarray(env[y_name]))
        return cotangents[y_name].astype(np.asarray(env[y_name]).dtype, copy=False)

    # ── helpers ─────────────────────────────────────────────────

    def _forward(self, env: Mapping[str, Any]) -> dict[str, Any]:
        """Forward pass; returns name → value map.  Inline literal
        operand refs decode through :func:`ast_ir.resolve_operand`."""
        from .ast_ir import resolve_operand
        values: dict[str, Any] = dict(env)
        for op, arity in zip(self.ir.ops, self._op_arity_cache):
            forward = _FORWARD_BY_OP_NAME[op.op_name]
            # The forward op consumes positional args in IR order.
            # Most ops take 1-5 ndarrays; ``polynomial`` takes
            # (y, coefs) where ``coefs`` is a python sequence and
            # therefore comes from an env binding, not a literal.
            args = [resolve_operand(ref, values) for ref in op.operand_refs]
            values[op.result_name] = forward(*args)
        return values

    def _resolve_inputs(
        self, op, values: Mapping[str, Any], arity: int,
    ) -> list[Any]:
        """Resolve the positional inputs to a single op call,
        decoding inline literals through ast_ir.resolve_operand."""
        from .ast_ir import resolve_operand
        return [resolve_operand(ref, values) for ref in op.operand_refs]

    @staticmethod
    def _unit_cotangent(value: Any) -> Any:
        """Cotangent seed: 1 for scalar outputs, ones_like for
        array outputs (so reduce_sum-shaped energies and direct
        elementwise outputs both seed correctly)."""
        arr = np.asarray(value)
        if arr.shape == ():
            return 1.0
        return np.ones_like(arr)


def make_gradient_program(
    ir_or_fn: Any,
) -> EnergyGradientProgram:
    """Build a gradient program from either an :class:`EnergyIRProgram`
    or a Python callable that ``lower_energy_function`` accepts.

    Builds once.  Subsequent forward/grad_y calls reuse the same
    program — see the build-call-count tests in
    ``test_energy_grad.py``.
    """
    if isinstance(ir_or_fn, EnergyIRProgram):
        ir = ir_or_fn
    elif callable(ir_or_fn):
        ir = lower_energy_function(ir_or_fn)
    else:
        raise TypeError(
            f"make_gradient_program: expected EnergyIRProgram or "
            f"callable, got {type(ir_or_fn).__name__}"
        )
    return EnergyGradientProgram(ir=ir)


# ─────────────────────────────────────────────────────────────────────────────
# T-step refinement loop
# ─────────────────────────────────────────────────────────────────────────────

def refine(
    y0: Any,
    program: EnergyGradientProgram,
    *,
    T: int,
    eta: float,
    params: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """T steps of gradient descent against the gradient program.

    ``y ← y − η · grad_y(y, ...params)``

    The program is reused across every step — no rebuild between
    iterations.  This mirrors the invariant the future MSL fused
    kernel will satisfy: bind once, loop T times.
    """
    if T < 0:
        raise ValueError(f"refine: T must be non-negative, got {T}")
    y_name = program.ir.arg_names[0]
    y = np.asarray(y0).copy()
    base_env: dict[str, Any] = dict(params or {})
    # Pin every param to the env once; only `y` mutates each step.
    for _ in range(T):
        base_env[y_name] = y
        g = program.grad_y(base_env)
        y = y - eta * g
    return y


__all__ = [
    "EnergyGradientError",
    "EnergyGradientProgram",
    "make_gradient_program",
    "refine",
]
