"""Shared adapter helpers for constrained-lane `to_graph_ir_view()`.

Phase B substrate.  All three constrained IR programs
(:class:`CliffordIRProgram`, :class:`ComplexIRProgram`,
:class:`EnergyIRProgram`) share the same internal op-call shape
(``op_name`` / ``operand_refs`` / ``result_name`` / ``python_attr``),
so they can share the projection logic.

The contract enforced here is documented in
``docs/spec/COMPILER_REFERENCE.md`` § "Constrained-lane Graph IR views":

  * 1:1 op projection — no reordering / merging / splitting.
  * Canonical op names — straight pass-through of the constrained
    IR's stored ``op_name`` field.  Adapters that need to invert
    backend aliases (e.g., turning ``complex_mobius`` back into
    ``mobius``) do so by storing the canonical name in the
    constrained IR itself — this helper does not perform alias
    inversion.
  * Fresh deep copy — the returned module is mutable and
    independent across calls.
  * Lane stamping + verification facts — supplied by the caller.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Sequence

from .graph_ir import (
    GraphIRFunction,
    GraphIRModule,
    IRArg,
    IROp,
    TENSOR_OPAQUE,
)


def build_graph_ir_view(
    *,
    function_name: str,
    arg_names: Sequence[str],
    ops: Iterable[Any],
    return_ref: str,
    lane: str,
    verification_facts: frozenset[str],
    value_kind: str | None = None,
) -> GraphIRModule:
    """Project a constrained-lane IR into a Graph IR module.

    Parameters
    ----------
    function_name:
        Name of the projected function — typically the source
        function's ``__qualname__``.
    arg_names:
        Argument-name tuple from the constrained IR (each adapter
        passes its own ``program.arg_names``).
    ops:
        Iterable of constrained ``IROpCall`` records.  Each must
        expose ``op_name`` / ``operand_refs`` / ``result_name``
        attributes — works uniformly across
        :class:`CliffordIROpCall`, :class:`ComplexIROpCall` (=
        :class:`tessera.compiler.ast_ir.IROpCall`), and
        :class:`EnergyIROpCall`.
    return_ref:
        The SSA ref the constrained IR returns.  Plumbed into
        ``GraphIRFunction.return_values`` so audit/explain can read
        the function's tail.
    lane:
        One of ``"clifford_jit"`` / ``"complex_jit"`` /
        ``"energy_jit"``.  Stamped on ``view.functions[0].lane``.
    verification_facts:
        The lane's invariant set.  Stamped on both
        ``view.functions[0].verification_facts`` and every IROp's
        ``verification_facts``.
    value_kind:
        Optional value-kind tag for each IROp.  When ``None``, ops
        leave the field as ``None`` (the documented "producer
        didn't claim" semantics).

    Returns
    -------
    GraphIRModule
        A fresh deep copy.  Mutations to the returned object never
        propagate back to a future call's view.
    """

    body: list[IROp] = []
    for call in ops:
        operands = list(call.operand_refs)
        body.append(IROp(
            result=call.result_name,
            op_name=call.op_name,
            operands=operands,
            operand_types=[str(TENSOR_OPAQUE)] * len(operands),
            result_type=str(TENSOR_OPAQUE),
            value_kind=value_kind,
            verification_facts=verification_facts,
        ))

    fn = GraphIRFunction(
        name=function_name,
        args=[IRArg(name=n, ir_type=TENSOR_OPAQUE) for n in arg_names],
        body=body,
        lane=lane,
        verification_facts=verification_facts,
        return_values=[return_ref],
    )
    module = GraphIRModule(functions=[fn])
    return copy.deepcopy(module)


__all__ = ["build_graph_ir_view"]
