"""G3 (2026-05-19) — propagate numeric_policy through Graph IR.

The :class:`tessera.compiler.primitive_coverage.NumericPolicy` record
lives on op specs in ``primitive_coverage._NUMERIC_POLICY_BY_NAME_FACTORIES``
— it says, e.g., that ``matmul`` runs with ``storage=bf16, accum=fp32``.
Before this pass, that information stayed in the registry; per-op IR
instances didn't carry it.  Fused chains (``softmax(matmul(a, b))``)
couldn't see that the matmul claimed fp32 accumulators, so they had
to make a conservative choice per-op.

This pass walks a :class:`GraphIRFunction` and stamps the
``numeric_policy`` attribute on every :class:`IROp` whose op_name
appears in the factory table.  It does not rewrite the op, doesn't
emit diagnostics, and is idempotent — repeated calls overwrite with
the same value.

A companion :func:`validate_numeric_policy_chain` flags mismatches
between an op's input storage and the upstream op's accumulator —
useful for catching "softmax expected fp32 but matmul output is
bf16" silently-wrong precision flows.

Design
------

* **Optional, not required.**  An ``IROp.numeric_policy = None``
  remains valid — the op falls back on its ``result_type`` storage.
  This means the pass is incremental: producers and consumers can
  start using ``numeric_policy`` without forcing every site to
  populate it.
* **Read-only against ``primitive_coverage``.**  No cycles introduced;
  the factory table is the source of truth.  Adding a new op's
  policy requires editing ``primitive_coverage.py``, not this module.
* **Lane-aware via ``GraphIRFunction.lane``.**  A future pass can
  read the function's lane and apply lane-specific overrides
  (e.g., ``@energy_jit`` forces fp32 for energy ops regardless of
  the catalog's default).  Today's pass is lane-agnostic — it
  applies the catalog policy unchanged.
"""

from __future__ import annotations

from .diagnostics import Diagnostic
from .graph_ir import GraphIRFunction, GraphIRModule, IROp


def _resolve_op_name(raw: str) -> str:
    """Strip the canonical ``tessera.`` prefix so registry lookups
    work uniformly on both ``"matmul"`` and ``"tessera.matmul"``."""

    if raw.startswith("tessera."):
        return raw[len("tessera."):]
    return raw


def _policy_for_op_name(op_name: str):
    """Return the :class:`NumericPolicy` registered for ``op_name``,
    or ``None`` when the catalog has no entry."""

    # Late import — keeps this module light at load time and avoids
    # cycles through primitive_coverage which imports many things.
    from .primitive_coverage import _NUMERIC_POLICY_BY_NAME_FACTORIES

    factory = _NUMERIC_POLICY_BY_NAME_FACTORIES.get(_resolve_op_name(op_name))
    if factory is None:
        return None
    return factory()


def propagate_numeric_policy(
    fn: GraphIRFunction,
    *,
    overwrite: bool = False,
) -> int:
    """Walk ``fn.body`` and stamp ``numeric_policy`` on every op that
    has one registered.

    Parameters
    ----------
    fn:
        The function to annotate.  Mutated in place.
    overwrite:
        When ``False`` (default), ops that already carry a non-None
        ``numeric_policy`` are skipped — useful when a producer
        already set a stronger policy (e.g., ``@energy_jit`` forced
        fp32 for an op the catalog would have left bf16).  When
        ``True``, all ops are re-stamped from the catalog.

    Returns
    -------
    int
        Number of ops that received a policy from this call.
    """

    stamped = 0
    for op in fn.body:
        if op.numeric_policy is not None and not overwrite:
            continue
        policy = _policy_for_op_name(op.op_name)
        if policy is None:
            continue
        op.numeric_policy = policy
        stamped += 1
    return stamped


def propagate_numeric_policy_module(
    module: GraphIRModule,
    *,
    overwrite: bool = False,
) -> int:
    """Stamp every op in every function of ``module``.  Returns the
    total count of ops that received a policy."""

    total = 0
    for fn in module.functions:
        total += propagate_numeric_policy(fn, overwrite=overwrite)
    return total


def validate_numeric_policy_chain(
    fn: GraphIRFunction,
) -> list[Diagnostic]:
    """Flag potential precision mismatches in a fused chain.

    For every op whose operand SSA refs match earlier ops' results,
    check that the producer's ``accum`` (or storage when accum is
    None) matches what the consumer's policy expects on its input.

    Mismatches today are warnings — the pass doesn't reject IR
    because mixed-precision chains are sometimes intentional
    (e.g., a bf16 → fp32 cast op is a legitimate bridge).  Returns
    the diagnostics list; ``.explain()`` can surface them.
    """

    # Build a producer map: result-ref → producing op.
    producer: dict[str, IROp] = {}
    for op in fn.body:
        if op.result is not None:
            producer[op.result] = op
            # Also accept the "%name" form some producers use.
            producer[f"%{op.result}"] = op

    diagnostics: list[Diagnostic] = []
    for op in fn.body:
        if op.numeric_policy is None:
            continue
        for operand in op.operands:
            upstream = producer.get(operand)
            if upstream is None or upstream.numeric_policy is None:
                continue
            # Compare upstream's accum (or storage when accum is None)
            # to this op's storage.
            upstream_out = (
                upstream.numeric_policy.accum
                or upstream.numeric_policy.storage
            )
            consumer_in = op.numeric_policy.storage
            if upstream_out != consumer_in:
                # Not always a bug — bf16-storage + fp32-accum is
                # canonical for matmul.  But flag for visibility.
                diagnostics.append(Diagnostic(
                    severity="info",
                    code="NUMERIC_POLICY_CHAIN_MISMATCH",
                    message=(
                        f"op {op.op_name!r} consumes "
                        f"{operand!r} produced as {upstream_out!r}; "
                        f"consumer policy expects "
                        f"storage={consumer_in!r}.  "
                        "Insert a cast if precision is intentional, "
                        "or align the policies if not."
                    ),
                    detail={
                        "consumer_op": op.op_name,
                        "consumer_storage": consumer_in,
                        "producer_op": upstream.op_name,
                        "producer_output_dtype": upstream_out,
                        "operand": operand,
                    },
                    lane=fn.lane,
                ))
    return diagnostics


__all__ = [
    "propagate_numeric_policy",
    "propagate_numeric_policy_module",
    "validate_numeric_policy_chain",
]
