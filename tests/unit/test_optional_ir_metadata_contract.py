"""Drift gate for the optional-IR-metadata contract (Phase A).

Per ``docs/architecture/frontend_substrate_plan.md`` § 1 (Optional
Metadata Contract + Drift Gate):

  Every new field on ``IROp`` / ``GraphIRFunction`` / ``GraphIRModule``
  must have a default value.  Adding a required field forces every
  producer (Python JIT, textual DSL, constrained-lane views) to
  compute it immediately — exactly the migration burden we're
  avoiding.

This file is that gate.  Failure means a field landed without a
default and without explicit grandfathering.  Add the field to
:data:`_GRANDFATHERED_REQUIRED_FIELDS` with a rationale comment if
the field is genuinely required (e.g., ``IROp.op_name``).
"""

from __future__ import annotations

import dataclasses

import pytest

from tessera.compiler.graph_ir import (
    GraphIRFunction,
    GraphIRModule,
    IROp,
)
from tessera.compiler.schedule_ir import (
    ScheduleFunction,
    ScheduleIRModule,
    ScheduleOp,
)
from tessera.compiler.tile_ir import (
    TileFunction,
    TileIRModule,
    TileOp,
)
from tessera.compiler.target_ir import (
    TargetFunction,
    TargetIRModule,
    TargetOp,
)


# Fields that are genuinely required at construction time.  Each entry
# has a rationale comment.  Adding to this set is a deliberate
# architectural decision — bumping the architecture-plan revision is
# expected.
#
# Issue 3 (2026-05-20) extended this gate to cover Schedule IR /
# Tile IR / Target IR — the same optional-metadata contract applies
# to every IR layer, not just Graph IR.  Adding a required field at
# any layer is a breaking change.
_GRANDFATHERED_REQUIRED_FIELDS: dict[type, frozenset[str]] = {
    IROp: frozenset({
        # ``result`` is positional but is allowed to be ``None`` —
        # required because the position matters for op identity.
        "result",
        # ``op_name`` is the canonical identifier (matmul, relu, ...).
        # A nameless op is meaningless.
        "op_name",
        # Operands + their types are paired and form the call shape;
        # absence isn't a valid state.
        "operands",
        "operand_types",
    }),
    GraphIRFunction: frozenset({
        # A function without a name can't be addressed in the module.
        "name",
    }),
    GraphIRModule: frozenset(),
    # ─── Schedule IR ─────────────────────────────────────────────
    ScheduleFunction: frozenset({
        "name",  # same rationale as GraphIRFunction.name
    }),
    ScheduleOp: frozenset({
        "op_name",  # same rationale as IROp.op_name
    }),
    ScheduleIRModule: frozenset(),
    # ─── Tile IR ─────────────────────────────────────────────────
    TileFunction: frozenset({
        "name",
    }),
    TileOp: frozenset({
        "op_name",
    }),
    TileIRModule: frozenset(),
    # ─── Target IR ───────────────────────────────────────────────
    TargetFunction: frozenset({
        "name",
    }),
    TargetOp: frozenset({
        "op_name",
    }),
    TargetIRModule: frozenset(),
}


def _required_fields(cls: type) -> set[str]:
    """Return field names whose default is :data:`dataclasses.MISSING`
    (i.e., must be supplied at construction time)."""

    out: set[str] = set()
    for f in dataclasses.fields(cls):
        has_default = (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING
        )
        if not has_default:
            out.add(f.name)
    return out


class TestOptionalMetadataContract:
    """Asserts the optional-metadata invariant for every IR class.

    Per the contract:
      * Producers fill what they know.
      * Consumers tolerate missing metadata.
      * Metadata semantics are stable when present.
    """

    @pytest.mark.parametrize(
        "cls",
        [
            IROp, GraphIRFunction, GraphIRModule,
            ScheduleOp, ScheduleFunction, ScheduleIRModule,
            TileOp, TileFunction, TileIRModule,
            TargetOp, TargetFunction, TargetIRModule,
        ],
        ids=lambda cls: cls.__name__,
    )
    def test_no_required_field_outside_allowlist(self, cls: type) -> None:
        actual_required = _required_fields(cls)
        allowed_required = _GRANDFATHERED_REQUIRED_FIELDS[cls]
        offending = actual_required - allowed_required
        assert offending == set(), (
            f"{cls.__name__} added required fields outside the "
            f"grandfathered allowlist: {sorted(offending)}.\n"
            f"\n"
            f"Per docs/architecture/frontend_substrate_plan.md § 1, "
            f"new IR fields must ship with a default value so "
            f"existing producers don't have to compute them.\n"
            f"\n"
            f"If this field is genuinely required (e.g., it forms "
            f"the op's identity), add it to "
            f"tests/unit/test_optional_ir_metadata_contract.py "
            f"_GRANDFATHERED_REQUIRED_FIELDS with a rationale "
            f"comment.\n"
            f"\n"
            f"Otherwise, change the field declaration to provide "
            f"a default value:\n"
            f"  {next(iter(offending))}: <Type> = <sensible_default>\n"
        )

    @pytest.mark.parametrize(
        "cls",
        [
            IROp, GraphIRFunction, GraphIRModule,
            ScheduleOp, ScheduleFunction, ScheduleIRModule,
            TileOp, TileFunction, TileIRModule,
            TargetOp, TargetFunction, TargetIRModule,
        ],
        ids=lambda cls: cls.__name__,
    )
    def test_allowlisted_fields_still_required(self, cls: type) -> None:
        """If a field is in the allowlist, it had better actually be
        required.  Otherwise the allowlist is stale documentation."""

        actual_required = _required_fields(cls)
        allowed_required = _GRANDFATHERED_REQUIRED_FIELDS[cls]
        stale = allowed_required - actual_required
        assert stale == set(), (
            f"{cls.__name__}: fields in _GRANDFATHERED_REQUIRED_FIELDS "
            f"are no longer required at construction: {sorted(stale)}.\n"
            f"Remove from the allowlist."
        )


class TestPhaseAFieldsLanded:
    """Confirm the Phase A optional fields actually exist + have the
    documented defaults.  Locks the wire shape so subsequent passes
    can consume them."""

    def test_irop_value_kind_defaults_to_none(self) -> None:
        op = IROp(
            result="r",
            op_name="matmul",
            operands=["%a", "%b"],
            operand_types=["t", "t"],
        )
        assert op.value_kind is None, (
            "IROp.value_kind must default to None so producers can "
            "distinguish 'didn't know' from 'definitely tensor' — "
            "per the architecture plan refinement on 2026-05-20."
        )

    def test_irop_verification_facts_defaults_to_empty_frozenset(self) -> None:
        op = IROp(
            result="r",
            op_name="matmul",
            operands=["%a", "%b"],
            operand_types=["t", "t"],
        )
        assert op.verification_facts == frozenset()

    def test_graph_ir_function_verification_facts_default(self) -> None:
        fn = GraphIRFunction(name="f")
        assert fn.verification_facts == frozenset()

    def test_graph_ir_function_source_hash_defaults_to_none(self) -> None:
        fn = GraphIRFunction(name="f")
        assert fn.source_hash is None


class TestProducersCanFillWhatTheyKnow:
    """Smoke test the three-line contract: producers fill, consumers
    read.  No required-field migration burden."""

    def test_producer_can_set_value_kind(self) -> None:
        op = IROp(
            result="r",
            op_name="rotor_sandwich",
            operands=["%a", "%b"],
            operand_types=["multivector", "multivector"],
            value_kind="multivector",
            verification_facts=frozenset({"ga_only"}),
        )
        assert op.value_kind == "multivector"
        assert "ga_only" in op.verification_facts

    def test_producer_can_leave_value_kind_unset(self) -> None:
        """The legacy producer pattern still works — no field
        explosion forced on existing code."""

        op = IROp(
            result="r",
            op_name="matmul",
            operands=["%a", "%b"],
            operand_types=["t", "t"],
        )
        # Existing audit / explain code reads ``op.value_kind`` and
        # must tolerate None without erroring.
        kind = op.value_kind  # consumer reads
        assert kind is None  # not the lying "tensor" default
