"""Phase B drift gate — `to_graph_ir_view()` contract enforcement.

Per ``docs/spec/COMPILER_REFERENCE.md`` § "Constrained-lane Graph IR
views", every constrained-lane IR program exposes a
``to_graph_ir_view() -> GraphIRModule`` adapter that:

  * Projects 1:1 — each constrained ``IROpCall`` becomes exactly
    one ``IROp`` in the view, in the same order.
  * Uses canonical op names — never backend-aliased names like
    ``complex_mobius``.
  * Returns a fresh deep copy per call — mutating one view doesn't
    poison another.
  * Stamps the lane on ``view.functions[0].lane``.
  * Stamps verification facts on ``view.functions[0].verification_facts``.
  * Stamps verification facts on each ``IROp.verification_facts``
    too, so per-op consumers don't have to walk back to the function.

Adding a new constrained lane requires extending the parametrize
list below.
"""

from __future__ import annotations

import pytest

from tessera.compiler.ast_ir import IROpCall as _SharedIROpCall
from tessera.compiler.clifford_jit import (
    CliffordIROpCall,
    CliffordIRProgram,
)
from tessera.compiler.complex_jit import (
    ComplexIRProgram,
    HOLOMORPHIC_OPS,
    NON_HOLOMORPHIC_OPS,
)
from tessera.compiler.energy_jit import EnergyIRProgram
from tessera.compiler.graph_ir import GraphIRModule


# ─────────────────────────────────────────────────────────────────────
# Fixture factories — minimal valid programs per lane.
# ─────────────────────────────────────────────────────────────────────


def _clifford_program() -> CliffordIRProgram:
    return CliffordIRProgram(
        arg_names=("a", "b"),
        ops=(
            CliffordIROpCall(
                op_name="clifford_geometric_product",
                operand_refs=("a", "b"),
                result_name="t0",
                python_attr="geometric_product",
            ),
            CliffordIROpCall(
                op_name="clifford_rotor_sandwich",
                operand_refs=("t0", "a"),
                result_name="r",
                python_attr="rotor_sandwich",
            ),
        ),
        return_ref="r",
    )


def _complex_program_holomorphic() -> ComplexIRProgram:
    return ComplexIRProgram(
        arg_names=("z",),
        ops=(
            _SharedIROpCall(
                op_name="complex_mul",
                operand_refs=("z", "z"),
                result_name="t0",
                python_attr="complex_mul",
            ),
            _SharedIROpCall(
                op_name="mobius",
                operand_refs=("t0",),
                result_name="r",
                python_attr="mobius",
            ),
        ),
        return_ref="r",
    )


def _complex_program_non_holomorphic() -> ComplexIRProgram:
    return ComplexIRProgram(
        arg_names=("z",),
        ops=(
            _SharedIROpCall(
                op_name="complex_conjugate",
                operand_refs=("z",),
                result_name="r",
                python_attr="complex_conjugate",
            ),
        ),
        return_ref="r",
    )


def _energy_program() -> EnergyIRProgram:
    return EnergyIRProgram(
        arg_names=("x", "W"),
        ops=(
            _SharedIROpCall(
                op_name="energy_quadratic",
                operand_refs=("x", "W"),
                result_name="r",
                python_attr="quadratic",
            ),
        ),
        return_ref="r",
    )


# Parametrize over the three lanes with their expected lane stamp
# + verification-facts contract.
_LANE_FIXTURES = [
    pytest.param(
        _clifford_program,
        "clifford_jit",
        frozenset({"ga_whitelisted"}),
        "multivector",
        id="clifford",
    ),
    pytest.param(
        _complex_program_holomorphic,
        "complex_jit",
        frozenset({"holomorphic"}),
        "complex",
        id="complex_holomorphic",
    ),
    pytest.param(
        _energy_program,
        "energy_jit",
        frozenset({"energy_whitelisted"}),
        "energy",
        id="energy",
    ),
]


# ─────────────────────────────────────────────────────────────────────
# Contract assertions — parametrized across every constrained lane.
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "make_program,expected_lane,expected_facts,expected_value_kind",
    _LANE_FIXTURES,
)
def test_view_returns_graph_ir_module(
    make_program, expected_lane: str,
    expected_facts: frozenset[str], expected_value_kind: str,
) -> None:
    program = make_program()
    view = program.to_graph_ir_view()
    assert isinstance(view, GraphIRModule)
    assert len(view.functions) == 1


@pytest.mark.parametrize(
    "make_program,expected_lane,expected_facts,expected_value_kind",
    _LANE_FIXTURES,
)
def test_one_to_one_op_projection(
    make_program, expected_lane: str,
    expected_facts: frozenset[str], expected_value_kind: str,
) -> None:
    program = make_program()
    view = program.to_graph_ir_view()
    fn = view.functions[0]
    # Op count matches.
    assert len(fn.body) == len(program.ops)
    # Op names match in order (canonical, not backend-aliased).
    for ir_op, constrained_call in zip(fn.body, program.ops):
        assert ir_op.op_name == constrained_call.op_name
        assert ir_op.result == constrained_call.result_name
        assert ir_op.operands == list(constrained_call.operand_refs)


@pytest.mark.parametrize(
    "make_program,expected_lane,expected_facts,expected_value_kind",
    _LANE_FIXTURES,
)
def test_lane_stamping(
    make_program, expected_lane: str,
    expected_facts: frozenset[str], expected_value_kind: str,
) -> None:
    program = make_program()
    view = program.to_graph_ir_view()
    assert view.functions[0].lane == expected_lane


@pytest.mark.parametrize(
    "make_program,expected_lane,expected_facts,expected_value_kind",
    _LANE_FIXTURES,
)
def test_verification_facts_on_function(
    make_program, expected_lane: str,
    expected_facts: frozenset[str], expected_value_kind: str,
) -> None:
    program = make_program()
    view = program.to_graph_ir_view()
    assert view.functions[0].verification_facts == expected_facts


@pytest.mark.parametrize(
    "make_program,expected_lane,expected_facts,expected_value_kind",
    _LANE_FIXTURES,
)
def test_verification_facts_propagated_to_ops(
    make_program, expected_lane: str,
    expected_facts: frozenset[str], expected_value_kind: str,
) -> None:
    program = make_program()
    view = program.to_graph_ir_view()
    for ir_op in view.functions[0].body:
        assert ir_op.verification_facts == expected_facts


@pytest.mark.parametrize(
    "make_program,expected_lane,expected_facts,expected_value_kind",
    _LANE_FIXTURES,
)
def test_value_kind_propagated_to_ops(
    make_program, expected_lane: str,
    expected_facts: frozenset[str], expected_value_kind: str,
) -> None:
    program = make_program()
    view = program.to_graph_ir_view()
    for ir_op in view.functions[0].body:
        assert ir_op.value_kind == expected_value_kind


@pytest.mark.parametrize(
    "make_program,expected_lane,expected_facts,expected_value_kind",
    _LANE_FIXTURES,
)
def test_fresh_deepcopy_per_call(
    make_program, expected_lane: str,
    expected_facts: frozenset[str], expected_value_kind: str,
) -> None:
    """Per the contract, two successive calls return distinct
    mutable trees.  Mutating one must never affect the other."""

    program = make_program()
    view_a = program.to_graph_ir_view()
    view_b = program.to_graph_ir_view()
    assert view_a is not view_b
    assert view_a.functions is not view_b.functions
    assert view_a.functions[0] is not view_b.functions[0]
    assert view_a.functions[0].body is not view_b.functions[0].body
    # Mutate view_a's body — view_b stays intact.
    view_a.functions[0].body.clear()
    assert len(view_b.functions[0].body) == len(program.ops)


@pytest.mark.parametrize(
    "make_program,expected_lane,expected_facts,expected_value_kind",
    _LANE_FIXTURES,
)
def test_return_ref_propagated(
    make_program, expected_lane: str,
    expected_facts: frozenset[str], expected_value_kind: str,
) -> None:
    program = make_program()
    view = program.to_graph_ir_view()
    assert view.functions[0].return_values == [program.return_ref]


# ─────────────────────────────────────────────────────────────────────
# Lane-specific tests — invariants that only one lane carries.
# ─────────────────────────────────────────────────────────────────────


class TestCanonicalOpNamesOnly:
    """The view must never emit backend-aliased names (e.g.,
    `complex_mobius`).  Aliases live in
    ``_M7_BACKEND_ALIASES`` and are applied by the audit walker, not
    by the view adapter."""

    _BACKEND_ALIASES_TO_REJECT = frozenset({
        "complex_mobius",
        "complex_stereographic",
    })

    def test_complex_view_uses_canonical_mobius_name(self) -> None:
        # Build a program that uses the canonical "mobius" name —
        # the view must preserve it as-is.
        program = ComplexIRProgram(
            arg_names=("z",),
            ops=(
                _SharedIROpCall(
                    op_name="mobius",
                    operand_refs=("z",),
                    result_name="r",
                    python_attr="mobius",
                ),
            ),
            return_ref="r",
        )
        view = program.to_graph_ir_view()
        op_names = [op.op_name for op in view.functions[0].body]
        assert "mobius" in op_names
        for backend_alias in self._BACKEND_ALIASES_TO_REJECT:
            assert backend_alias not in op_names, (
                f"View emitted backend-aliased name {backend_alias!r} "
                f"— must use canonical names only.  See "
                f"docs/spec/COMPILER_REFERENCE.md § Constrained-lane "
                f"Graph IR views."
            )


class TestHolomorphicFactConditional:
    """The Complex view's ``holomorphic`` fact fires only when
    every op is in :data:`HOLOMORPHIC_OPS`.  Programs with even
    one anti-holomorphic op project without the fact."""

    def test_all_holomorphic_program_carries_fact(self) -> None:
        program = _complex_program_holomorphic()
        for op in program.ops:
            assert op.op_name in HOLOMORPHIC_OPS
        view = program.to_graph_ir_view()
        assert "holomorphic" in view.functions[0].verification_facts

    def test_non_holomorphic_program_omits_fact(self) -> None:
        program = _complex_program_non_holomorphic()
        # Sanity: the test fixture must actually be non-holomorphic.
        assert any(op.op_name in NON_HOLOMORPHIC_OPS for op in program.ops)
        view = program.to_graph_ir_view()
        assert "holomorphic" not in view.functions[0].verification_facts


class TestEmptyProgram:
    """A constrained IR with zero ops still produces a valid view —
    just an empty body."""

    def test_clifford_empty_program(self) -> None:
        program = CliffordIRProgram(
            arg_names=("x",), ops=(), return_ref="x",
        )
        view = program.to_graph_ir_view()
        assert view.functions[0].body == []
        assert view.functions[0].lane == "clifford_jit"
        assert view.functions[0].verification_facts == frozenset(
            {"ga_whitelisted"}
        )
