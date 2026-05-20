"""Phase C body tests — one test class per pass.

Each pass body has two locked behaviors:

  1. The pass produces the documented mutation.
  2. The pass is **idempotent** (running it twice == running it once)
     and **producer-respecting** (existing values are kept).

These tests intentionally use synthetic ``IROp`` instances rather
than going through ``@tessera.jit``, so the behavior under test
isolates the pass itself.
"""

from __future__ import annotations

import pytest

from tessera.compiler.graph_ir import (
    GraphIRFunction,
    IROp,
    SourceSpan,
)
from tessera.compiler.normalization import (
    canonicalize_op_names,
    propagate_source_positions,
    propagate_value_kinds,
    propagate_verification_facts,
    run_normalization_pipeline,
    set_lane_provenance,
)


def _op(
    op_name: str,
    *,
    result: str = "r",
    operands: list[str] | None = None,
    **kwargs,
) -> IROp:
    return IROp(
        op_name=op_name,
        operands=operands if operands is not None else ["%a", "%b"],
        operand_types=(
            ["t"] * len(operands) if operands is not None else ["t", "t"]
        ),
        result=result,
        result_type="t",
        **kwargs,
    )


class TestCanonicalizeOpNames:
    def test_strips_tessera_prefix(self) -> None:
        fn = GraphIRFunction(name="f")
        fn.body.append(_op("tessera.matmul"))
        canonicalize_op_names(fn)
        assert fn.body[0].op_name == "matmul"

    def test_leaves_non_prefixed_alone(self) -> None:
        fn = GraphIRFunction(name="f")
        fn.body.append(_op("matmul"))
        canonicalize_op_names(fn)
        assert fn.body[0].op_name == "matmul"

    def test_does_not_double_strip(self) -> None:
        """A name that happens to contain 'tessera.' in the middle
        (e.g., an explicit namespace prefix) is left alone — only
        a leading ``tessera.`` is stripped."""

        fn = GraphIRFunction(name="f")
        fn.body.append(_op("ns.tessera.matmul"))
        canonicalize_op_names(fn)
        assert fn.body[0].op_name == "ns.tessera.matmul"

    def test_idempotent(self) -> None:
        fn = GraphIRFunction(name="f")
        fn.body.append(_op("tessera.matmul"))
        canonicalize_op_names(fn)
        canonicalize_op_names(fn)
        assert fn.body[0].op_name == "matmul"

    def test_multiple_ops(self) -> None:
        fn = GraphIRFunction(name="f")
        fn.body.append(_op("tessera.matmul", result="t0"))
        fn.body.append(_op("tessera.relu", result="t1", operands=["%t0"]))
        fn.body.append(_op("softmax", result="r", operands=["%t1"]))
        canonicalize_op_names(fn)
        names = [op.op_name for op in fn.body]
        assert names == ["matmul", "relu", "softmax"]


class TestPropagateSourcePositions:
    def test_inherits_from_unique_producer(self) -> None:
        fn = GraphIRFunction(name="f")
        upstream = _op(
            "matmul", result="t0",
            source_span=SourceSpan(line=42, col=5),
        )
        downstream = _op("relu", result="r", operands=["%t0"])
        fn.body.extend([upstream, downstream])
        propagate_source_positions(fn)
        assert downstream.source_span is not None
        assert downstream.source_span.line == 42

    def test_leaves_already_set_alone(self) -> None:
        original = SourceSpan(line=10, col=2)
        fn = GraphIRFunction(name="f")
        fn.body.append(_op(
            "matmul",
            source_span=original,
            operands=["%a"],
        ))
        propagate_source_positions(fn)
        assert fn.body[0].source_span is original

    def test_leaves_none_when_no_producer_has_span(self) -> None:
        fn = GraphIRFunction(name="f")
        fn.body.append(_op("matmul", result="t0"))  # no span
        fn.body.append(_op("relu", result="r", operands=["%t0"]))
        propagate_source_positions(fn)
        assert fn.body[1].source_span is None

    def test_idempotent(self) -> None:
        fn = GraphIRFunction(name="f")
        upstream = _op(
            "matmul", result="t0",
            source_span=SourceSpan(line=7, col=1),
        )
        downstream = _op("relu", result="r", operands=["%t0"])
        fn.body.extend([upstream, downstream])
        propagate_source_positions(fn)
        first_span = downstream.source_span
        propagate_source_positions(fn)
        assert downstream.source_span is first_span

    def test_handles_bare_ref_form(self) -> None:
        """Some producers emit operand refs without the ``%``
        prefix.  The pass should resolve both forms."""

        fn = GraphIRFunction(name="f")
        upstream = _op(
            "matmul", result="t0",
            source_span=SourceSpan(line=1, col=1),
        )
        # Operand uses "t0" not "%t0"
        downstream = _op("relu", result="r", operands=["t0"])
        fn.body.extend([upstream, downstream])
        propagate_source_positions(fn)
        assert downstream.source_span is not None


class TestSetLaneProvenance:
    def test_default_lane_is_preserved(self) -> None:
        fn = GraphIRFunction(name="f")
        # Default is "tessera_jit" per the dataclass.
        set_lane_provenance(fn)
        assert fn.lane == "tessera_jit"

    def test_empty_string_upgraded_to_tessera_jit(self) -> None:
        fn = GraphIRFunction(name="f", lane="")
        set_lane_provenance(fn)
        assert fn.lane == "tessera_jit"

    def test_unknown_lane_upgraded_to_tessera_jit(self) -> None:
        fn = GraphIRFunction(name="f", lane="not_a_real_lane")
        set_lane_provenance(fn)
        assert fn.lane == "tessera_jit"

    @pytest.mark.parametrize(
        "lane",
        [
            "tessera_jit",
            "textual_dsl",
            "clifford_jit",
            "complex_jit",
            "energy_jit",
        ],
    )
    def test_real_lane_is_preserved(self, lane: str) -> None:
        fn = GraphIRFunction(name="f", lane=lane)
        set_lane_provenance(fn)
        assert fn.lane == lane

    def test_idempotent(self) -> None:
        fn = GraphIRFunction(name="f", lane="complex_jit")
        set_lane_provenance(fn)
        set_lane_provenance(fn)
        assert fn.lane == "complex_jit"


class TestPropagateValueKinds:
    @pytest.mark.parametrize(
        "op_name,expected_kind",
        [
            ("clifford_geometric_product", "multivector"),
            ("clifford_rotor_sandwich", "multivector"),
            ("complex_mul", "complex"),
            ("complex_conjugate", "complex"),
            ("mobius", "complex"),
            ("stereographic", "complex"),
            ("cross_ratio", "complex"),
            ("dbar", "complex"),
            ("energy_quadratic", "energy"),
            ("ebm_langevin_step", "energy"),
            # General tensor ops in the catalog → "tensor"
            ("matmul", "tensor"),
            ("relu", "tensor"),
            ("softmax", "tensor"),
        ],
    )
    def test_derives_kind_from_op_name(
        self, op_name: str, expected_kind: str,
    ) -> None:
        fn = GraphIRFunction(name="f")
        fn.body.append(_op(op_name))
        propagate_value_kinds(fn)
        assert fn.body[0].value_kind == expected_kind, op_name

    def test_unknown_op_leaves_none(self) -> None:
        """Per architecture refinement: don't lie via a default
        like 'tensor'.  An op the catalog doesn't know stays None."""

        fn = GraphIRFunction(name="f")
        fn.body.append(_op("some_unknown_op"))
        propagate_value_kinds(fn)
        assert fn.body[0].value_kind is None

    def test_producer_set_value_kind_is_respected(self) -> None:
        """View adapters set value_kind explicitly — the pass must
        not overwrite their choice."""

        fn = GraphIRFunction(name="f")
        fn.body.append(_op("matmul", value_kind="mixed"))
        propagate_value_kinds(fn)
        assert fn.body[0].value_kind == "mixed"

    def test_idempotent(self) -> None:
        fn = GraphIRFunction(name="f")
        fn.body.append(_op("matmul"))
        propagate_value_kinds(fn)
        first = fn.body[0].value_kind
        propagate_value_kinds(fn)
        assert fn.body[0].value_kind == first


class TestPropagateVerificationFacts:
    def test_clifford_lane_stamps_ga_only(self) -> None:
        fn = GraphIRFunction(name="f", lane="clifford_jit")
        fn.body.append(_op("clifford_geometric_product"))
        propagate_verification_facts(fn)
        assert "ga_only" in fn.body[0].verification_facts

    def test_complex_lane_stamps_holomorphic_only_for_holomorphic_ops(
        self,
    ) -> None:
        fn = GraphIRFunction(name="f", lane="complex_jit")
        fn.body.append(_op("complex_mul"))          # holomorphic
        fn.body.append(_op("complex_conjugate"))    # NOT holomorphic
        propagate_verification_facts(fn)
        assert "holomorphic" in fn.body[0].verification_facts
        assert fn.body[1].verification_facts == frozenset()

    def test_energy_lane_stamps_energy_whitelisted(self) -> None:
        fn = GraphIRFunction(name="f", lane="energy_jit")
        fn.body.append(_op("energy_quadratic"))
        fn.body.append(_op("ebm_langevin_step"))
        propagate_verification_facts(fn)
        assert "energy_whitelisted" in fn.body[0].verification_facts
        assert "energy_whitelisted" in fn.body[1].verification_facts

    def test_tessera_jit_lane_stamps_nothing(self) -> None:
        """The general lane has no whitelist invariant to claim."""

        fn = GraphIRFunction(name="f", lane="tessera_jit")
        fn.body.append(_op("matmul"))
        propagate_verification_facts(fn)
        assert fn.body[0].verification_facts == frozenset()

    def test_producer_set_facts_respected(self) -> None:
        fn = GraphIRFunction(name="f", lane="complex_jit")
        fn.body.append(_op(
            "complex_mul",
            verification_facts=frozenset({"holomorphic", "custom_fact"}),
        ))
        propagate_verification_facts(fn)
        # Producer's set kept verbatim — pass does not add or remove.
        assert fn.body[0].verification_facts == frozenset(
            {"holomorphic", "custom_fact"}
        )

    def test_idempotent(self) -> None:
        fn = GraphIRFunction(name="f", lane="clifford_jit")
        fn.body.append(_op("clifford_geometric_product"))
        propagate_verification_facts(fn)
        first = fn.body[0].verification_facts
        propagate_verification_facts(fn)
        assert fn.body[0].verification_facts == first


class TestFullPipelineEndToEnd:
    """End-to-end: run the full pipeline on a synthetic function and
    verify each pass observed the previous passes' work."""

    def test_pipeline_normalizes_synthetic_clifford_function(self) -> None:
        fn = GraphIRFunction(name="cliff", lane="clifford_jit")
        fn.body.append(_op(
            "tessera.clifford_geometric_product",  # prefixed
            result="t0",
            source_span=SourceSpan(line=10, col=5),
        ))
        fn.body.append(_op(
            "tessera.clifford_rotor_sandwich",  # prefixed, no span
            result="r",
            operands=["%t0", "%a"],
        ))
        run_normalization_pipeline(fn)
        # canonicalize: prefix stripped
        assert fn.body[0].op_name == "clifford_geometric_product"
        assert fn.body[1].op_name == "clifford_rotor_sandwich"
        # propagate_source_positions: downstream inherits upstream
        assert fn.body[1].source_span is not None
        assert fn.body[1].source_span.line == 10
        # propagate_value_kinds: clifford_* → multivector
        assert fn.body[0].value_kind == "multivector"
        # propagate_verification_facts: clifford_jit + clifford_* → ga_only
        assert "ga_only" in fn.body[0].verification_facts
        # Function-level lane unchanged
        assert fn.lane == "clifford_jit"

    def test_pipeline_idempotent_on_complex_function(self) -> None:
        fn = GraphIRFunction(name="hol", lane="complex_jit")
        fn.body.append(_op(
            "tessera.complex_mul",
            source_span=SourceSpan(line=1, col=1),
        ))
        run_normalization_pipeline(fn)
        snapshot = (
            fn.body[0].op_name,
            fn.body[0].value_kind,
            fn.body[0].verification_facts,
        )
        run_normalization_pipeline(fn)
        assert (
            fn.body[0].op_name,
            fn.body[0].value_kind,
            fn.body[0].verification_facts,
        ) == snapshot
