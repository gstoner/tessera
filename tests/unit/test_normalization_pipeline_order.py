"""Phase C skeleton order test — `NORMALIZATION_PIPELINE` shape.

Locks the documented ordering of normalization passes per
``docs/architecture/frontend_substrate_plan.md`` § 3.  Reordering,
renaming, or removing a pass requires updating both the architecture
doc and this test in lock step.

This test runs against the **skeleton** — every pass body is a no-op
stub today.  Pass-body PRs that follow this commit may only fill in
function bodies; they may not change the tuple shape without
updating both files here.
"""

from __future__ import annotations

from tessera.compiler import (
    NORMALIZATION_PIPELINE,
    run_normalization_pipeline,
)
from tessera.compiler.graph_ir import GraphIRFunction, IROp
from tessera.compiler.normalization import (
    canonicalize_op_names,
    propagate_numeric_policy,
    propagate_source_positions,
    propagate_value_kinds,
    propagate_verification_facts,
    set_lane_provenance,
)


# Locked ordering — six entries in the documented sequence.
EXPECTED_ORDER: tuple[str, ...] = (
    "canonicalize_op_names",
    "propagate_source_positions",
    "set_lane_provenance",
    "propagate_value_kinds",
    "propagate_numeric_policy",
    "propagate_verification_facts",
)


class TestPipelineShape:
    def test_pipeline_is_a_tuple(self) -> None:
        assert isinstance(NORMALIZATION_PIPELINE, tuple)

    def test_pipeline_has_exactly_six_passes(self) -> None:
        assert len(NORMALIZATION_PIPELINE) == 6, (
            "Phase C skeleton declares 6 passes.  Adding or removing "
            "a pass requires updating both:\n"
            "  * docs/architecture/frontend_substrate_plan.md § 3\n"
            "  * this test's EXPECTED_ORDER tuple\n"
            "in lock step."
        )


class TestPipelineOrder:
    def test_first_pass_is_canonicalize_op_names(self) -> None:
        assert NORMALIZATION_PIPELINE[0] is canonicalize_op_names, (
            "Canonical op naming MUST run first so subsequent passes "
            "see ``matmul`` instead of ``tessera.matmul`` and can "
            "uniformly look up the catalog."
        )

    def test_last_pass_is_propagate_verification_facts(self) -> None:
        assert NORMALIZATION_PIPELINE[-1] is propagate_verification_facts, (
            "verification_facts propagation MUST run last so it can "
            "consume lane + value_kind + numeric_policy decisions."
        )

    def test_exact_order_matches_expected(self) -> None:
        actual = tuple(p.__name__ for p in NORMALIZATION_PIPELINE)
        assert actual == EXPECTED_ORDER, (
            f"Pipeline ordering drifted:\n"
            f"  expected: {EXPECTED_ORDER}\n"
            f"  actual:   {actual}\n"
            f"\n"
            f"If this drift is intentional, update both:\n"
            f"  * docs/architecture/frontend_substrate_plan.md § 3\n"
            f"  * this test's EXPECTED_ORDER tuple"
        )

    def test_canonicalize_runs_before_value_kinds(self) -> None:
        """value_kind derivation keys on canonical op names, so
        canonicalize_op_names must run first."""

        names = [p.__name__ for p in NORMALIZATION_PIPELINE]
        assert names.index("canonicalize_op_names") < names.index(
            "propagate_value_kinds"
        )

    def test_lane_provenance_runs_before_verification_facts(self) -> None:
        """verification_facts derivation depends on knowing the lane."""

        names = [p.__name__ for p in NORMALIZATION_PIPELINE]
        assert names.index("set_lane_provenance") < names.index(
            "propagate_verification_facts"
        )

    def test_numeric_policy_runs_after_canonicalize(self) -> None:
        """numeric_policy keys on canonical names (G3)."""

        names = [p.__name__ for p in NORMALIZATION_PIPELINE]
        assert names.index("canonicalize_op_names") < names.index(
            "propagate_numeric_policy"
        )


class TestSkeletonInvariants:
    """Confirm every pass exposes the documented signature even
    when its body is still a stub.  Locks the wire shape so body
    PRs only have to fill in implementations."""

    def test_each_pass_is_callable(self) -> None:
        for p in NORMALIZATION_PIPELINE:
            assert callable(p)

    def test_each_pass_accepts_graph_ir_function(self) -> None:
        """Smoke: every pass takes a GraphIRFunction and returns
        ``None`` (in-place mutation contract)."""

        fn = GraphIRFunction(name="probe")
        for p in NORMALIZATION_PIPELINE:
            result = p(fn)
            assert result is None, (
                f"Pass {p.__name__} returned {result!r}; "
                f"the contract is in-place mutation (returns None)"
            )

    def test_pipeline_is_no_op_on_empty_function(self) -> None:
        """The skeleton commit must not mutate ``fn`` — every pass
        body is a stub.  Empty function should round-trip cleanly."""

        fn = GraphIRFunction(name="empty")
        run_normalization_pipeline(fn)
        assert fn.name == "empty"
        assert fn.body == []
        assert fn.lane == "tessera_jit"
        assert fn.verification_facts == frozenset()

    def test_propagate_numeric_policy_is_wired(self) -> None:
        """The G3 pass is the only one wired today — its body
        already exists.  Verify it still works through the
        normalization wrapper."""

        fn = GraphIRFunction(name="m")
        op = IROp(
            result="r",
            op_name="matmul",
            operands=["%a", "%b"],
            operand_types=["t", "t"],
            result_type="t",
        )
        fn.body.append(op)
        propagate_numeric_policy(fn)
        # matmul's policy is bf16 storage / fp32 accum per the
        # G3 factory table.
        assert op.numeric_policy is not None
        assert op.numeric_policy.storage == "bf16"

    def test_run_normalization_pipeline_is_idempotent(self) -> None:
        """A core invariant from the architecture doc: every pass
        is idempotent.  Running the whole pipeline twice produces
        the same result."""

        fn1 = GraphIRFunction(name="m")
        fn2 = GraphIRFunction(name="m")
        # Use matmul so propagate_numeric_policy actually fires.
        for fn in (fn1, fn2):
            fn.body.append(IROp(
                result="r",
                op_name="matmul",
                operands=["%a", "%b"],
                operand_types=["t", "t"],
                result_type="t",
            ))
        run_normalization_pipeline(fn1)
        run_normalization_pipeline(fn1)  # second pass
        run_normalization_pipeline(fn2)  # single pass
        # Both functions should have the same numeric_policy.
        assert fn1.body[0].numeric_policy == fn2.body[0].numeric_policy
