from __future__ import annotations

import pytest

from tessera.compiler.autotune_v2 import (
    BayesianAutotuner,
    GEMMWorkload,
    TuningConfig,
    TuningResult,
)
from tessera.compiler.benchmark_row import MeasuredResourceVector
from tessera.compiler.composition_cost import (
    COMPOSITION_MODEL,
    EXHAUSTIVE_ORACLE_ACTION_LIMIT,
    CompositionCalibration,
    CompositionCandidate,
    TileAction,
    compare_inferred_action_dag,
    estimate_composition,
    infer_action_dag,
    prune_composition_candidates,
)
from tessera.compiler.graph_ir import GraphIRFunction, IRArg, IROp, tensor_ir_type


def _digest(char: str) -> str:
    return char * 64


def _calibration() -> CompositionCalibration:
    return CompositionCalibration(
        memory_bytes_per_ms=1000.0,
        communication_bytes_per_ms=100.0,
        provenance={"source": "exact_device_packet", "domain": "device"},
        digest=_digest("c"),
    )


def _action(
    action_id: str,
    *,
    compute_ms: float,
    memory_bytes: int = 0,
    communication_bytes: int = 0,
    queue: str = "compute:0",
    resource: str = "device:0",
    depends_on: tuple[str, ...] = (),
    digest_char: str = "a",
) -> TileAction:
    vector = MeasuredResourceVector(
        compute_time_ms=compute_ms,
        bytes_moved=memory_bytes,
        communication_bytes=communication_bytes,
        queue_identity=queue,
        resource_identity=resource,
        timing_provenance={"source": "hip_event", "domain": "device"},
        artifact_digest=_digest(digest_char),
    ).as_dict()
    return TileAction(action_id, vector, depends_on)


def test_action_dag_search_finds_overlap_across_independent_resource_lanes():
    candidate = CompositionCandidate("overlap", (
        _action(
            "network",
            compute_ms=0.01,
            communication_bytes=1000,
            queue="comm:0",
        ),
        _action("compute", compute_ms=8.0, queue="compute:0"),
        _action("consume", compute_ms=1.0, depends_on=("network", "compute")),
    ))

    estimate = estimate_composition(candidate, _calibration())

    # network=max(.01, 10ms) overlaps with 8ms compute; the consumer follows.
    assert estimate.predicted_makespan_ms == pytest.approx(11.0)
    assert estimate.orders_examined == 2
    assert estimate.exhaustive
    assert estimate.action_order[-1] == "consume"
    assert estimate.method == COMPOSITION_MODEL
    assert estimate.selector_authority == "latency_ms"
    assert not estimate.eligible_for_promotion


def test_same_queue_actions_are_serial_even_without_data_dependency():
    candidate = CompositionCandidate("serial", (
        _action("left", compute_ms=4.0),
        _action("right", compute_ms=6.0),
    ))
    estimate = estimate_composition(candidate, _calibration())
    assert estimate.predicted_makespan_ms == pytest.approx(10.0)


def test_composition_can_reverse_standalone_scalar_order_without_selecting():
    # TileRT M3 counterexample: the surrounding step already carries 8ms of
    # network work. B is the faster standalone kernel (4ms versus 10ms), but
    # adds another 4ms to the bottleneck network lane.
    surrounding_a = _action(
        "surrounding_a",
        compute_ms=0.001,
        communication_bytes=800,
        queue="comm:surrounding",
    )
    surrounding_b = _action(
        "surrounding_b",
        compute_ms=0.001,
        communication_bytes=800,
        queue="comm:surrounding",
    )
    kernel_a = _action("kernel_a", compute_ms=10.0, queue="compute:a")
    kernel_b = _action(
        "kernel_b",
        compute_ms=4.0,
        communication_bytes=400,
        queue="comm:b",
    )
    candidate_a = CompositionCandidate("a", (surrounding_a, kernel_a))
    candidate_b = CompositionCandidate("b", (surrounding_b, kernel_b))

    estimate_a = estimate_composition(candidate_a, _calibration())
    estimate_b = estimate_composition(candidate_b, _calibration())

    assert kernel_b.resource_vector["compute_time_ms"] < kernel_a.resource_vector[
        "compute_time_ms"
    ]
    assert estimate_a.predicted_makespan_ms == pytest.approx(10.001)
    assert estimate_b.predicted_makespan_ms == pytest.approx(12.0)
    assert not estimate_a.eligible_for_promotion
    assert not estimate_b.eligible_for_promotion


def test_pruning_removes_only_exhaustively_analyzed_clear_losers():
    fast = CompositionCandidate("fast", (_action("f", compute_ms=4.0),))
    near = CompositionCandidate("near", (_action("n", compute_ms=4.5),))
    slow = CompositionCandidate("slow", (_action("s", compute_ms=8.0),))

    result = prune_composition_candidates(
        (slow, near, fast), _calibration(), relative_margin=0.25
    )

    assert result.retained == ("near", "fast")
    assert result.pruned == ("slow",)
    assert result.selector_authority == "latency_ms"
    assert not result.eligible_for_promotion
    assert not hasattr(result, "winner")


def test_bounded_nonexhaustive_search_retains_candidate():
    wide = CompositionCandidate("wide", tuple(
        _action(
            f"a{i}",
            compute_ms=1.0,
            queue=f"compute:{i}",
            resource=f"device:{i}",
        )
        for i in range(4)
    ))
    exact = CompositionCandidate("exact", (_action("e", compute_ms=1.0),))

    result = prune_composition_candidates(
        (wide, exact), _calibration(), relative_margin=0.0, max_orders=2
    )

    wide_estimate = next(e for e in result.estimates if e.candidate_id == "wide")
    assert not wide_estimate.exhaustive
    assert "wide" in result.retained


def test_w5_2g_wide_dag_uses_deterministic_list_scheduler_not_factorial_search():
    actions = tuple(
        _action(
            f"a{i:02d}",
            compute_ms=float(i + 1),
            queue=f"compute:{i}",
            resource=f"device:{i}",
            digest_char="a",
        )
        for i in range(EXHAUSTIVE_ORACLE_ACTION_LIMIT + 4)
    )
    candidate = CompositionCandidate("wide_list", actions)

    first = estimate_composition(candidate, _calibration())
    second = estimate_composition(candidate, _calibration())

    assert first.search_method == "critical_path_list"
    assert not first.exhaustive
    assert first.orders_examined == 1
    assert first.action_order == second.action_order
    assert first.predicted_makespan_ms == pytest.approx(12.0)
    assert first.lower_bound_ms == pytest.approx(12.0)
    assert first.list_schedule_makespan_ms == pytest.approx(12.0)


def test_w5_2g_small_dag_retains_exhaustive_oracle_and_bounds_list_schedule():
    candidate = CompositionCandidate("small_oracle", (
        _action("root", compute_ms=2.0),
        _action("long", compute_ms=7.0, queue="compute:1", depends_on=("root",)),
        _action("short", compute_ms=3.0, queue="compute:2", depends_on=("root",)),
        _action("join", compute_ms=1.0, depends_on=("long", "short")),
    ))

    estimate = estimate_composition(candidate, _calibration())

    assert estimate.search_method == "exhaustive_small_dag_oracle"
    assert estimate.exhaustive
    assert estimate.orders_examined == 2
    assert estimate.lower_bound_ms <= estimate.predicted_makespan_ms
    assert estimate.predicted_makespan_ms <= estimate.list_schedule_makespan_ms


def test_w5_2g_lower_bound_can_safely_prune_a_wide_inexact_candidate():
    serial = tuple(
        _action(
            f"serial:{i}",
            compute_ms=2.0,
            depends_on=((f"serial:{i - 1}",) if i else ()),
            digest_char="b",
        )
        for i in range(EXHAUSTIVE_ORACLE_ACTION_LIMIT + 1)
    )
    slow = CompositionCandidate("provably_slow", serial)
    fast = CompositionCandidate("fast", (_action("fast", compute_ms=1.0),))

    result = prune_composition_candidates(
        (slow, fast), _calibration(), relative_margin=0.0
    )

    slow_estimate = result.estimates[0]
    assert not slow_estimate.exhaustive
    assert slow_estimate.lower_bound_ms == pytest.approx(18.0)
    assert result.pruned == ("provably_slow",)


def test_dag_rejects_unknown_dependencies_and_cycles():
    with pytest.raises(ValueError, match="unknown dependencies"):
        CompositionCandidate("missing", (
            _action("a", compute_ms=1.0, depends_on=("missing",)),
        ))
    with pytest.raises(ValueError, match="cycle"):
        CompositionCandidate("cycle", (
            _action("a", compute_ms=1.0, depends_on=("b",)),
            _action("b", compute_ms=1.0, depends_on=("a",)),
        ))


def test_action_requires_measured_resource_vector_from_benchmark_row():
    with pytest.raises(ValueError, match="measured resource vector"):
        TileAction.from_benchmark_row(
            "a", {"hot_path_metadata": {}}, depends_on=()
        )


def test_r3_consumes_the_r2_autotune_record_without_schema_translation():
    tuner = BayesianAutotuner(GEMMWorkload(
        64,
        64,
        64,
        arch="gfx1151",
        movement={
            "queue_identity": "compute:gfx1151:0",
            "resource_identity": "device:gfx1151:0",
        },
    ))
    tuner._results.append(TuningResult(
        TuningConfig(32, 32, 32),
        latency_ms=0.05,
        tflops=1.0,
        method="measured",
        timing_provenance={"source": "hip_event", "domain": "device"},
    ))

    action = TileAction.from_benchmark_row("gemm", tuner.cost_measurements()[0])
    estimate = estimate_composition(
        CompositionCandidate("scheduled_gemm", (action,)), _calibration()
    )

    assert estimate.predicted_makespan_ms >= 0.05
    assert estimate.selector_authority == "latency_ms"
    assert not estimate.eligible_for_promotion


def _graph_op(result: str, name: str, operands: list[str], **kwargs) -> IROp:
    ty = tensor_ir_type(("4",), "fp32")
    return IROp(
        result=result,
        op_name=name,
        operands=operands,
        operand_types=[str(ty)] * len(operands),
        result_type=str(ty),
        inferred_type=ty,
        kwargs=kwargs,
    )


def _vector(char: str) -> dict:
    return _action(char, compute_ms=1.0, digest_char=char).resource_vector


def test_w5_2e_infers_ssa_edges_but_preserves_pure_parallelism():
    ty = tensor_ir_type(("4",), "fp32")
    left = _graph_op("left", "tessera.add", ["%x", "%x"])
    right = _graph_op("right", "tessera.mul", ["%y", "%y"])
    joined = _graph_op("out", "tessera.add", ["%left", "%right"])
    fn = GraphIRFunction(
        "f",
        args=[IRArg("x", ty), IRArg("y", ty)],
        result_types=[ty],
        body=[left, right, joined],
        return_values=["%out"],
    )

    inferred = infer_action_dag(
        fn, (_vector("a"), _vector("b"), _vector("c")),
        action_ids=("left", "right", "joined"),
    )

    assert inferred.actions[0].depends_on == ()
    assert inferred.actions[1].depends_on == ()
    assert set(inferred.actions[2].depends_on) == {"left", "right"}
    assert all("ssa_value_flow" in edge.reasons for edge in inferred.dependencies)
    assert len(inferred.graph_analysis_digest) == 64

    candidate, consumed = CompositionCandidate.from_graph(
        "physical_family", fn, (_vector("a"), _vector("b"), _vector("c")),
        action_ids=("left", "right", "joined"),
    )
    assert candidate.actions == consumed.actions
    assert set(candidate.actions[-1].depends_on) == {"left", "right"}
    assert candidate.schedule_object is not None
    assert len(candidate.schedule_object.digest) == 64
    assert candidate.schedule_object.edges

    reference = (
        _action("left", compute_ms=1.0),
        _action("right", compute_ms=1.0),
        _action("joined", compute_ms=1.0, depends_on=("left", "right")),
    )
    parity = compare_inferred_action_dag(inferred, reference)
    assert parity.exact


def test_w5_2e_unknown_alias_facts_add_conservative_reference_edges():
    ty = tensor_ir_type(("4",), "fp32")
    left = _graph_op("left", "tessera.add", ["%x", "%x"])
    unknown = _graph_op("opaque", "tessera.unknown_external", ["%hidden"])
    right = _graph_op("right", "tessera.mul", ["%y", "%y"])
    fn = GraphIRFunction(
        "f",
        args=[IRArg("x", ty), IRArg("y", ty)],
        result_types=[ty],
        body=[left, unknown, right],
        return_values=["%right"],
    )
    inferred = infer_action_dag(
        fn,
        (_vector("a"), _vector("b"), _vector("c")),
        action_ids=("left", "opaque", "right"),
    )
    reference = (
        _action("left", compute_ms=1.0),
        _action("opaque", compute_ms=1.0, depends_on=("left",)),
        _action("right", compute_ms=1.0, depends_on=("opaque",)),
    )
    parity = compare_inferred_action_dag(inferred, reference)
    assert parity.conservative
    assert parity.exact
    assert any(
        "unknown_alias_fact" in edge.reasons for edge in inferred.dependencies
    )


@pytest.mark.parametrize(
    ("op_name", "kwargs", "reason"),
    [
        ("tessera.randn", {"tessera.stochastic_identity": "seed_counter"},
         "stochastic_identity"),
        ("tessera.scf.if.begin", {"region": "then"}, "region_boundary"),
        ("tessera.unknown_external", {}, "unregistered_effect"),
    ],
)
def test_w5_2e_fail_closed_barriers_serialize_surrounding_actions(
    op_name: str, kwargs: dict, reason: str
):
    ty = tensor_ir_type(("4",), "fp32")
    before = _graph_op("before", "tessera.add", ["%x", "%x"])
    barrier = _graph_op("barrier", op_name, ["%y"], **kwargs)
    after = _graph_op("after", "tessera.mul", ["%z", "%z"])
    fn = GraphIRFunction(
        "f",
        args=[IRArg("x", ty), IRArg("y", ty), IRArg("z", ty)],
        result_types=[ty], body=[before, barrier, after],
        return_values=["%after"],
    )
    inferred = infer_action_dag(
        fn, (_vector("a"), _vector("b"), _vector("c")),
        action_ids=("before", "barrier", "after"),
    )
    edge_reasons = {
        (edge.predecessor, edge.successor): set(edge.reasons)
        for edge in inferred.dependencies
    }
    assert reason in edge_reasons[("before", "barrier")]
    assert reason in edge_reasons[("barrier", "after")]


def test_ordered_collectives_order_against_each_other_not_local_work():
    """CORRECTED SEMANTICS (PR #625 review).

    An ordered collective constrains the order OF COLLECTIVES — every rank
    must issue them in the same relative order. It does not pin unrelated
    local work: data flowing through a collective is ordered by the
    SSA/alias/memory-dependence edges instead. The old blanket rule
    (serialize against every surrounding op) contradicted the plan's own
    overlap requirement — W5.2d's MegaMoE row REQUIRES dispatch(c+1) to
    precede combine(c) — and made any pipeline containing transport infer a
    total chain, erasing the compute/communication overlap R3 exists to
    model. Measured before the fix: 12 MegaMoE actions inferred all 66
    edges of the complete order.
    """
    ty = tensor_ir_type(("4",), "fp32")
    first = _graph_op("first", "tessera.all_reduce", ["%x"])
    local = _graph_op("local", "tessera.mul", ["%y", "%y"])
    second = _graph_op("second", "tessera.all_reduce", ["%z"])
    fn = GraphIRFunction(
        "f",
        args=[IRArg("x", ty), IRArg("y", ty), IRArg("z", ty)],
        result_types=[ty], body=[first, local, second],
        return_values=["%second"],
    )
    inferred = infer_action_dag(
        fn, (_vector("a"), _vector("b"), _vector("c")),
        action_ids=("first", "local", "second"),
    )
    edges = {
        (edge.predecessor, edge.successor): set(edge.reasons)
        for edge in inferred.dependencies
    }
    # The two collectives keep their relative order...
    assert "ordered_collective" in edges[("first", "second")]
    # ...and independent local work is free to overlap either of them.
    assert ("first", "local") not in edges
    assert ("local", "second") not in edges


def test_pure_op_with_disjoint_values_has_no_memory_dependence():
    """A REGISTERED pure op touches no hidden state, so it cannot depend on
    an effectful op through memory unless they share a value. Without this
    every effectful op serialized all surrounding pure work (PR #625)."""
    from tessera.compiler.graph_dataflow import analyze_graph_dataflow

    ty = tensor_ir_type(("4",), "fp32")
    collective = _graph_op("collective", "tessera.all_reduce", ["%x"])
    disjoint = _graph_op("disjoint", "tessera.mul", ["%y", "%y"])
    shared = _graph_op("shared", "tessera.mul", ["%collective", "%y"])
    fn = GraphIRFunction(
        "f",
        args=[IRArg("x", ty), IRArg("y", ty)],
        result_types=[ty], body=[collective, disjoint, shared],
        return_values=["%shared"],
    )
    analysis = analyze_graph_dataflow(fn)
    assert not analysis.has_memory_dependence(collective, disjoint)
    # Sharing a value keeps the dependence — the refinement is alias-gated.
    assert analysis.has_memory_dependence(collective, shared)


@pytest.mark.parametrize("margin", [-0.1, float("nan"), float("inf")])
def test_pruning_rejects_invalid_margin(margin: float):
    candidate = CompositionCandidate("only", (_action("a", compute_ms=1.0),))
    with pytest.raises(ValueError, match="relative_margin"):
        prune_composition_candidates(
            (candidate,), _calibration(), relative_margin=margin
        )
