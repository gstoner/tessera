"""SO-3 (integrated-plan queue order 3) — the MegaMoE producer consumes
INFERRED dependence edges; the hand-authored plan DAG is the coverage oracle.

Before this slice, `composition_candidate_for_megamoe_plan` executed the
hand-written `_action_dependencies` edges directly. Now the chunk pipeline
is represented as Graph IR (`megamoe_graph_function`: dispatch/combine are
the registered ordered collectives, expert compute is pure matmul), the
production candidate's edges come from `infer_action_dag` (W2.1/W2.2
facts), and `compare_inferred_action_dag` must show the generated edges
cover every hand edge or construction FAILS CLOSED (#31: the hand DAG is
the declared oracle, never silently weakened). Additional conservative
edges — the ordered-collective total order — are reported separately, per
the SO-3 acceptance. The candidate carries the content-addressed
`ScheduleObject`, whose digest is the schedule identity downstream
consumers stamp.
"""

from __future__ import annotations

import pytest

from tessera.compiler import benchmark_row as br
from tessera.compiler.megamoe_overlap import (
    build_megamoe_overlap_plan,
    composition_candidate_for_megamoe_plan,
    megamoe_graph_function,
    megamoe_inferred_composition,
    megamoe_issue_order,
)


def _plan(num_chunks: int = 4):
    return build_megamoe_overlap_plan(
        plan_id="so3-megamoe",
        num_tokens=96,
        num_chunks=num_chunks,
        capacities=[32] * num_chunks,
        dispatch_buffer_bytes=[4096] * num_chunks,
    )


def _rows(plan):
    def vector(index):
        return {
            "schema": br.RESOURCE_VECTOR_SCHEMA,
            "usage": br.RESOURCE_VECTOR_USAGE,
            "selector_authority": br.SCALAR_SELECTOR_AUTHORITY,
            "compute_time_ms": 1.0 + 0.125 * index,
            "bytes_moved": 4096,
            "communication_bytes": 2048,
            "queue_identity": "hip:0",
            "resource_identity": "gfx1151",
            "timing_provenance": {"source": "device_event", "domain": "device"},
            "artifact_digest": "ab" * 32,
        }

    return {
        action: {"hot_path_metadata": {"resource_vector": vector(index)}}
        for index, action in enumerate(sorted(plan.action_dependencies))
    }


def _resource_rows(plan):
    """Rows in whichever shape TileAction.from_benchmark_row accepts."""
    rows = _rows(plan)
    from tessera.compiler.composition_cost import TileAction

    try:
        TileAction.from_benchmark_row("probe:0", next(iter(rows.values())))
        return rows
    except Exception:
        # Fall back to the flat resource-vector shape.
        return {
            action: payload["hot_path_metadata"]["resource_vector"]
            for action, payload in rows.items()
        }


def test_issue_order_is_a_deterministic_topological_order():
    plan = _plan()
    order = megamoe_issue_order(plan)
    assert sorted(order) == sorted(plan.action_dependencies)
    position = {action: index for index, action in enumerate(order)}
    for action, deps in plan.action_dependencies.items():
        for dep in deps:
            assert position[dep] < position[action], (dep, action)
    assert order == megamoe_issue_order(plan)
    # The overlap requirement is visible in the order itself.
    assert position["dispatch:1"] < position["combine:0"]


def test_graph_function_uses_registered_collective_semantics():
    from tessera.compiler.effects import Effect, registered_op_effect

    function = megamoe_graph_function(_plan())
    kinds = [op.op_name for op in function.body]
    assert kinds.count("tessera.moe_dispatch") == 4
    assert kinds.count("tessera.moe_combine") == 4
    assert kinds.count("tessera.matmul") == 4
    assert registered_op_effect("tessera.moe_dispatch", {}) == Effect.collective
    assert registered_op_effect("tessera.moe_combine", {}) == Effect.collective


def test_inferred_edges_cover_the_hand_oracle_and_report_extras():
    plan = _plan()
    candidate, inferred, parity = megamoe_inferred_composition(
        plan, _resource_rows(plan)
    )
    assert parity.conservative
    assert not parity.missing_reference_edges
    # Extra edges are reported separately, never silently merged — but they
    # must be the SOUND ones (transitive collective ordering), not a total
    # chain. See test_overlap_is_preserved below for the teeth.
    assert parity.additional_conservative_edges
    for predecessor, successor in parity.additional_conservative_edges:
        kinds = {predecessor.split(":")[0], successor.split(":")[0]}
        assert kinds <= {"dispatch", "combine"}, (predecessor, successor)
    # Every inferred dependency names its supporting reasons.
    assert all(dep.reasons for dep in inferred.dependencies)


def test_overlap_is_preserved_not_serialized():
    """PR #625 review, P1 — the defect this file previously encoded as
    healthy conservatism.

    Emitting the per-chunk transport as ordered collectives used to make
    `infer_action_dag` serialize every collective against every surrounding
    operation, so a 12-action plan inferred all 66 edges of the complete
    order: R3 would then estimate and prune MegaMoE overlap plans as
    sequential pipelines, which is exactly the property the plan exists to
    choose between. The inferred DAG must keep compute off the collective
    chain.
    """
    plan = _plan()
    _, inferred, _ = megamoe_inferred_composition(plan, _resource_rows(plan))
    actions = len(inferred.actions)
    edges = {(d.predecessor, d.successor) for d in inferred.dependencies}
    assert len(edges) < actions * (actions - 1) // 2, "inferred a total chain"
    # The canonical overlap: compute of the next chunk is NOT forced after the
    # previous chunk's combine.
    assert ("combine:0", "compute:1") not in edges
    # Independent expert compute stays independent.
    assert ("compute:0", "compute:1") not in edges
    # ...while the collectives keep their required relative order.
    assert ("dispatch:0", "dispatch:1") in edges
    assert ("combine:0", "combine:1") in edges


def test_schedule_digest_binds_plan_identity():
    """PR #625 review, P2 — action ids and graph shape are functions of the
    chunk COUNT, so two plans sharing a plan_id and chunk count but differing
    in capacities, buffer sizes, token ranges, or the in-flight limit used to
    content-address to the SAME digest with identical benchmark rows."""
    left = _plan()
    right = build_megamoe_overlap_plan(
        plan_id=left.plan_id,
        num_tokens=96,
        num_chunks=4,
        capacities=[16] * 4,
        dispatch_buffer_bytes=[8192] * 4,
    )
    assert left.artifact_digest != right.artifact_digest
    left_candidate = composition_candidate_for_megamoe_plan(
        left, _resource_rows(left))
    right_candidate = composition_candidate_for_megamoe_plan(
        right, _resource_rows(right))
    assert (left_candidate.schedule_object.digest
            != right_candidate.schedule_object.digest)
    # The candidate id stays the plan id, so pruning keys are unchanged.
    assert left_candidate.candidate_id == left.plan_id


def test_candidate_carries_the_content_addressed_schedule_object():
    plan = _plan()
    rows = _resource_rows(plan)
    first = composition_candidate_for_megamoe_plan(plan, rows)
    second = composition_candidate_for_megamoe_plan(plan, rows)
    digest = first.schedule_object.digest
    assert isinstance(digest, str) and len(digest) == 64
    assert digest == second.schedule_object.digest
    assert first.schedule_object.edges  # inferred edges, not hand edges


def test_generation_regression_fails_closed(monkeypatch):
    """Teeth: if the Graph representation ever loses the registered
    collective semantics (a generation regression), the inferred DAG stops
    covering the hand oracle's ordering edges — and construction must
    refuse rather than execute a weaker schedule than the oracle demands."""
    from tessera.compiler import megamoe_overlap as mm
    from tessera.compiler.graph_ir import GraphIRFunction, IRArg, IROp, IRType

    plan = _plan()

    def weakened_graph(p):
        # Every action becomes a PURE op over per-chunk private values: the
        # SSA chains survive, but every ordered-collective and capacity
        # edge in the hand DAG disappears from the inference.
        tensor = IRType("tensor<*xf32>")
        ops = []
        for action_id in mm.megamoe_issue_order(p):
            kind, chunk = action_id.split(":")
            prev = {
                "dispatch": f"%in{chunk}",
                "compute": f"%dispatch{chunk}",
                "combine": f"%compute{chunk}",
            }[kind]
            ops.append(
                IROp(
                    result=f"%{kind}{chunk}",
                    op_name="tessera.tanh",
                    operands=[prev],
                    operand_types=["tensor<*xf32>"],
                    result_type="tensor<*xf32>",
                )
            )
        return GraphIRFunction(
            name=p.plan_id,
            args=[IRArg(f"in{c.index}", tensor) for c in p.chunks],
            body=ops,
            return_values=[f"%combine{p.num_chunks - 1}"],
        )

    monkeypatch.setattr(mm, "megamoe_graph_function", weakened_graph)
    with pytest.raises(ValueError, match="does not cover the hand-authored"):
        mm.megamoe_inferred_composition(plan, _resource_rows(plan))


def test_missing_action_evidence_still_fails_closed():
    plan = _plan()
    rows = dict(_resource_rows(plan))
    rows.pop("compute:2")
    with pytest.raises(ValueError, match="must be total"):
        composition_candidate_for_megamoe_plan(plan, rows)
