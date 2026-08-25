"""Executable MegaMoE chunk/action plans and R3 composition integration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import heapq

from .composition_cost import (
    EXHAUSTIVE_ORACLE_MAX_ORDERS,
    ActionDAGParity,
    CompositionCalibration,
    CompositionCandidate,
    CompositionPruningResult,
    InferredActionDAG,
    TileAction,
    compare_inferred_action_dag,
    prune_composition_candidates,
)
from .graph_ir import GraphIRFunction, IRArg, IROp, IRType
from .schedule_object import ScheduleObject

MEGAMOE_OVERLAP_PLAN_SCHEMA = "tessera.megamoe_overlap_plan.v1"


@dataclass(frozen=True)
class MegaMoEChunk:
    index: int
    start: int
    stop: int
    expert_capacity: int
    dispatch_buffer_bytes: int

    def __post_init__(self) -> None:
        if self.index < 0 or self.start < 0 or self.stop <= self.start:
            raise ValueError("MegaMoE chunks require a valid index and non-empty range")
        if self.expert_capacity < 1:
            raise ValueError("MegaMoE expert capacity must be >= 1")
        if self.dispatch_buffer_bytes < 0:
            raise ValueError("MegaMoE dispatch buffer bytes must be non-negative")


@dataclass(frozen=True)
class MegaMoEOverlapPlan:
    plan_id: str
    num_tokens: int
    pipeline_stages: int
    max_in_flight_chunks: int
    chunks: tuple[MegaMoEChunk, ...]
    action_dependencies: Mapping[str, tuple[str, ...]]
    deterministic_combine: bool
    max_live_dispatch_bytes: int
    artifact_digest: str
    schema: str = MEGAMOE_OVERLAP_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise ValueError("MegaMoE plan_id must be non-empty")
        if self.num_tokens < 1:
            raise ValueError("MegaMoE num_tokens must be >= 1")
        if self.pipeline_stages not in (1, 2):
            raise ValueError("MegaMoE pipeline_stages must be 1 or 2")
        if self.max_in_flight_chunks < 1:
            raise ValueError("MegaMoE max_in_flight_chunks must be >= 1")
        if not self.deterministic_combine:
            raise ValueError("MegaMoE combine order must be deterministic")
        if self.max_live_dispatch_bytes < 0:
            raise ValueError("MegaMoE live dispatch bytes must be non-negative")
        if self.schema != MEGAMOE_OVERLAP_PLAN_SCHEMA:
            raise ValueError("unsupported MegaMoE overlap-plan schema")
        expected_start = 0
        for expected_index, chunk in enumerate(self.chunks):
            if chunk.index != expected_index or chunk.start != expected_start:
                raise ValueError("MegaMoE chunks must be contiguous and index ordered")
            expected_start = chunk.stop
        if expected_start != self.num_tokens:
            raise ValueError("MegaMoE chunks must cover every token exactly once")
        expected_actions = {
            f"{kind}:{chunk.index}"
            for chunk in self.chunks
            for kind in ("dispatch", "compute", "combine")
        }
        if set(self.action_dependencies) != expected_actions:
            raise ValueError(
                "MegaMoE action DAG must cover every chunk phase exactly once"
            )
        for action, dependencies in self.action_dependencies.items():
            if action in dependencies or not set(dependencies).issubset(
                expected_actions
            ):
                raise ValueError("MegaMoE action DAG has invalid dependencies")
        _require_digest(self.artifact_digest)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def deterministic_combine_order(self) -> tuple[int, ...]:
        return tuple(chunk.index for chunk in self.chunks)


def build_megamoe_overlap_plan(
    *,
    plan_id: str,
    num_tokens: int,
    num_chunks: int,
    capacities: Sequence[int],
    dispatch_buffer_bytes: Sequence[int],
    pipeline_stages: int = 2,
    max_in_flight_chunks: int = 2,
    workspace_capacity_bytes: int | None = None,
) -> MegaMoEOverlapPlan:
    """Build a deterministic, capacity-bounded executable overlap plan."""

    if num_tokens < 1 or num_chunks < 1:
        raise ValueError("MegaMoE num_tokens and num_chunks must be >= 1")
    actual_chunks = min(num_chunks, num_tokens)
    if len(capacities) != actual_chunks or len(dispatch_buffer_bytes) != actual_chunks:
        raise ValueError("MegaMoE capacity vectors must match the actual chunk count")
    q, remainder = divmod(num_tokens, actual_chunks)
    chunks: list[MegaMoEChunk] = []
    start = 0
    for index in range(actual_chunks):
        stop = start + q + (1 if index < remainder else 0)
        chunks.append(
            MegaMoEChunk(
                index=index,
                start=start,
                stop=stop,
                expert_capacity=int(capacities[index]),
                dispatch_buffer_bytes=int(dispatch_buffer_bytes[index]),
            )
        )
        start = stop
    live_limit = min(max_in_flight_chunks, actual_chunks)
    if live_limit < 1:
        raise ValueError("MegaMoE max_in_flight_chunks must be >= 1")
    if actual_chunks > 1 and live_limit < 2:
        raise ValueError(
            "MegaMoE overlap requires two live chunks so dispatch(c+1) can "
            "precede combine(c)"
        )
    live_bytes = max(
        sum(
            chunk.dispatch_buffer_bytes
            for chunk in chunks[offset : offset + live_limit]
        )
        for offset in range(actual_chunks)
    )
    if workspace_capacity_bytes is not None:
        if workspace_capacity_bytes < 0:
            raise ValueError("MegaMoE workspace capacity must be non-negative")
        if live_bytes > workspace_capacity_bytes:
            raise ValueError(
                "MegaMoE overlap plan exceeds workspace capacity: "
                f"requires {live_bytes} bytes, limit is {workspace_capacity_bytes}"
            )
    dependencies = _action_dependencies(actual_chunks, live_limit)
    payload = {
        "schema": MEGAMOE_OVERLAP_PLAN_SCHEMA,
        "plan_id": plan_id,
        "num_tokens": num_tokens,
        "pipeline_stages": pipeline_stages,
        "max_in_flight_chunks": live_limit,
        "chunks": [chunk.__dict__ for chunk in chunks],
        "action_dependencies": dependencies,
        "deterministic_combine": True,
        "max_live_dispatch_bytes": live_bytes,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return MegaMoEOverlapPlan(
        plan_id=plan_id,
        num_tokens=num_tokens,
        pipeline_stages=pipeline_stages,
        max_in_flight_chunks=live_limit,
        chunks=tuple(chunks),
        action_dependencies=dependencies,
        deterministic_combine=True,
        max_live_dispatch_bytes=live_bytes,
        artifact_digest=digest,
    )


_PHASE_RANK = {"dispatch": 0, "compute": 1, "combine": 2}
_MOE_TENSOR = "tensor<*xf32>"


def megamoe_issue_order(plan: MegaMoEOverlapPlan) -> tuple[str, ...]:
    """Deterministic topological issue order of the plan's action DAG.

    Kahn traversal with a (chunk, phase) heap tie-break — the order every
    consumer shares, so the Graph representation, the inferred DAG, the
    resource-vector binding, and the parity oracle all index one sequence.
    """

    dependencies = plan.action_dependencies
    indegree = {action: len(deps) for action, deps in dependencies.items()}
    successors: dict[str, list[str]] = {}
    for action, deps in dependencies.items():
        for dep in deps:
            successors.setdefault(dep, []).append(action)

    def key(action: str) -> tuple[int, int]:
        kind, chunk = action.split(":")
        return (int(chunk), _PHASE_RANK[kind])

    heap = [(key(action), action) for action, deg in indegree.items() if deg == 0]
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        _, action = heapq.heappop(heap)
        order.append(action)
        for successor in successors.get(action, []):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(heap, (key(successor), successor))
    if len(order) != len(dependencies):
        raise ValueError("MegaMoE action DAG has a cycle")
    return tuple(order)


def megamoe_graph_function(plan: MegaMoEOverlapPlan) -> GraphIRFunction:
    """The chunk pipeline as Graph IR, in the deterministic issue order.

    dispatch/combine are the registered ordered collectives
    (`tessera.moe_dispatch` / `tessera.moe_combine`); expert compute is pure
    `tessera.matmul` consuming its own chunk's dispatch result. Edges are NOT
    written here — `infer_action_dag` derives them from these registered
    semantics (SSA flow + ordered-collective serialization), which is what
    makes the R3 candidate a consumer of W2.1/W2.2 facts (SO-3)."""

    tensor = IRType(_MOE_TENSOR)
    ops: list[IROp] = []
    for action_id in megamoe_issue_order(plan):
        kind, chunk = action_id.split(":")
        if kind == "dispatch":
            ops.append(
                IROp(
                    result=f"%dispatch{chunk}",
                    op_name="tessera.moe_dispatch",
                    operands=["%routed"],
                    operand_types=[_MOE_TENSOR],
                    result_type=_MOE_TENSOR,
                )
            )
        elif kind == "compute":
            ops.append(
                IROp(
                    result=f"%compute{chunk}",
                    op_name="tessera.matmul",
                    operands=[f"%dispatch{chunk}", "%weights"],
                    operand_types=[_MOE_TENSOR, _MOE_TENSOR],
                    result_type=_MOE_TENSOR,
                )
            )
        else:
            ops.append(
                IROp(
                    result=f"%combine{chunk}",
                    op_name="tessera.moe_combine",
                    operands=[f"%compute{chunk}"],
                    operand_types=[_MOE_TENSOR],
                    result_type=_MOE_TENSOR,
                )
            )
    return GraphIRFunction(
        name=plan.plan_id,
        args=[IRArg("routed", tensor), IRArg("weights", tensor)],
        body=ops,
        return_values=[f"%combine{plan.num_chunks - 1}"],
    )


def megamoe_inferred_composition(
    plan: MegaMoEOverlapPlan,
    benchmark_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[CompositionCandidate, InferredActionDAG, ActionDAGParity]:
    """SO-3: the production candidate's edges are INFERRED, hand edges are
    the coverage oracle.

    The plan's hand-authored `action_dependencies` stop being the executed
    DAG and become the declared oracle (#31): `compare_inferred_action_dag`
    must show the generated edges cover every hand edge, or construction
    fails closed — a generation regression can never silently weaken the
    schedule. Additional conservative edges (the ordered-collective total
    order) are reported separately on the parity result, per the SO-3
    acceptance. The returned candidate carries the content-addressed
    `ScheduleObject` (digest included) built from the inferred edges."""

    expected = set(plan.action_dependencies)
    if set(benchmark_rows) != expected:
        missing = sorted(expected - set(benchmark_rows))
        extra = sorted(set(benchmark_rows) - expected)
        raise ValueError(
            f"MegaMoE action evidence must be total; missing={missing}, extra={extra}"
        )
    order = megamoe_issue_order(plan)
    vectors = [
        TileAction.from_benchmark_row(
            action_id, benchmark_rows[action_id]
        ).resource_vector
        for action_id in order
    ]
    candidate, inferred = CompositionCandidate.from_graph(
        plan.plan_id, megamoe_graph_function(plan), vectors, action_ids=order
    )
    # Bind the PLAN's identity into the schedule identity (PR #625 review):
    # the action ids and the graph shape are functions of the chunk COUNT, so
    # two plans that differ in token ranges, expert capacities, dispatch-buffer
    # sizes, or the in-flight limit would otherwise content-address to the same
    # digest with identical benchmark rows. `plan.artifact_digest` already
    # content-addresses exactly those fields, so the schedule object carries it.
    schedule = ScheduleObject(
        f"{plan.plan_id}@{plan.artifact_digest}",
        inferred.schedule_object.actions,
        inferred.schedule_object.edges,
        inferred.schedule_object.roles,
        inferred.schedule_object.residency,
    )
    candidate = CompositionCandidate(plan.plan_id, candidate.actions, schedule)
    reference = tuple(
        TileAction(
            action_id,
            vectors[index],
            tuple(plan.action_dependencies[action_id]),
        )
        for index, action_id in enumerate(order)
    )
    parity = compare_inferred_action_dag(inferred, reference)
    if not parity.conservative:
        raise ValueError(
            "inferred MegaMoE action DAG does not cover the hand-authored "
            f"oracle edges: missing={parity.missing_reference_edges}"
        )
    return candidate, inferred, parity


def composition_candidate_for_megamoe_plan(
    plan: MegaMoEOverlapPlan,
    benchmark_rows: Mapping[str, Mapping[str, Any]],
) -> CompositionCandidate:
    """Bind measured R2 rows to every action in one executable plan.

    Since SO-3 the returned candidate's dependencies are inferred from the
    registered Graph semantics (see `megamoe_inferred_composition`); the
    hand-authored plan DAG is the fail-closed coverage oracle."""

    candidate, _, _ = megamoe_inferred_composition(plan, benchmark_rows)
    return candidate


def prune_megamoe_overlap_plans(
    plans: Sequence[MegaMoEOverlapPlan],
    benchmark_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    calibration: CompositionCalibration,
    *,
    relative_margin: float = 0.25,
    max_orders: int = EXHAUSTIVE_ORACLE_MAX_ORDERS,
) -> CompositionPruningResult:
    """R3 prune-only analysis for executable MegaMoE plans."""

    candidates = tuple(
        composition_candidate_for_megamoe_plan(plan, benchmark_rows[plan.plan_id])
        for plan in plans
    )
    return prune_composition_candidates(
        candidates,
        calibration,
        relative_margin=relative_margin,
        max_orders=max_orders,
    )


def _action_dependencies(
    num_chunks: int, max_in_flight_chunks: int
) -> dict[str, tuple[str, ...]]:
    dependencies: dict[str, tuple[str, ...]] = {}
    for chunk in range(num_chunks):
        dispatch_dependencies: list[str] = []
        if chunk == 1:
            dispatch_dependencies.append("dispatch:0")
        elif chunk >= 2:
            dispatch_dependencies.append(f"combine:{chunk - 2}")
        capacity_predecessor = chunk - max_in_flight_chunks
        if capacity_predecessor >= 0:
            predecessor = f"combine:{capacity_predecessor}"
            if predecessor not in dispatch_dependencies:
                dispatch_dependencies.append(predecessor)
        dependencies[f"dispatch:{chunk}"] = tuple(dispatch_dependencies)
        dependencies[f"compute:{chunk}"] = (f"dispatch:{chunk}",)
        combine_dependencies = [f"compute:{chunk}"]
        if chunk > 0:
            combine_dependencies.append(f"combine:{chunk - 1}")
        if chunk == 0 and num_chunks > 1:
            combine_dependencies.append("dispatch:1")
        dependencies[f"combine:{chunk}"] = tuple(combine_dependencies)
    return dependencies


def _require_digest(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("MegaMoE artifact digest must be 64 lowercase hex characters")


__all__ = [
    "MEGAMOE_OVERLAP_PLAN_SCHEMA",
    "MegaMoEChunk",
    "MegaMoEOverlapPlan",
    "build_megamoe_overlap_plan",
    "composition_candidate_for_megamoe_plan",
    "megamoe_graph_function",
    "megamoe_inferred_composition",
    "megamoe_issue_order",
    "prune_megamoe_overlap_plans",
]
