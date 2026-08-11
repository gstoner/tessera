from __future__ import annotations

import hashlib

import pytest

from tessera.compiler.benchmark_row import MeasuredResourceVector
from tessera.compiler.composition_cost import CompositionCalibration
from tessera.compiler.megamoe_overlap import (
    build_megamoe_overlap_plan,
    composition_candidate_for_megamoe_plan,
    prune_megamoe_overlap_plans,
)


def _plan(plan_id: str, *, chunks: int = 3, workspace: int | None = None):
    return build_megamoe_overlap_plan(
        plan_id=plan_id,
        num_tokens=12,
        num_chunks=chunks,
        capacities=[4] * chunks,
        dispatch_buffer_bytes=[1024] * chunks,
        pipeline_stages=2,
        max_in_flight_chunks=2,
        workspace_capacity_bytes=workspace,
    )


def _row(action_id: str, *, compute_ms: float) -> dict:
    digest = hashlib.sha256(action_id.encode("utf-8")).hexdigest()
    kind = action_id.split(":", 1)[0]
    communication_bytes = 1000 if kind in {"dispatch", "combine"} else 0
    vector = MeasuredResourceVector(
        compute_time_ms=compute_ms,
        bytes_moved=512,
        communication_bytes=communication_bytes,
        queue_identity="comm:0" if communication_bytes else "compute:0",
        resource_identity="device:0",
        timing_provenance={"source": "exact_device", "domain": "device"},
        artifact_digest=digest,
    ).as_dict()
    return {"hot_path_metadata": {"resource_vector": vector}}


def _rows(plan, *, compute_ms: float) -> dict:
    return {
        action_id: _row(action_id, compute_ms=compute_ms)
        for action_id in plan.action_dependencies
    }


def _calibration() -> CompositionCalibration:
    return CompositionCalibration(
        memory_bytes_per_ms=1_000_000.0,
        communication_bytes_per_ms=1_000.0,
        provenance={"source": "exact_device", "domain": "device"},
        digest="c" * 64,
    )


def test_chunk_plan_is_content_addressed_capacity_bounded_and_deterministic():
    plan = _plan("three_chunks", workspace=2048)

    assert [(chunk.start, chunk.stop) for chunk in plan.chunks] == [
        (0, 4), (4, 8), (8, 12)
    ]
    assert plan.max_live_dispatch_bytes == 2048
    assert plan.deterministic_combine_order == (0, 1, 2)
    assert len(plan.artifact_digest) == 64
    assert plan.action_dependencies["compute:1"] == ("dispatch:1",)
    assert "combine:0" in plan.action_dependencies["dispatch:2"]
    assert "combine:0" in plan.action_dependencies["combine:1"]


def test_chunk_plan_rejects_workspace_overflow_and_one_frame_overlap():
    with pytest.raises(ValueError, match="workspace capacity"):
        _plan("too_large", workspace=2047)
    with pytest.raises(ValueError, match="two live chunks"):
        build_megamoe_overlap_plan(
            plan_id="one_frame",
            num_tokens=8,
            num_chunks=2,
            capacities=[4, 4],
            dispatch_buffer_bytes=[1024, 1024],
            max_in_flight_chunks=1,
        )


def test_r3_evidence_must_cover_every_executable_action():
    plan = _plan("totality", chunks=2)
    rows = _rows(plan, compute_ms=1.0)
    rows.pop("combine:1")
    with pytest.raises(ValueError, match="must be total"):
        composition_candidate_for_megamoe_plan(plan, rows)


def test_r3_prunes_plans_but_cannot_promote_one():
    fast = _plan("fast", chunks=2)
    slow = _plan("slow", chunks=2)
    result = prune_megamoe_overlap_plans(
        (slow, fast),
        {
            "fast": _rows(fast, compute_ms=1.0),
            "slow": _rows(slow, compute_ms=10.0),
        },
        _calibration(),
        relative_margin=0.25,
    )

    assert result.retained == ("fast",)
    assert result.pruned == ("slow",)
    assert not result.eligible_for_promotion
    assert result.selector_authority == "latency_ms"
    assert not hasattr(result, "winner")
