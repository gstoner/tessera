from __future__ import annotations

import threading

import pytest
import numpy as np

from tessera import runtime
from tessera import collectives
from tessera.compiler.pipeline_runtime import OptimizerShardTransport, parse_pipeline_steps


STEPS = [
    {"clock": 0, "micro_batch": 0, "phase": "forward", "region": "warmup", "stage": 0},
    {"clock": 1, "micro_batch": 0, "phase": "forward", "region": "warmup", "stage": 1},
    {"clock": 2, "micro_batch": 0, "phase": "backward", "region": "steady", "stage": 1},
    {"clock": 3, "micro_batch": 1, "phase": "forward", "region": "steady", "stage": 0},
    {"clock": 4, "micro_batch": 0, "phase": "backward", "region": "cooldown", "stage": 0},
]


def test_runtime_consumes_compiler_order_and_overlaps_transport() -> None:
    compute_order: list[int] = []
    collective_started = threading.Event()
    release_collective = threading.Event()
    overlap_observed = threading.Event()

    def run_stage(step) -> None:
        compute_order.append(step.clock)
        if step.clock == 3:
            assert collective_started.wait(timeout=2.0)
            overlap_observed.set()
            release_collective.set()

    def run_collective(step) -> None:
        if step.clock == 2:
            collective_started.set()
            assert release_collective.wait(timeout=2.0)

    result = runtime.execute_emitted_pipeline(
        {"tessera.pipeline_steps": STEPS},
        run_stage=run_stage,
        run_collective=run_collective,
        collective_after=lambda step: step.clock == 2,
    )
    assert compute_order == [0, 1, 2, 3, 4]
    assert overlap_observed.is_set()
    assert result.collective_count == 1
    assert result.peak_inflight_collectives == 1


def test_runtime_rejects_schedule_clock_drift() -> None:
    rows = [dict(row) for row in STEPS]
    rows[2]["clock"] = 7
    with pytest.raises(ValueError, match="unique, ordered, and contiguous"):
        parse_pipeline_steps(rows)


def test_optimizer_shard_descriptors_drive_transport_and_ownership() -> None:
    rows = [dict(row) for row in STEPS]
    rows[2]["collectives"] = [{
        "kind": "reduce_scatter",
        "tensor": "weight.grad",
        "optimizer_shard": True,
        "normalize": True,
    }]
    rows[3]["collectives"] = [{
        "kind": "all_gather",
        "tensor": "weight.grad",
        "optimizer_shard": True,
    }]
    transport = OptimizerShardTransport(
        collectives.adapter(backend="mock", world_size=2, mesh_axes={"dp": 2}),
        {
            "weight.grad": [
                np.arange(4, dtype=np.float32),
                np.arange(4, dtype=np.float32) + 4,
            ],
        },
    )
    result = runtime.execute_emitted_pipeline(
        {"tessera.pipeline_steps": rows},
        run_stage=lambda _step: None,
        collective_runtime=transport,
        transport_workers=2,
    )
    assert result.collective_count == 2
    assert transport.ownership["weight.grad"] == "replicated"
    expected = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    for value in transport.values("weight.grad"):
        np.testing.assert_allclose(value, expected)


def test_optimizer_shard_rejects_invalid_ownership_transition() -> None:
    rows = [dict(STEPS[0])]
    rows[0]["collectives"] = [{
        "kind": "all_gather",
        "tensor": "weight",
        "optimizer_shard": True,
    }]
    transport = OptimizerShardTransport(
        collectives.adapter(backend="mock", world_size=2, mesh_axes={"dp": 2}),
        {"weight": [np.ones(2, np.float32), np.ones(2, np.float32)]},
    )
    with pytest.raises(RuntimeError, match="rank-local"):
        runtime.execute_emitted_pipeline(
            {"tessera.pipeline_steps": rows},
            run_stage=lambda _step: None,
            collective_runtime=transport,
        )
