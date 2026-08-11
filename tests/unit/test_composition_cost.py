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
    CompositionCalibration,
    CompositionCandidate,
    TileAction,
    estimate_composition,
    prune_composition_candidates,
)


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
        _action(f"a{i}", compute_ms=20.0, queue=f"compute:{i}")
        for i in range(4)
    ))
    exact = CompositionCandidate("exact", (_action("e", compute_ms=1.0),))

    result = prune_composition_candidates(
        (wide, exact), _calibration(), relative_margin=0.0, max_orders=2
    )

    wide_estimate = next(e for e in result.estimates if e.candidate_id == "wide")
    assert not wide_estimate.exhaustive
    assert "wide" in result.retained


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


@pytest.mark.parametrize("margin", [-0.1, float("nan"), float("inf")])
def test_pruning_rejects_invalid_margin(margin: float):
    candidate = CompositionCandidate("only", (_action("a", compute_ms=1.0),))
    with pytest.raises(ValueError, match="relative_margin"):
        prune_composition_candidates(
            (candidate,), _calibration(), relative_margin=margin
        )
