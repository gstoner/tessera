from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.residual_evaluator import (
    ForwardCapture,
    ResidualEvidence,
    capture_treeverse_forward,
    execute_treeverse_region_adjoint,
    measure_counted_region_policy_cohort,
    measure_treeverse_region_candidate,
    measure_residual_candidate,
    retained_residual_bytes,
    select_residual_policy,
    treeverse_candidates,
)


def _evidence(policy: str, work: int, residual_bytes: int, *, exact: bool = True) -> ResidualEvidence:
    return ResidualEvidence(
        target="x86",
        operation="tessera.matmul",
        shape_bucket=(64, 64, 64),
        dtype="f32",
        policy=policy,  # type: ignore[arg-type]
        retained_residual_bytes=residual_bytes,
        backward_samples_ns=(work - 1, work, work + 1),
        timing_domain="synchronized_host_wall",
        provenance="zen5_exact_device",
        exact_device=exact,
    )


def test_retained_bytes_counts_shared_views_once() -> None:
    allocation = np.empty((32,), dtype=np.float32)
    residuals = {"left": allocation[:16], "right": allocation[16:]}
    assert retained_residual_bytes(residuals) == allocation.nbytes


def test_measurement_covers_complete_backward_and_stamps_attributes() -> None:
    x = np.arange(8, dtype=np.float32)
    calls = 0

    def forward() -> ForwardCapture:
        return ForwardCapture(output=x * x, residuals=(x.copy(),))

    def backward(cotangent, residuals):
        nonlocal calls
        calls += 1
        return 2.0 * residuals[0] * cotangent

    row = measure_residual_candidate(
        target="x86",
        operation="tessera.square",
        shape_bucket=x.shape,
        dtype="f32",
        policy="save",
        forward=forward,
        backward=backward,
        cotangent=np.ones_like(x),
        warmup=1,
        repetitions=3,
        provenance="zen5_exact_device",
        exact_device=True,
    )
    assert calls == 4
    assert row.retained_residual_bytes == x.nbytes
    assert len(row.backward_samples_ns) == 3
    attrs = row.compiler_attributes()
    assert attrs["tessera.residual.retained_bytes"] == x.nbytes
    assert attrs["tessera.backward_work_ns"] == row.backward_median_ns


def test_selector_uses_measured_work_subject_to_memory_budget() -> None:
    save = _evidence("save", 10, 4096)
    recompute = _evidence("recompute", 30, 0)
    assert select_residual_policy((save, recompute), memory_budget_bytes=8192).policy == "save"
    assert select_residual_policy((save, recompute), memory_budget_bytes=0).policy == "recompute"


def test_model_or_non_device_rows_cannot_select_or_stamp() -> None:
    modeled = _evidence("treeverse", 5, 0, exact=False)
    with pytest.raises(ValueError, match="execution-derived"):
        modeled.compiler_attributes()
    with pytest.raises(ValueError, match="no execution-derived"):
        select_residual_policy((modeled,), memory_budget_bytes=0)


def test_treeverse_is_candidate_pruning_not_a_promotion_verdict() -> None:
    no_checkpoints, two_checkpoints, save_all = treeverse_candidates(
        steps=8,
        state_bytes=1024,
        measured_step_work_ns=50,
        memory_budgets=(0, 2048, 8192),
    )
    assert no_checkpoints.checkpoint_indices == ()
    assert two_checkpoints.checkpoint_slots == 2
    assert save_all.checkpoint_slots == 7
    assert (
        no_checkpoints.estimated_backward_work_ns
        > two_checkpoints.estimated_backward_work_ns
        > save_all.estimated_backward_work_ns
    )
    assert not any(row.selector_eligible for row in (no_checkpoints, two_checkpoints, save_all))


def test_treeverse_region_adjoint_executes_exact_candidate_replay() -> None:
    (candidate,) = treeverse_candidates(
        steps=6,
        state_bytes=16,
        measured_step_work_ns=10,
        memory_budgets=(32,),
    )
    initial = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    def forward_step(index, state):
        return state * np.float32(1.0 + 0.1 * (index + 1))

    def backward_step(index, before, after, cotangent):
        del before, after
        return cotangent * np.float32(1.0 + 0.1 * (index + 1))

    capture = capture_treeverse_forward(candidate=candidate, initial_state=initial, forward_step=forward_step)
    assert retained_residual_bytes(capture.residuals) == candidate.retained_residual_bytes
    execution = execute_treeverse_region_adjoint(
        cotangent=np.ones_like(initial),
        residuals=capture.residuals,
        forward_step=forward_step,
        backward_step=backward_step,
    )
    expected_scale = np.prod(np.array([1.0 + 0.1 * (i + 1) for i in range(candidate.steps)]))
    np.testing.assert_allclose(execution.input_cotangent, expected_scale)
    assert execution.replayed_steps == candidate.estimated_replayed_steps
    assert execution.backward_steps == candidate.steps


def test_executed_treeverse_can_select_only_with_exact_device_provenance() -> None:
    (candidate,) = treeverse_candidates(
        steps=4,
        state_bytes=8,
        measured_step_work_ns=10,
        memory_budgets=(8,),
    )
    initial = np.ones((2,), dtype=np.float32)

    row = measure_treeverse_region_candidate(
        target="x86",
        operation="test.counted_region",
        shape_bucket=initial.shape,
        dtype="f32",
        candidate=candidate,
        initial_state=initial,
        forward_step=lambda _index, state: state * np.float32(1.25),
        backward_step=lambda _index, _before, _after, ct: ct * np.float32(1.25),
        cotangent=np.ones_like(initial),
        warmup=0,
        repetitions=2,
        provenance="zen5_exact_device",
        exact_device=True,
    )
    assert row.policy == "hybrid"
    assert row.retained_residual_bytes == candidate.retained_residual_bytes
    assert row.forward_steps == candidate.steps + candidate.estimated_replayed_steps
    assert row.replayed_steps == candidate.estimated_replayed_steps
    assert row.backward_steps == candidate.steps
    assert row.evidence_record()["policy"] == "hybrid"
    assert row.selector_eligible


def test_counted_region_executes_save_recompute_hybrid_cohort() -> None:
    initial = np.array([0.25, 0.5], dtype=np.float32)

    def forward_step(index, state):
        return state * np.float32(1.1 + 0.05 * index)

    def backward_step(index, before, after, cotangent):
        del before, after
        return cotangent * np.float32(1.1 + 0.05 * index)

    rows = measure_counted_region_policy_cohort(
        target="x86",
        operation="test.affine_counted_region",
        shape_bucket=initial.shape,
        dtype="f32",
        steps=5,
        state_bytes=initial.nbytes,
        measured_step_work_ns=10,
        memory_budgets=(0, initial.nbytes * 2, initial.nbytes * 5),
        initial_state=initial,
        forward_step=forward_step,
        backward_step=backward_step,
        cotangent=np.ones_like(initial),
        warmup=0,
        repetitions=1,
        provenance="host_functional_packet",
        exact_device=False,
    )

    assert [row.policy for row in rows] == ["recompute", "hybrid", "save"]
    assert rows[0].replayed_steps > rows[1].replayed_steps > rows[2].replayed_steps
    assert rows[0].retained_residual_bytes < rows[1].retained_residual_bytes < rows[2].retained_residual_bytes
    assert all(row.backward_steps == 5 for row in rows)
    assert not any(row.selector_eligible for row in rows)
