"""Lattice reasoning benchmark oracle tests."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmarks.lattice_reasoning_core import (  # noqa: E402
    LatticeReasoningBenchmark,
    LatticeReasoningConfig,
    candidate_counts,
    gqa_decode_core,
    lattice_alpha,
    ldt_step,
    mamba2_ssd_core,
    masked_softmax,
    mopd_policy_loss_core,
    threshold_eliminate,
)


def _one_hot_solution(digits: np.ndarray, V: int) -> np.ndarray:
    return np.eye(V, dtype=bool)[digits]


def test_lattice_alpha_or_keeps_multi_solution_candidates() -> None:
    state = np.ones((1, 2, 2, 3), dtype=bool)
    s0 = _one_hot_solution(np.array([[[0, 1], [2, 0]]]), 3)
    s1 = _one_hot_solution(np.array([[[1, 1], [2, 2]]]), 3)
    alpha = lattice_alpha(np.stack([s0, s1]), state)
    assert alpha[0, 0, 0, 0]
    assert alpha[0, 0, 0, 1]
    assert not alpha[0, 0, 0, 2]
    assert alpha[0, 1, 0, 2]


def test_threshold_elimination_preserves_high_confidence_target_candidates() -> None:
    state = np.ones((1, 2, 2, 3), dtype=bool)
    target_digits = np.array([[[0, 1], [2, 0]]])
    target = _one_hot_solution(target_digits, 3)
    logits = np.where(target, 3.0, -3.0).astype(np.float32)
    eliminated = threshold_eliminate(state, logits, theta_elim=0.1)
    assert np.all(eliminated[target])
    assert not np.any(eliminated[~target])


def test_empty_cell_and_cls_paths_mark_conflict() -> None:
    state = np.ones((1, 2, 2, 3), dtype=bool)
    logits = np.full(state.shape, 3.0, dtype=np.float32)
    state[0, 0, 0, :] = False
    empty = ldt_step(state, logits, -4.0)
    assert empty.conflict is True

    cls = ldt_step(np.ones((1, 2, 2, 3), dtype=bool), logits, 4.0)
    assert cls.conflict is True


def test_singleton_lattice_marks_solved_without_branch() -> None:
    state = _one_hot_solution(np.array([[[0, 1], [2, 0]]]), 3)
    logits = np.where(state, 3.0, -3.0).astype(np.float32)
    step = ldt_step(state, logits, -4.0)
    assert step.solved is True
    assert step.conflict is False
    assert step.branch_cell is None
    assert step.pinned_candidate is None


def test_branch_pin_selects_only_alive_candidates() -> None:
    state = np.zeros((1, 2, 2, 4), dtype=bool)
    state[..., 1:3] = True
    logits = np.full(state.shape, -4.0, dtype=np.float32)
    logits[..., 1:3] = 2.0
    step = ldt_step(state, logits, -4.0, rng=np.random.default_rng(7))
    assert step.branch_cell is not None
    assert step.pinned_candidate in {1, 2}
    assert candidate_counts(step.lattice)[step.branch_cell] == 1


def test_masked_softmax_zeroes_dead_candidates() -> None:
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    mask = np.array([[True, False, True]])
    probs = masked_softmax(logits, mask)
    assert probs[0, 1] == 0.0
    np.testing.assert_allclose(np.sum(probs, axis=-1), np.array([1.0]), rtol=1e-6)


def test_model_primitive_reference_kernels_are_finite() -> None:
    rng = np.random.default_rng(0)
    student = rng.normal(size=(1, 3, 4)).astype(np.float32)
    teachers = rng.normal(size=(2, 1, 3, 4)).astype(np.float32)
    loss = mopd_policy_loss_core(student, teachers, np.array([0.7, 0.3], dtype=np.float32))
    assert np.isfinite(loss)

    x = rng.normal(size=(1, 4, 4)).astype(np.float32)
    y = mamba2_ssd_core(x, -0.1 * np.ones_like(x), np.ones_like(x), np.ones_like(x), np.zeros_like(x))
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))

    q = rng.normal(size=(1, 4, 4)).astype(np.float32)
    k = rng.normal(size=(1, 5, 2, 4)).astype(np.float32)
    v = rng.normal(size=(1, 5, 2, 4)).astype(np.float32)
    gqa = gqa_decode_core(q, k, v)
    assert gqa.shape == q.shape
    assert np.all(np.isfinite(gqa))


def test_lattice_reasoning_benchmark_reference_row_runs() -> None:
    cfg = LatticeReasoningConfig(B=1, H=4, W=4, V=4, K=3, seed=0)
    rows = LatticeReasoningBenchmark(warmup=0, reps=1).rows(cfg)
    names = {row.operator.name for row in rows}
    assert "lattice_reasoning_step" in names
    assert "mopd_policy_loss_core" in names
    assert "mamba2_ssd_core" in names
    assert "gqa_decode_core" in names
    assert "latent_moe_core" in names
    for row in rows:
        assert row.correctness.passed is not False
