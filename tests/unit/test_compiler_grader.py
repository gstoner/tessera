"""Phase E3 / Pillar 3 — the internal TensorBench-style grader (EVALUATOR_PLAN.md §8).

Portable: the grade contract (no vacuous pass, failure surfacing, task registry,
duplicate guard). Darwin: every seed task passes on hidden inputs — and each
grade carries a real check, never an empty/vacuous pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.compiler_grader import (
    CheckResult,
    Grade,
    grade_all,
    task,
    task_names,
)


# ── portable: the grade contract ─────────────────────────────────────────────

def test_empty_grade_is_a_failure_no_vacuous_pass():
    assert Grade("t", ()).passed is False


def test_all_passing_checks_pass_and_a_failure_fails():
    ok = Grade("t", (CheckResult("a", True), CheckResult("b", True)))
    assert ok.passed and ok.failures == ()
    bad = Grade("t", (CheckResult("a", True), CheckResult("b", False, "nope")))
    assert not bad.passed and len(bad.failures) == 1


def test_seed_task_set_is_registered():
    names = task_names()
    assert "matmul/cross_path/metal_vs_accelerate" in names
    assert "matmul_gelu/fusion/horizontal" in names
    assert len(names) >= 4


def test_duplicate_task_name_is_rejected():
    with pytest.raises(ValueError, match="duplicate task"):
        task("matmul/apple_gpu/rung7")(lambda rng: [])


# ── Darwin: every seed task passes on hidden inputs ──────────────────────────

@pytest.mark.hardware_apple_gpu
def test_all_seed_tasks_pass_on_hidden_inputs():
    grades = grade_all(np.random.default_rng(20260612))
    assert len(grades) >= 4
    for g in grades:
        assert g.checks, f"{g.task}: vacuous task (no checks)"
        assert g.passed, f"{g.task} failed: {[c.detail for c in g.failures]}"


@pytest.mark.hardware_apple_gpu
def test_grades_are_stable_across_independent_hidden_draws():
    """Re-grading on a different RNG must still pass — the checks are real
    invariants, not fit to one input set."""
    for seed in (1, 2, 3):
        for g in grade_all(np.random.default_rng(seed)):
            assert g.passed, f"{g.task} (seed={seed}): {[c.detail for c in g.failures]}"
