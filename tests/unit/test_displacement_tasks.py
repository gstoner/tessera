"""M5 follow-on — the displaced synthesizer lanes graded in the scored env.

Importing tessera.compiler.displacement_tasks registers one grader task per
shipped fusion lane; each is graded on hidden inputs via the displacement
verdict. A lane that ever diverges from its reference fails the grade.
"""

from __future__ import annotations

import numpy as np

from tessera.compiler import compiler_grader as G
from tessera.compiler import displacement_tasks as DT  # noqa: F401 (registers tasks)


def test_lanes_registered_as_tasks():
    names = set(G.task_names())
    for kind in ("matmul_epilogue", "norm_chain", "attention", "pointwise"):
        assert f"displacement/{kind}" in names


def test_grade_all_no_divergence():
    grades = G.grade_all(np.random.default_rng(2024))
    for g in grades:
        for c in g.checks:
            if c.name.endswith("not_divergent"):
                assert c.passed, f"{g.task}: {c.detail}"


def test_displacement_lanes_pass_or_only_provenance_skips():
    # On a Metal host every displacement task should fully pass. Off Metal the
    # only allowed failure is the "executes_on_metal" provenance check — never a
    # divergence or a wrong-on-metal check.
    for kind in ("matmul_epilogue", "norm_chain", "attention", "pointwise"):
        g = G.grade(f"displacement/{kind}", np.random.default_rng(7))
        for c in g.failures:
            assert c.name.endswith("executes_on_metal"), \
                f"unexpected failure {c.name}: {c.detail}"


def test_no_vacuous_pass():
    # Each task emits checks (a zero-check task fails by construction).
    for kind in ("matmul_epilogue", "norm_chain", "attention", "pointwise"):
        g = G.grade(f"displacement/{kind}", np.random.default_rng(1))
        assert len(g.checks) > 0
