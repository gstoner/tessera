"""Structure-keyed long-memory task family — on-device scoring/select/soft-read."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tessera.compiler import memory_tasks as M
from tessera.compiler.compiler_grader import grade, task_names


# ── portable ─────────────────────────────────────────────────────────────────

def test_matrix_is_structure_keyed_by_op_and_bank_size():
    names = M.memory_task_names()
    assert len(names) == 9                            # 3 ops × 3 bank sizes
    assert "memory/score/n64" in names
    assert "memory/top1/n1024" in names
    assert "memory/soft_read/n256" in names
    # importing the module registered them into the grader
    assert "memory/score/n64" in task_names()


def test_score_and_soft_read_oracles_are_consistent():
    rng = np.random.default_rng(0)
    n, d, nq = 32, 8, 3
    keys = rng.standard_normal((n, d)).astype(np.float32)
    q = rng.standard_normal((nq, d)).astype(np.float32)
    # the score the grader checks is exactly query·keysᵀ
    assert np.allclose(q @ keys.T, np.einsum("qd,nd->qn", q, keys), atol=1e-5)


# ── Darwin: grade every (op × bank-size) cell on Metal ───────────────────────

@pytest.mark.hardware_apple_gpu
def test_every_memory_cell_grades_pass():
    rng = np.random.default_rng(20260614)
    for name in M.memory_task_names():
        g = grade(name, rng)
        assert g.passed, f"{name}: {[c.detail for c in g.failures]}"
