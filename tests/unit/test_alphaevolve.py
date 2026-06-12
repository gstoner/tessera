"""Phase E3 / Pillar 3 — AlphaEvolve-style evaluator-driven search (EVALUATOR_PLAN.md §8).

The headline test is negative: the scoring harness must reject a faster-but-
failing candidate. That is the Sakana invariant — the grader, not the candidate,
decides.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tessera.compiler.alphaevolve import (
    ScoredCandidate,
    candidate_from_task,
    gated_fitness,
    select,
)
from tessera.compiler.compiler_grader import CheckResult, Grade
from tessera.compiler.magellan import INF

_PASS = Grade("t", (CheckResult("ok", True),))
_FAIL = Grade("t", (CheckResult("wrong", False, "miscompile"),))


# ── portable: the scoring harness gates the search ───────────────────────────

def test_gated_fitness_rejects_a_failing_grade():
    assert gated_fitness(_PASS, 1.0) == 1.0
    assert gated_fitness(_FAIL, 0.1) == INF        # fast but wrong → rejected


def test_select_rejects_the_reward_hack_even_though_it_is_fastest():
    cands = [
        ScoredCandidate("correct_slow", _PASS, 3.0),
        ScoredCandidate("correct_fast", _PASS, 1.0),
        ScoredCandidate("cheat_fast", _FAIL, 0.1),   # best perf, but FAILS the grader
    ]
    winner = select(cands)
    assert winner is not None and winner.label == "correct_fast"
    assert winner.label != "cheat_fast"             # the Sakana invariant holds


def test_select_none_when_nothing_passes():
    assert select([ScoredCandidate("a", _FAIL, 0.1)]) is None


# ── Darwin: drive the search with the real grader ────────────────────────────

@pytest.mark.skipif(sys.platform != "darwin", reason="real grader tasks execute on Metal.")
def test_search_over_real_graded_tasks_picks_a_passing_candidate():
    rng = np.random.default_rng(20260612)
    cands = [
        candidate_from_task("matmul/apple_gpu/rung7", rng, perf_ms=2.0),
        candidate_from_task("matmul/cross_path/metal_vs_accelerate", rng, perf_ms=1.0),
    ]
    assert all(c.grade.passed for c in cands)        # real grader corroborates
    winner = select(cands)
    assert winner is not None and winner.grade.passed
    assert winner.perf_ms == 1.0                     # fastest passing candidate
