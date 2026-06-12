"""Phase E3 / Pillar 3 — Magellan-style gated heuristic evolution (EVALUATOR_PLAN.md §6)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.magellan import INF, dtype_policy_fitness, evolve, search


def _mm(a, b):
    return ts.ops.matmul(a, b)


_MM = ts.jit(target="apple_gpu")(_mm)


# ── portable: the gated optimizers ───────────────────────────────────────────

def test_search_finds_the_min_of_a_synthetic_fitness():
    best, f, hist = search([0, 1, 2, 3, 4, 5, 6], lambda x: (x - 3) ** 2)
    assert best == 3 and f == 0 and len(hist) == 7


def test_correctness_gate_changes_the_answer():
    """fitness=x (smaller better) but INF below 3 (correctness fails for small) —
    the gate must reject 0,1,2 and pick 3, not 0. This is the reward-hack guard."""
    best, f, _ = search([0, 1, 2, 3, 4, 5], lambda x: float(x) if x >= 3 else INF)
    assert best == 3 and f == 3.0


def test_search_returns_none_when_nothing_feasible():
    best, f, _ = search([0, 1, 2], lambda x: INF)
    assert best is None and f == INF


def test_evolve_hill_climbs_to_the_optimum():
    best, f, hist = evolve(0, lambda x: [x - 1, x + 1], lambda x: (x - 3) ** 2, iters=10)
    assert best == 3 and f == 0
    assert len(hist) >= 2                       # it actually moved


def test_evolve_never_takes_an_infeasible_neighbor():
    # fitness valid only at even x; start at 0, neighbors ±1 → odd (INF) rejected,
    # so it can't move and converges at the start.
    best, _, _ = evolve(0, lambda x: [x - 1, x + 1], lambda x: x * x if x % 2 == 0 else INF)
    assert best == 0


# ── Darwin: evolve a real heuristic gated behind measured correctness ─────────

@pytest.mark.skipif(sys.platform != "darwin", reason="fitness measures on Metal.")
def test_dtype_policy_search_returns_a_feasible_policy():
    rng = np.random.default_rng(20260612)
    thresholds = [0, 256, 99999]               # always-f16 … always-f32
    best, best_f, hist = search(
        thresholds, lambda t: dtype_policy_fitness(_MM, (128, 256), rng, t, reps=4)
    )
    assert best is not None, "no correctness-passing dtype policy found"
    assert best_f < INF and best in thresholds
    assert len(hist) == 3
