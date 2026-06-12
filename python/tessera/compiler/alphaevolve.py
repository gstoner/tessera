"""Phase E3 / Pillar 3 — AlphaEvolve-style evaluator-driven search
(docs/audit/compiler/EVALUATOR_PLAN.md §6, §8; AlphaEvolve, 2025).

AlphaEvolve evolves candidate programs scored by an automated fitness function.
Its (and FunSearch's) defining property — *the scoring function IS the system* —
is the thesis this whole effort was built around: the agent is a commodity, the
**scoring harness** is the product. Here the harness is the
:mod:`compiler_grader` (correctness oracles + hidden inputs + anti-cheat), and a
candidate's fitness is **its perf gated behind passing the grader**.

The load-bearing demonstration is negative: a candidate that is *faster but
wrong*, or that reward-hacks via a silent fallback, scores ``INF`` and **cannot
win**, no matter how good its perf looks — because the grader, not the candidate,
decides. That is exactly the gap Sakana's CUDA-engineer hit (a sandbox exploit
that skipped the correctness check); leading with an honest grader closes it by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from tessera.compiler.compiler_grader import Grade, grade
from tessera.compiler.magellan import INF, search


def gated_fitness(g: Grade, perf_ms: float) -> float:
    """Perf gated behind the full grader: a failing grade ⇒ ``INF`` (rejected),
    regardless of perf. The AlphaEvolve/Sakana invariant in one line."""
    return perf_ms if g.passed else INF


@dataclass(frozen=True)
class ScoredCandidate:
    label: str
    grade: Grade            # correctness on hidden inputs + anti-cheat
    perf_ms: float          # lower is better


def select(candidates: list[ScoredCandidate]) -> ScoredCandidate | None:
    """Evaluator-driven selection: among candidates that **pass the grader**, the
    best perf. A faster-but-failing (or reward-hacking) candidate is rejected.
    ``None`` if none pass."""
    best, _f, _hist = search(candidates, lambda c: gated_fitness(c.grade, c.perf_ms))
    return best


def candidate_from_task(name: str, rng: object, *, perf_ms: float) -> ScoredCandidate:
    """Build a scored candidate by grading a real compiler task on hidden inputs
    (the passed-in RNG) and attaching a perf number."""
    return ScoredCandidate(name, grade(name, rng), perf_ms)
