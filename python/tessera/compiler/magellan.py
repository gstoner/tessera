"""Phase E3 / Pillar 3 — Magellan-style heuristic evolution
(docs/audit/compiler/EVALUATOR_PLAN.md §6; Magellan, C4ML@CGO'26).

Magellan evolves compiler heuristics with macro-benchmark feedback: a heuristic
template with symbolic hyperparameters, an autotuner fills them, fitness comes
from a benchmark. The load-bearing design — and the reason it doesn't reward-hack
(the Sakana lesson) — is that **fitness is perf gated behind correctness**: a
candidate that is faster-but-wrong scores ``INF`` and is rejected.

This builds that loop on Tessera's substrate: ``search`` / ``evolve`` are the
gated optimizers (fitness ``INF`` ⇒ candidate rejected), and ``dtype_policy_fitness``
is a concrete heuristic — "use f16 when the problem is at least ``threshold``,
else f32" — whose fitness is **measured latency** (flywheel) **gated behind an
f16-vs-f32 correctness check**. Now that deterministic records + the grader exist
(the gate you named), this lane is unblocked.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

INF = float("inf")


def search(
    candidates: list[Any], fitness_fn: Callable[[Any], float]
) -> tuple[Any, float, list[tuple[Any, float]]]:
    """Gated grid search: ``fitness_fn`` returns ``INF`` for a correctness-failing
    candidate (rejected); the lowest finite fitness wins. Returns
    ``(best, best_fitness, history)`` — ``(None, INF, history)`` if none feasible."""
    history = [(c, fitness_fn(c)) for c in candidates]
    feasible = [(c, f) for c, f in history if f < INF]
    if not feasible:
        return None, INF, history
    best_c, best_f = min(feasible, key=lambda cf: cf[1])
    return best_c, best_f, history


def evolve(
    start: Any,
    mutate: Callable[[Any], list[Any]],
    fitness_fn: Callable[[Any], float],
    *,
    iters: int = 8,
) -> tuple[Any, float, list[tuple[Any, float]]]:
    """Hill-climb with gated fitness (the evolutionary loop): from ``start``,
    move to the best feasible neighbor each step until no improvement. Same
    correctness gate as :func:`search` — an infeasible neighbor is never taken."""
    cur, cur_f = start, fitness_fn(start)
    history: list[tuple[Any, float]] = [(cur, cur_f)]
    for _ in range(iters):
        scored = [(c, fitness_fn(c)) for c in mutate(cur)]
        feasible = [(c, f) for c, f in scored if f < INF]
        if not feasible:
            break
        best_c, best_f = min(feasible, key=lambda cf: cf[1])
        history.append((best_c, best_f))
        if not (best_f < cur_f):           # converged (no improvement)
            break
        cur, cur_f = best_c, best_f
    return cur, cur_f, history


def dtype_policy_fitness(
    fn: Any,
    sizes: tuple[int, ...],
    rng: Any,
    threshold: int,
    *,
    target: str = "apple_gpu",
    rtol: float = 2e-2,
    reps: int = 6,
) -> float:
    """Fitness of the policy "use f16 when size ≥ ``threshold``, else f32" over a
    workload set: **total measured latency**, or ``INF`` if any f16 choice
    diverges from the f32 reference (the correctness gate that keeps the search
    honest). Lower is better."""
    import numpy as np

    from tessera.compiler.evaluator import run_native
    from tessera.compiler.flywheel import measure_latency

    total = 0.0
    for s in sizes:
        use_f16 = s >= threshold
        np_dtype = np.float16 if use_f16 else np.float32
        a = rng.standard_normal((s, s)).astype(np_dtype)
        b = rng.standard_normal((s, s)).astype(np_dtype)

        if use_f16:                         # correctness gate: f16 ≈ f32 reference
            out, native = run_native(target, fn, (a, b))
            if not native:
                return INF
            ref = a.astype(np.float32) @ b.astype(np.float32)
            if not np.allclose(np.asarray(out, dtype=np.float32), ref, rtol=rtol, atol=1e-2):
                return INF

        stats = measure_latency(target, fn, (a, b), reps=reps)
        if stats is None:
            return INF
        total += stats.median_ms
    return total
