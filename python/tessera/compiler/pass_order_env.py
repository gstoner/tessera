"""Phase-ordering as a gated action space (CompilerGym lens).

CompilerGym exposes compiler tasks as RL environments — phase ordering being the
canonical one: passes are actions, reward is improvement over a reference
pipeline (``-Oz``).  Tessera already has the *agents* (``magellan`` /
``alphaevolve`` — gated heuristic search, perf-behind-correctness) and the
*environment authority* (the evaluator).  What was missing is the **env/agent
split for pass ordering**: an action space of pass sequences with a gated reward.

This module is that environment.  An action is an ordering of the lowering-
pipeline passes.  The reward is gated by **correctness = dependency validity**
(an ordering that violates a pass's prerequisites is rejected with ``INF`` — the
Sakana invariant), and among valid orderings the cost is **fusion effectiveness**:
fusion passes only eliminate ops when they run *before* lowering freezes the
graph, so ordering matters.  ``magellan.search`` / ``magellan.evolve`` drive it
unchanged — the env exposes ``fitness``; the agent optimizes it.

The pass set + dependency constraints mirror the real ``tessera-lower-to-
apple_gpu-runtime`` / NVIDIA pipelines (EffectAnnotation → Canonicalize → fusion
passes → TileIRLowering → WarpSpec; CollectiveInsertion after EffectAnnotation —
see CLAUDE.md "Collective Insertion Order").  The cost model is a deliberately
simple, honest proxy for fuse-before-lower, not a cycle-accurate simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from tessera.compiler.magellan import INF, evolve, search


@dataclass(frozen=True)
class PassSpec:
    """One pipeline pass: its kind, how many ops it fuses away (if it runs before
    lowering), and the passes that must precede it."""

    name: str
    kind: str                                  # annotate | canonicalize | fuse | lower | collective
    fuses: int = 0                             # ops eliminated when applied pre-lowering
    must_follow: tuple[str, ...] = ()


# The real apple_gpu/NVIDIA pipeline spine + dependency constraints.
PIPELINE_PASSES: tuple[PassSpec, ...] = (
    PassSpec("effect_annotation", "annotate"),
    PassSpec("canonicalize", "canonicalize", must_follow=("effect_annotation",)),
    PassSpec("swiglu_fusion", "fuse", fuses=2, must_follow=("canonicalize",)),
    PassSpec("mla_fusion", "fuse", fuses=2, must_follow=("canonicalize",)),
    PassSpec("nsa_fusion", "fuse", fuses=1, must_follow=("canonicalize",)),
    # CollectiveInsertion reads effect annotations (CLAUDE.md): must follow them.
    PassSpec("collective_insertion", "collective", must_follow=("effect_annotation",)),
    PassSpec("tile_lowering", "lower", must_follow=("canonicalize",)),
    PassSpec("warp_spec", "lower", must_follow=("tile_lowering",)),
)


@dataclass(frozen=True)
class PassOrderEnv:
    """A gated environment over orderings of ``passes``."""

    passes: tuple[PassSpec, ...] = PIPELINE_PASSES
    base_ops: int = 24                         # ops in the model graph before fusion
    _by_name: dict[str, PassSpec] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_name", {p.name: p for p in self.passes})

    # ── action space ─────────────────────────────────────────────────────────
    def pass_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.passes)

    def default_ordering(self) -> tuple[str, ...]:
        """The canonical pipeline order — the reference (``-Oz``-equivalent)."""
        return self.pass_names()

    # ── correctness gate: dependency validity ────────────────────────────────
    def is_valid(self, ordering: tuple[str, ...]) -> bool:
        if set(ordering) != set(self.pass_names()) or len(ordering) != len(self.passes):
            return False                       # must be a permutation of all passes
        pos = {name: i for i, name in enumerate(ordering)}
        for p in self.passes:
            for dep in p.must_follow:
                if pos[dep] > pos[p.name]:      # prerequisite runs too late
                    return False
        return True

    def violations(self, ordering: tuple[str, ...]) -> list[str]:
        """Human-readable dependency violations (empty ⇒ valid)."""
        if set(ordering) != set(self.pass_names()):
            return ["ordering is not a permutation of the pipeline passes"]
        pos = {name: i for i, name in enumerate(ordering)}
        out = []
        for p in self.passes:
            for dep in p.must_follow:
                if pos[dep] > pos[p.name]:
                    out.append(f"{p.name} must follow {dep}")
        return out

    # ── reward: fusion effectiveness (lower cost = better) ───────────────────
    def cost(self, ordering: tuple[str, ...]) -> float:
        """Ops remaining after the ordering.  A fusion pass only removes ops if it
        runs before the first lowering pass freezes the graph."""
        first_lower = min(
            (i for i, n in enumerate(ordering)
             if self._by_name[n].kind == "lower"),
            default=len(ordering),
        )
        fused = sum(
            self._by_name[n].fuses
            for n in ordering[:first_lower]
            if self._by_name[n].kind == "fuse"
        )
        return float(self.base_ops - fused)

    def fitness(self, ordering: tuple[str, ...]) -> float:
        """Gated fitness for the agents: ``INF`` if the ordering is invalid
        (rejected — the Sakana invariant), else the op-cost (minimize)."""
        if not self.is_valid(tuple(ordering)):
            return INF
        return self.cost(tuple(ordering))

    def reward(self, ordering: tuple[str, ...]) -> float:
        """CompilerGym-style reward: reduction in cost vs the default pipeline.
        ``-inf`` for an invalid (rejected) ordering."""
        f = self.fitness(tuple(ordering))
        if f == INF:
            return -INF
        return self.cost(self.default_ordering()) - f


# ── agents: drive the env with the existing gated searchers ──────────────────


def _neighbors(ordering: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Adjacent-swap mutations — the local action steps."""
    out = []
    for i in range(len(ordering) - 1):
        nxt = list(ordering)
        nxt[i], nxt[i + 1] = nxt[i + 1], nxt[i]
        out.append(tuple(nxt))
    return out


def search_best_order(
    env: PassOrderEnv,
    candidates: list[tuple[str, ...]],
) -> tuple[tuple[str, ...] | None, float, list]:
    """Gated grid search over explicit candidate orderings (``magellan.search``).
    Invalid orderings score ``INF`` and are rejected."""
    return search(candidates, env.fitness)


def evolve_best_order(
    env: PassOrderEnv,
    start: tuple[str, ...] | None = None,
    *,
    steps: int = 64,
) -> tuple[tuple[str, ...], float]:
    """Gated hill-climb (``magellan.evolve``) from ``start`` (default: the
    canonical order) over adjacent-swap mutations.  Never accepts an invalid
    ordering (its fitness is ``INF``)."""
    start = start or env.default_ordering()
    best, best_f, _hist = evolve(start, _neighbors, env.fitness, iters=steps)
    return best, best_f


__all__ = [
    "PassSpec",
    "PIPELINE_PASSES",
    "PassOrderEnv",
    "search_best_order",
    "evolve_best_order",
]
