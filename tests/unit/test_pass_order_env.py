"""Phase-ordering action space — gated pass-ordering environment (CompilerGym)."""

from __future__ import annotations

import itertools

from tessera.compiler.magellan import INF
from tessera.compiler.pass_order_env import (
    PIPELINE_PASSES,
    PassOrderEnv,
    evolve_best_order,
    search_best_order,
)


def test_default_ordering_is_valid():
    env = PassOrderEnv()
    assert env.is_valid(env.default_ordering())
    assert env.violations(env.default_ordering()) == []


def test_dependency_violation_is_rejected():
    env = PassOrderEnv()
    # put a fusion pass before canonicalize → violates must_follow
    order = list(env.default_ordering())
    ci = order.index("canonicalize")
    sf = order.index("swiglu_fusion")
    order[ci], order[sf] = order[sf], order[ci]
    bad = tuple(order)
    assert not env.is_valid(bad)
    assert env.fitness(bad) == INF                 # gated → rejected
    assert env.reward(bad) == -INF
    assert env.violations(bad)                     # names the broken constraint


def test_non_permutation_is_invalid():
    env = PassOrderEnv()
    assert not env.is_valid(("effect_annotation",))           # missing passes
    assert not env.is_valid(env.default_ordering() + ("x",))  # extra pass


def test_fusion_before_lowering_beats_fusion_after():
    env = PassOrderEnv()
    # canonical order fuses everything before tile_lowering → minimal cost
    good = env.default_ordering()
    assert env.is_valid(good)
    good_cost = env.cost(good)

    # move tile_lowering before the fusion passes (still dependency-valid:
    # tile_lowering only requires canonicalize) → fusion runs after lowering →
    # eliminates nothing → higher cost.
    order = [p for p in good if p not in ("tile_lowering",)]
    ci = order.index("canonicalize")
    order.insert(ci + 1, "tile_lowering")
    late_fusion = tuple(order)
    assert env.is_valid(late_fusion)
    assert env.cost(late_fusion) > good_cost       # lost the fusion opportunity


def test_search_rejects_invalid_and_finds_min_cost():
    env = PassOrderEnv()
    # a small candidate set: the canonical order + a few permutations
    names = env.pass_names()
    cands = [env.default_ordering()]
    # add some random-ish permutations (deterministic via itertools)
    for perm in itertools.islice(itertools.permutations(names), 0, 200, 37):
        cands.append(perm)
    best, best_f, history = search_best_order(env, cands)
    assert best is not None
    assert best_f < INF and env.is_valid(best)
    # the best valid ordering fuses all 5 ops before lowering → cost base-5
    assert best_f == env.base_ops - 5
    # every infeasible candidate scored INF
    assert any(f == INF for _c, f in history)


def test_evolve_stays_valid_and_improves_or_holds():
    env = PassOrderEnv()
    best, best_f = evolve_best_order(env, steps=128)
    assert env.is_valid(best)                      # never accepts an invalid order
    assert best_f <= env.cost(env.default_ordering())   # never worse than start


def test_reward_is_improvement_over_default_pipeline():
    env = PassOrderEnv()
    # the default pipeline is the reference; its reward is 0 (no improvement over
    # itself) and it is the optimum here, so nothing scores positive.
    assert env.reward(env.default_ordering()) == 0.0


def test_pipeline_passes_mirror_real_spine():
    names = {p.name for p in PIPELINE_PASSES}
    assert {"effect_annotation", "canonicalize", "tile_lowering",
            "collective_insertion"} <= names
