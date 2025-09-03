# Tessera MORL Starter (v1)

This starter adds **Multi-Objective Reinforcement Learning (MORL)** building blocks to the Tessera programming model.

## What’s inside
- `docs/` – split spec with **merge markers** so you can reassemble into a single doc.
- `morl/` – Tessera kernels for Pareto frontier, scalarization, gradient surgery (PCGrad), and reduction helpers.
- `python/` – a minimal training loop for **MO-PPO** with a tiny gridworld **DeepSeaTreasure** and **ResourceGathering** envs (pure-Python, no extra deps).
- `mlir/` – example **Tessera Target IR** snippets and a sample pipeline script.
- `tests/` – small tests for Pareto ops (Python fallback for quick validation).

> Note: Kernels are written in Tessera-style DSL and accompanied by an illustrative Target-IR form to guide lowering. Stubs/fallbacks are provided to run on CPU today while you wire to your tessera runtime.

## Quick start
```bash
# (A) Python-only functional check (Pareto ops + MO-PPO loop with small nets)
python3 python/train_moppo_demo.py --env deep_sea --steps 1000 --pref 0.5,0.5

# (B) Swap in Tessera kernels when the runtime is connected
# export TESSERA_RUNTIME=1  # (your env/flag)
python3 python/train_moppo_demo.py --use-tessera --env deep_sea
```

## Files of interest
- `morl/pareto_ops.tsr` – O(N²) tile-parallel non-dominated filter.
- `morl/scalarize.tsr` – Linear and Tchebycheff scalarization.
- `morl/pcgrad.tsr` – Pairwise gradient conflict resolution.
- `mlir/pareto_ops_example.mlir` – Target-IR sketch + lowering notes.
- `python/train_moppo_demo.py` – MO-PPO training loop with preferences.
