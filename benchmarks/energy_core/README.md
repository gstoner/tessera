# Energy-core benchmark

Generic EBM benchmark — sister to `clifford_core` and `grid_ai_core`.
Domain-neutral; exercises the EBM primitives the Apple GPU MSL kernels
back: quadratic energy, stable logsumexp partition, Langevin step,
linear annealing schedule.

## Composition

| Piece                          | Source                                                 |
|--------------------------------|--------------------------------------------------------|
| Initial state sampling         | `tessera.rng.RNGKey` + `normal`                       |
| Quadratic energy               | `tessera.ebm.energy.energy_quadratic`                  |
| Annealing schedule             | `annealing_schedule` (T_max → T_min linear)            |
| Langevin step                  | `tessera.ebm.energy.langevin_step`                     |
| Partition function             | `tessera.ebm.partition.partition_exact_from_energies`  |

Forward (deterministic from cfg.seed):

```
y_0    = normal(init_key, (B, D))
target = normal(target_key, (D,))
sched  = linspace(T_max, T_min, n_steps)
for step, T in enumerate(sched):
    y_{step+1} = langevin_step(
        y_step, lambda a: energy_quadratic(a, target),
        eta, T, chain_key.fold_in(step))
Z      = partition_exact_from_energies(energies(y_final), T_min)
log_Z  = log(Z)
```

## Run

```
PYTHONPATH=.:python python benchmarks/energy_core/core.py \
    --reps 5 --output /tmp/energy.json
```

## Status

| Axis                    | Status              | Notes                          |
|-------------------------|---------------------|--------------------------------|
| Numerical contract      | locked, deterministic | bit-identical at fp32 |
| Backend                 | reference (CPU/numpy) | Apple GPU MSL fast paths via `tessera.ebm.*` when Darwin + f32 |
| IR-visible              | `tests/tessera-ir/phase7/energy_core_ir_visible.mlir` | 5 generic EBM ops roundtrip |
| JSON schema             | Architecture Decision #12 | ingestible by `tools/roofline_tools/` |

## Why this benchmark exists

Per-op tests in `python/tessera/ebm/` cover individual primitives.  This
benchmark catches **composition** bugs — the kind where a refactor of the
EBM IR or a numeric-stability change subtly breaks the annealed-Langevin
trajectory without breaking any single op.  `langevin_chain_oracle`
re-derives the expected trajectory inline (analytic gradient + same RNG
lineage), so any divergence localizes to composition (op order, RNG
threading, gradient handoff).
