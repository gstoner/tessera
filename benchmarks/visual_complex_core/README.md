# Visual-complex-core benchmark — GA × EBM cross-lane

Sister of `clifford_core` and `energy_core` — composes both lanes'
primitives in one flow.  Matches the M7 visual-complex milestone shape
(see `docs/status/visual_complex.md`) but stays domain-neutral.

## Composition

| Piece                          | Lane | Source                                  |
|--------------------------------|------|-----------------------------------------|
| Rotor sampling                 | GA   | `RotorSampler` (reused from clifford_core) |
| Sandwich rotation              | GA   | `tessera.ga.rotor_sandwich`             |
| Clifford-norm energy           | GA→EBM | `clifford_energy` (½ ‖state − target‖² via `ga.norm_squared`) |
| Annealing schedule             | EBM  | `annealing_schedule` (reused from energy_core) |
| Langevin step (analytic grad)  | EBM  | inline update with `_clifford_energy_grad` |
| Grade projection               | GA   | `tessera.ga.grade_projection(state, 2)` |
| Partition function             | EBM  | `tessera.ebm.partition.partition_exact_from_energies` |

Forward (deterministic from cfg.seed):

```
x      = normal(init_key, (B, 8))         # Cl(3, 0) coefficients
target = normal(target_key, (8,))
rotors = RotorSampler.next_rotor() × n_rotors

# GA lane
state = x
for R in rotors:
    state = rotor_sandwich(R, state)

# EBM lane (analytic gradient ∂E/∂c_i = c_i − t_i for Euclidean Cl(3,0))
sched = linspace(T_max, T_min, n_steps)
for step, T in enumerate(sched):
    state = state − η · (state − target) + √(2ηT) · ξ_step

# GA lane (multivector projection)
bivec = grade_projection(state, 2)

# EBM lane (scalar invariant)
Z = partition_exact_from_energies(½‖state − target‖², T_min)
```

## Run

```
PYTHONPATH=.:python python benchmarks/visual_complex_core/core.py \
    --reps 5 --output /tmp/vc.json
```

## Status

| Axis                    | Status              | Notes                          |
|-------------------------|---------------------|--------------------------------|
| Numerical contract      | locked, deterministic | bit-identical at fp32 |
| Cross-lane oracle       | `composition_oracle` re-derives the full chain inline | matches model output to fp32 tolerance |
| Backend                 | reference (CPU/numpy) | both lanes route through their Apple GPU MSL fast paths when Darwin + Cl(3, 0) + f32 |
| IR-visible              | `tests/tessera-ir/phase7/visual_complex_core_ir_visible.mlir` | 6 GA+EBM generic ops co-located in one function |
| JSON schema             | Architecture Decision #12 | ingestible by `tools/roofline_tools/` |

## Why this benchmark exists

Per-lane benchmarks (`clifford_core`, `energy_core`) already lock each
lane in isolation.  This benchmark catches **cross-lane composition**
bugs — the class of regression where a layout change in one lane (e.g.
multivector tensor shape) silently breaks the other lane's consumer
(e.g. EBM energy evaluator expecting that exact layout).

`composition_oracle` re-derives the chain inline using the same
primitives in the same order — any divergence localizes to cross-lane
composition (GA→EBM data flow, RNG threading across lanes, gradient
contract between energy and Langevin) rather than per-lane correctness.

The lit fixture co-locates GA and EBM ops in one function body — a
single lowering pipeline (when one ships) must handle both lanes' ops in
one traversal.
