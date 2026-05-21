# Clifford-core benchmark

Generic GA / Clifford-algebra benchmark — sister surface to
`benchmarks/grid_ai_core`.  Domain-neutral; exercises the GA primitives the
Apple GPU MSL kernels back, without baking in any specific application.

## Composition

| Piece                          | Source                                                 |
|--------------------------------|--------------------------------------------------------|
| Multivector tiling             | `tile_multivectors`                                    |
| Deterministic rotor sampling   | `RotorSampler` (Philox RNGKey)                         |
| Rotor construction             | `tessera.ga.rotor_from_axis`                           |
| Sandwich rotation              | `tessera.ga.rotor_sandwich`                            |
| Geometric-product composition  | `tessera.ga.geometric_product`                         |
| Grade projection               | `tessera.ga.grade_projection`                          |
| Scalar invariant               | `tessera.ga.norm_squared`                              |

Forward (Cl(3, 0) — 8 blades, Apple GPU MSL fast path):

```
x : (B, 8) multivector coefficients
  → tile_multivectors
  → RotorSampler → n_rotors rotors
  → sandwich chain: cur = rotor_sandwich(R_i, cur) for each i
  → geometric_product chain: composed = R_0 · R_1 · ... · x
  → grade_projection(sandwiched, grade=2)
  → norm_squared(composed)
```

## Run

```
PYTHONPATH=.:python python benchmarks/clifford_core/core.py \
    --reps 5 --output /tmp/clifford.json
```

## Status

| Axis                    | Status              | Notes                          |
|-------------------------|---------------------|--------------------------------|
| Numerical contract      | locked, deterministic | bit-identical at fp32 |
| Backend                 | reference (CPU/numpy) | Apple GPU MSL fast paths when Darwin + Cl(3, 0) + f32 |
| IR-visible              | `tests/tessera-ir/phase7/clifford_core_ir_visible.mlir` | 5 generic Clifford ops roundtrip |
| JSON schema             | Architecture Decision #12 | ingestible by `tools/roofline_tools/` |

## Why this benchmark exists

Per-op tests in `python/tessera/ga/` already cover individual primitives.
This benchmark exists to catch *composition* bugs — the kind where a
refactor of `CliffordIRProgram`'s SSA reference format or fused-kernel
boundary subtly breaks the rotor → sandwich → grade-projection chain
without breaking any single op.  An independent CPU oracle re-derives the
expected output via the same primitives in the same order; any divergence
localizes to composition, not to per-op correctness.
