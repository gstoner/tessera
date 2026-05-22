<<MERGE_START: MORL_Spec>>
# Tessera MORL Design — Part I

## Motivation
Multi-objective RL (MORL) optimizes **vector rewards** `r_t ∈ R^M` under a single policy π(a|s). Downstream consumers specify a **preference vector** `w ∈ Δ^{M-1}` or a per-episode constraint (`C ≤ cmax`). Tessera’s tile model maps naturally to common MORL primitives:
- batched rollout with per-objective accumulation,
- **Pareto frontier** filtering,
- **scalarization** (linear / (ε-)Tchebycheff / Chebyshev),
- **gradient surgery** (PCGrad) to avoid conflicting updates.

## Algorithms included
- **MO-PPO**: compute per-objective advantages `A^(m)`, scalarize to `Â = S(A; w)`, then PPO update.
- **PCGrad**: project gradient g_i so dot(g_i, g_j) ≥ 0 for all j.
- **Lexicographic** (optional): first satisfy primary objective, then improve secondary ones.

## Data model
Let batch B, horizon T, objectives M.
- Rewards: `R ∈ ℝ^{B×T×M}`
- Advantages: `A ∈ ℝ^{B×T×M}`
- Preferences: `w ∈ ℝ^{M}` (‖w‖₁=1, w≥0) or constraint vector `cmax`
- Scalarized advantages: `Â ∈ ℝ^{B×T}`

Tessera kernels provide:
- `pareto_filter`: indices of non-dominated points along last dim M or across a set of points.
- `scalarize_linear`, `scalarize_tchebycheff`.
- `pcgrad_pairwise`: conflict-avoidant gradient composition.

## Kernel surfaces
**Pareto filter** (point-set, O(N²) per tile, N ≤ 1024 typical)
```tessera
// morl/pareto_ops.tsr
package morl

@kernel pareto_filter(
    points: tensor<*xMxf32>,   // N×M points; last dim = objectives
    out_mask: tensor<*xi1>     // N bools; 1 if non-dominated
) tile[TM=128] {
  // Shared memory for a tile of points (TM×M)
  smem p_tile: f32[TM, M]
  smem dominated: i1[TM]

  for base in range(0, N, TM) {
    load p_tile from points[base:base+TM, :]
    fill dominated[:] = 0

    // O(TM²) within tile; cross-tile comparisons stream in chunks
    for other in range(0, N, TM) {
      smem q_tile: f32[TM, M]
      load q_tile from points[other:other+TM, :]
      barrier()

      // Each thread handles one candidate i
      parallel_for i in 0..TM {
        if (base+i >= N) continue
        if (dominated[i]) continue
        // Check if any q dominates p_i (q >= p_i ∀m and > for some m)
        acc_all = true
        acc_strict = false
        for j in 0..TM {
          if (other+j >= N) break
          all_ge = true; any_gt = false
          for m in 0..M {
            all_ge &= (q_tile[j,m] >= p_tile[i,m])
            any_gt |= (q_tile[j,m] >  p_tile[i,m])
          }
          if (all_ge & any_gt) { dominated[i] = 1; break }
        }
      }
      barrier()
    }

    store out_mask[base:base+TM] = not dominated[:]
  }
}
```
