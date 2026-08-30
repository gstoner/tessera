// Activation rematerialization emits a QUADRATIC number of clones for a deep
// producer chain, because the greedy selection prices each candidate by its own
// recompute cost while materialization clones it once per surviving consumer —
// and a tagged consumer contributes one consumer per clone it will itself
// produce. Measured before the bound existed: a 2000-deep chain took 4,001 ops
// to 2,001,002 in 16.4s.
//
// The chain below is 12 deep with every intermediate also summed into a sink,
// so all 12 activations are live to the end and the budget selects the whole
// chain — the shape that compounds.
//
// RUN: tessera-opt --tessera-activation-rematerialization="max-clone-expansion=0" \
// RUN:   %s | FileCheck %s --check-prefix=UNBOUNDED
// RUN: tessera-opt --tessera-activation-rematerialization="max-clone-expansion=2" \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=BOUNDED

// With the bound disabled the plan is taken as the greedy chose it. The
// projection is exact by construction — emitted ops = input + projected clones
// - selected originals erased (25 + 88 - 22 = 91) — which is what makes it
// usable as a pre-materialization gate rather than a post-hoc count.
// UNBOUNDED: tessera.remat_auto_selected = 22
// UNBOUNDED-SAME: tessera.remat_projected_clones = 88

// At 2x the function's op count the plan is trimmed by splitting the tagged
// chain at its midpoint rather than peeling its downstream end: 88 projected
// clones fall to 41 while only 2 of the 22 selections are given up, because
// removing a middle element turns one chain of length K into two of K/2 and
// takes the projection from ~K^2/2 to ~K^2/4 in a single drop.
// BOUNDED: REMAT_PLAN_CLONE_BOUND
// BOUNDED-SAME: dropped 2 selection(s)
// BOUNDED: tessera.remat_auto_selected = 20
// BOUNDED-SAME: tessera.remat_projected_clones = 41

func.func @deep_chain(%a: tensor<512x512xf32>) -> tensor<512x512xf32>
    attributes {tessera.remat_budget_mb = 1 : i64} {
  %v0 = "tessera.gelu"(%a) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v1 = "tessera.gelu"(%v0) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v2 = "tessera.gelu"(%v1) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v3 = "tessera.gelu"(%v2) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v4 = "tessera.gelu"(%v3) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v5 = "tessera.gelu"(%v4) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v6 = "tessera.gelu"(%v5) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v7 = "tessera.gelu"(%v6) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v8 = "tessera.gelu"(%v7) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v9 = "tessera.gelu"(%v8) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v10 = "tessera.gelu"(%v9) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v11 = "tessera.gelu"(%v10) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %s0 = "tessera.add"(%a, %v0)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %s1 = "tessera.add"(%s0, %v1)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %s2 = "tessera.add"(%s1, %v2)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %s3 = "tessera.add"(%s2, %v3)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %s4 = "tessera.add"(%s3, %v4)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %s5 = "tessera.add"(%s4, %v5)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %s6 = "tessera.add"(%s5, %v6)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %s7 = "tessera.add"(%s6, %v7)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %s8 = "tessera.add"(%s7, %v8)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %s9 = "tessera.add"(%s8, %v9)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %s10 = "tessera.add"(%s9, %v10)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %s11 = "tessera.add"(%s10, %v11)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  return %s11 : tensor<512x512xf32>
}
