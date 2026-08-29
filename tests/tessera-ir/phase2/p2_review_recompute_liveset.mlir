// P2 code review (2026-08-29) — InsertRecomputePass compared a cumulative sum
// against the memory budget, never subtracting a value's bytes when its last
// use passed. The quantity checked was therefore "bytes produced since the last
// checkpoint", not live-set size, contradicting both the pass header and
// Decision #10's greedy live-set scan.
//
// This fixture lives here rather than beside the pass because
// src/solvers/scaling_resilience/tests only runs when the tessera_sr dialect is
// configured into the build; the budget path needs no tessera_sr op, so it can
// be covered unconditionally.
//
// RUN: tessera-opt --tessera-insert-recompute="memory-budget-mb=2" \
// RUN:   --allow-unregistered-dialect -split-input-file %s | FileCheck %s

// Each 1 MiB result dies at the next op, so true peak liveness is 2 MiB and
// fits the 2 MB budget. Under the cumulative sum this crossed after two ops and
// inserted checkpoints a program that needed none then had to recompute around.
// CHECK: tessera_sr.num_checkpoints = 0
// CHECK: @every_value_dies_immediately
func.func @every_value_dies_immediately(%a: tensor<512x512xf32>)
    -> tensor<512x512xf32> {
  %v0 = "tessera.gelu"(%a) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v1 = "tessera.gelu"(%v0) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v2 = "tessera.gelu"(%v1) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v3 = "tessera.gelu"(%v2) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v4 = "tessera.gelu"(%v3) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v5 = "tessera.gelu"(%v4) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  return %v5 : tensor<512x512xf32>
}

// -----

// Genuinely concurrent liveness must still trigger: four 1 MiB activations are
// all held until the reduction consumes them, so the peak really does exceed
// the budget. Without this the fix above would just be a disabled pass.
// CHECK: tessera_sr.num_checkpoints = 1
// CHECK: @values_held_live_still_checkpoint
func.func @values_held_live_still_checkpoint(%a: tensor<512x512xf32>)
    -> tensor<512x512xf32> {
  %v0 = "tessera.gelu"(%a) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v1 = "tessera.gelu"(%a) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v2 = "tessera.gelu"(%a) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %v3 = "tessera.gelu"(%a) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %s0 = "tessera.add"(%v0, %v1)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %s1 = "tessera.add"(%s0, %v2)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %s2 = "tessera.add"(%s1, %v3)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  return %s2 : tensor<512x512xf32>
}
