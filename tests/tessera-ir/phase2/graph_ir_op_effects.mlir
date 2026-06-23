// RUN: tessera-opt --canonicalize --cse %s | FileCheck %s
//
// Phase 1 (front-to-back closure plan): per-op MemoryEffectsOpInterface on the
// non-pure Graph-IR ops makes generic CSE/DCE *sound* and precise —
//   - deterministic value ops (tessera.adam / adamw / arch.weighted_sum / ...)
//     carry [Pure], so identical instances CSE and dead instances DCE;
//   - effectful ops (random dropout / gumbel_softmax, stateful kv_cache / ring,
//     collective all_reduce, moe transport) carry explicit MemWrite/MemRead, so
//     they are never merged or removed by the optimizer.
// Before this change all of these ops had an empty trait list (no effect
// interface), which is conservatively safe but leaves the contract implicit.
// See docs/audit/compiler/COMPILER_AUDIT.md (Phase 1).

// ── Pure: two identical adam steps collapse to one (CSE). ──────────────────── #
// CHECK-LABEL: func.func @cse_pure_adam
// CHECK: tessera.adam
// CHECK-NOT: tessera.adam
func.func @cse_pure_adam(%p: tensor<4xf32>, %g: tensor<4xf32>, %m1: tensor<4xf32>, %m2: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %a:3 = "tessera.adam"(%p,%g,%m1,%m2) : (tensor<4xf32>,tensor<4xf32>,tensor<4xf32>,tensor<4xf32>) -> (tensor<4xf32>,tensor<4xf32>,tensor<4xf32>)
  %b:3 = "tessera.adam"(%p,%g,%m1,%m2) : (tensor<4xf32>,tensor<4xf32>,tensor<4xf32>,tensor<4xf32>) -> (tensor<4xf32>,tensor<4xf32>,tensor<4xf32>)
  return %a#0, %b#0 : tensor<4xf32>, tensor<4xf32>
}

// ── Random: two identical dropouts must NOT merge (distinct samples). ──────── #
// CHECK-LABEL: func.func @no_cse_random_dropout
// CHECK-COUNT-2: tessera.dropout
func.func @no_cse_random_dropout(%x: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %a = "tessera.dropout"(%x) {p = 1.000000e-01 : f64} : (tensor<4xf32>) -> tensor<4xf32>
  %b = "tessera.dropout"(%x) {p = 1.000000e-01 : f64} : (tensor<4xf32>) -> tensor<4xf32>
  return %a, %b : tensor<4xf32>, tensor<4xf32>
}

// ── DCE: dead Pure optimizer step is removed; stateful append is preserved. ── #
// CHECK-LABEL: func.func @dce_pure_keep_state
// CHECK-NOT: tessera.adamw
// CHECK: tessera.kv_cache.append
func.func @dce_pure_keep_state(%p: tensor<4xf32>, %g: tensor<4xf32>, %c: !tessera.kv_cache, %k: tensor<2x4xf32>, %v: tensor<2x4xf32>) {
  %u:2 = "tessera.adamw"(%p,%g) : (tensor<4xf32>,tensor<4xf32>) -> (tensor<4xf32>,tensor<4xf32>)
  %w = "tessera.kv_cache.append"(%c,%k,%v) : (!tessera.kv_cache, tensor<2x4xf32>, tensor<2x4xf32>) -> !tessera.kv_cache
  return
}

// ── Collective all_reduce (MemWrite) is preserved even if its result is dead. ─ #
// CHECK-LABEL: func.func @keep_collective
// CHECK: tessera.all_reduce
func.func @keep_collective(%x: tensor<4xf32>) {
  %0 = "tessera.all_reduce"(%x) : (tensor<4xf32>) -> tensor<4xf32>
  return
}
