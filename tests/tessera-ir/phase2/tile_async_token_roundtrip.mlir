// Phase B (2026-06-23) — !tile.async_token round-trip.
//
// The async-copy completion token is the SSA edge that replaces the transitional
// tile.barrier_id / tile.depends_on string markers: tile.async_copy produces it,
// tile.wait_async and tile.mma consume it. It rides the ops' existing
// Variadic<AnyType> operand/result slots, so no op signature changed and the four
// sync ops + the type all parse under the STRICT driver (no
// --allow-unregistered-dialect) — proof the type is first-class, not opaque.
//
// RUN: %tessera_strict_opt %s | FileCheck %s
// RUN: %tessera_strict_opt %s | %tessera_strict_opt | FileCheck %s

// CHECK-LABEL: func.func @async_token_edge
func.func @async_token_edge(%A: tensor<16x16xf16>, %B: tensor<16x16xf16>)
    -> tensor<16x16xf32> {
  // CHECK: %[[CP:.*]]:2 = tile.async_copy %{{.*}} : (tensor<16x16xf16>) -> (tensor<16x16xf16>, !tile.async_token)
  %tile, %tok = "tile.async_copy"(%A)
      : (tensor<16x16xf16>) -> (tensor<16x16xf16>, !tile.async_token)
  // CHECK: tile.wait_async %[[CP]]#1 : (!tile.async_token) -> ()
  "tile.wait_async"(%tok) : (!tile.async_token) -> ()
  // CHECK: tile.mma %[[CP]]#0, %{{.*}}, %[[CP]]#1 : (tensor<16x16xf16>, tensor<16x16xf16>, !tile.async_token) -> tensor<16x16xf32>
  %C = "tile.mma"(%tile, %B, %tok)
      : (tensor<16x16xf16>, tensor<16x16xf16>, !tile.async_token)
        -> tensor<16x16xf32>
  return %C : tensor<16x16xf32>
}
