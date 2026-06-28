// P1 (S_SERIES_GAP_CLOSURE_PLAN §6.A/§6.B) — the `tessera.view` 0-view op
// (contiguous reshape sharing storage) parses, verifies, round-trips, and folds
// like its structural siblings. `reshape` already had its own ODS op; `view` is
// the new one. Pure, SameOperandsAndResultElementType: a view that changes the
// element type is rejected. An identity view (result type == input) folds away.
//
// RUN: tessera-opt %s | tessera-opt | FileCheck %s
// RUN: tessera-opt %s -canonicalize | FileCheck %s --check-prefix=FOLD

// CHECK-LABEL: func.func @view_chain
func.func @view_chain(%x: tensor<2x3x4xf32>) -> tensor<24xf32> {
  // CHECK: tessera.view %{{.*}} {shape = [6, 4]}
  %a = "tessera.view"(%x) {shape = [6, 4]}
      : (tensor<2x3x4xf32>) -> tensor<6x4xf32>
  // CHECK: tessera.view %{{.*}} {shape = [24]}
  %b = "tessera.view"(%a) {shape = [24]} : (tensor<6x4xf32>) -> tensor<24xf32>
  return %b : tensor<24xf32>
}

// FOLD-LABEL: func.func @identity_view
func.func @identity_view(%x: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // an identity view (result type == operand type) folds to its input
  // FOLD-NOT: tessera.view
  %a = "tessera.view"(%x) {shape = [4, 5]} : (tensor<4x5xf32>) -> tensor<4x5xf32>
  // FOLD: return %arg0
  return %a : tensor<4x5xf32>
}

// FOLD-LABEL: func.func @dynamic_view_preserved
func.func @dynamic_view_preserved(%x: tensor<?xf32>) -> tensor<?xf32> {
  // type equality is not runtime identity for dynamic shapes — must NOT fold
  // FOLD: tessera.view
  %a = "tessera.view"(%x) {shape = [8]} : (tensor<?xf32>) -> tensor<?xf32>
  return %a : tensor<?xf32>
}
