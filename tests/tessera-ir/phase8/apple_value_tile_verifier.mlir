// Stage 12 — standalone Apple value Tile verifier accepts registered allowlist.
//
// REQUIRES: tessera-apple-backend
//
// RUN: %tessera_strict_opt %s -tessera-verify-apple-value-tile-ir | FileCheck %s

func.func @allowed_value_tile(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>)
    -> tensor<4x16xf32> {
  // CHECK-LABEL: func.func @allowed_value_tile
  // CHECK: tile.matmul
  %0 = tile.matmul %a, %b {source = "tessera.matmul"}
    : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func.func @allowed_clifford_value_tile
// CHECK: tile.clifford_geometric_product
func.func @allowed_clifford_value_tile(%a: tensor<2x8xf32>,
                                       %b: tensor<2x8xf32>)
                                       -> tensor<2x8xf32> {
  %0 = tile.clifford_geometric_product %a, %b
    {p = 3 : i64, q = 0 : i64, source = "tessera.clifford.geometric_product"}
    : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}
