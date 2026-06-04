// Stage 12 — standalone Apple value Tile verifier accepts registered allowlist.
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
