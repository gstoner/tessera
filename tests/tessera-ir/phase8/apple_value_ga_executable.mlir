// Stage 16E — cl30 Clifford geometric product emits an Apple GPU value call.
//
// RUN: %tessera_strict_opt %s -tessera-lower-to-apple_gpu-full | FileCheck %s

// CHECK-LABEL: func.func @clifford_geometric_product_value
// CHECK: tessera_apple.gpu.kernel_call
// CHECK-SAME: abi = "msl_clifford_geo_product_cl30_value_f32"
// CHECK-SAME: framework = "Metal"
// CHECK-SAME: op_kind = "clifford_geometric_product"
// CHECK-SAME: p = 3
// CHECK-SAME: q = 0
// CHECK-SAME: status = "executable"
// CHECK-SAME: symbol = "tessera_apple_gpu_clifford_geo_product_cl30_value_f32"
// CHECK-SAME: : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK-NOT: tile.clifford_geometric_product
func.func @clifford_geometric_product_value(%a: tensor<2x8xf32>,
                                            %b: tensor<2x8xf32>)
    -> tensor<2x8xf32> {
  %0 = tessera.clifford.geometric_product %a, %b
    {p = 3 : i64, q = 0 : i64}
    : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}
