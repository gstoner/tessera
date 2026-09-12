// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile %s | FileCheck %s
module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512", tessera.launch_bindings = ["x", "out"]} {
  func.func @absolute(%x: tensor<3x17xf32> {tessera.dim_names = ["3", "17"]}) -> tensor<3x17xf32> {
    %out = tessera.absolute %x : (tensor<3x17xf32>) -> tensor<3x17xf32>
    return %out : tensor<3x17xf32>
  }
}
// CHECK: llvm.func @tessera_tile_x86_unary_abs
// CHECK-SAME: tessera.absolute_contract
// CHECK-SAME: numeric_policy = "ieee_abs_clear_sign"
// CHECK-SAME: shape = array<i64: 3, 17>
// CHECK: tile.elementwise_kernel
// CHECK-SAME: kind = "abs"
// CHECK-NOT: tessera.absolute %
