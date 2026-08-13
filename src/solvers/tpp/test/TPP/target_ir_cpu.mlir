// RUN: tessera-opt %s -tpp-legalize-space-time -lower-tpp-to-target-ir | FileCheck %s

module attributes {tessera.target = "cpu"} {
  func.func @grad(%x: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %y = "tpp.grad"(%x) {axis = 1 : i64, scheme = "central", order = 4 : i64, spacing = [0.5 : f64, 0.25 : f64]} : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %y : tensor<8x8xf32>
  }
}

// CHECK: tpp.grad
// CHECK-SAME: spacing = [5.000000e-01, 2.500000e-01]
// CHECK-SAME: tessera.target_ir.call = "ts_stencil_grad_cpu"
// CHECK-SAME: tessera.target_ir.lowered
// CHECK-SAME: tessera.target_ir.status = "executable"
