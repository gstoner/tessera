// RUN: tessera-opt %s -tpp-halo-infer | FileCheck %s
//
// Real halo inference: a central-difference gradient on a rank-2 field needs
// a radius-1 ghost cell in every spatial dimension, emitted as a structured
// i64 array (plus the max-width convenience scalar and an `inferred` marker).
func.func @halo_example(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %y = "tpp.grad"(%x) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: tpp.grad
  // CHECK-SAME: tpp.halo = [1, 1]
  // CHECK-SAME: tpp.halo.inferred
  // CHECK-SAME: tpp.halo.width = 1
  return %y : tensor<32x32xf32>
}

// A radius-2 (5-wide) stencil kernel implies a 2-cell halo per dimension,
// derived directly from the kernel operand's shape.
func.func @stencil_r2(%x: tensor<64x64xf32>, %k: tensor<5x5xf32>) -> tensor<64x64xf32> {
  %y = "tpp.stencil.apply"(%x, %k) : (tensor<64x64xf32>, tensor<5x5xf32>) -> tensor<64x64xf32>
  // CHECK: tpp.stencil.apply
  // CHECK-SAME: tpp.halo = [2, 2]
  // CHECK-SAME: tpp.halo.width = 2
  return %y : tensor<64x64xf32>
}

// A 4th-order gradient reaches +/-2 cells; an explicit `axis` makes it
// directional so only that dimension gets a halo.
func.func @grad_o4_dirx(%x: tensor<48x48xf32>) -> tensor<48x48xf32> {
  %y = "tpp.grad"(%x) { order = 4 : i64, axis = 1 : i64 } : (tensor<48x48xf32>) -> tensor<48x48xf32>
  // CHECK: tpp.grad
  // CHECK-SAME: tpp.halo = [0, 2]
  return %y : tensor<48x48xf32>
}
