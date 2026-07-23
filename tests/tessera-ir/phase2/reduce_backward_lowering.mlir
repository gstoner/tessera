// RUN: tessera-opt --tessera-to-linalg %s | FileCheck %s
//
// Runtime mean extent and equal-share extrema ties lower through the shared
// linalg path.  No target emitter has to reinterpret Graph-level policy.

module {
  func.func @dynamic_mean_backward(
      %x: tensor<2x?xf32>, %mean: tensor<2xf32>, %dy: tensor<2xf32>)
      -> tensor<2x?xf32> {
    %dx = "tessera.reduce_backward"(%x, %mean, %dy)
        {kind = "mean", axis = 1 : i64} :
        (tensor<2x?xf32>, tensor<2xf32>, tensor<2xf32>)
        -> tensor<2x?xf32>
    return %dx : tensor<2x?xf32>
  }

  // CHECK-LABEL: func.func @dynamic_mean_backward
  // CHECK: tensor.dim
  // CHECK: arith.divf
  // CHECK: linalg.generic
  // CHECK-NOT: tessera.reduce_backward

  func.func @max_tie_backward(
      %x: tensor<2x4xf32>, %max: tensor<2xf32>, %dy: tensor<2xf32>)
      -> tensor<2x4xf32> {
    %dx = "tessera.reduce_backward"(%x, %max, %dy)
        {kind = "max", axis = 1 : i64, tie_policy = "equal"} :
        (tensor<2x4xf32>, tensor<2xf32>, tensor<2xf32>)
        -> tensor<2x4xf32>
    return %dx : tensor<2x4xf32>
  }

  // CHECK-LABEL: func.func @max_tie_backward
  // CHECK: arith.cmpf oeq
  // CHECK: linalg.reduce
  // CHECK: arith.divf
  // CHECK: arith.select
  // CHECK-NOT: tessera.reduce_backward
}
