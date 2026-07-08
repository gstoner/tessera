// RUN: tessera-opt %s -tpp-halo-infer -tpp-distribute-halo | FileCheck %s
//
// With no mesh on the module the ghost fill is a local periodic wrap, not a
// neighbour comm: local_only = true and overlap = none.  The exchange op is
// still materialised so codegen has a single place to emit the boundary fill.

func.func @grad(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %y = "tpp.grad"(%x) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  return %y : tensor<32x32xf32>
}

// CHECK: tpp.halo.exchange %arg0
// CHECK-SAME: tpp.dist.local_only = true
// CHECK-SAME: tpp.dist.overlap = "none"
