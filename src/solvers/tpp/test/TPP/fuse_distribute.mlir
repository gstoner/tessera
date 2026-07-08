// RUN: tessera-opt %s -tpp-legalize-space-time -tpp-halo-infer -tpp-fuse-stencil-time -tpp-distribute-halo | FileCheck %s
//
// End-to-end fusion payoff: two sibling gradients that read %h share ONE halo
// exchange (of the union halo [1,1]) instead of one each.  The exchange count
// is exactly one.

module attributes {tessera.mesh.axes = ["x", "y"]} {
  func.func @grads(%h: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
    %hx = "tpp.grad"(%h) { axis = 0 : i64 } : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %hy = "tpp.grad"(%h) { axis = 1 : i64 } : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %hx, %hy : tensor<64x64xf32>, tensor<64x64xf32>
  }
}

// Exactly one exchange, carrying the fused [1,1] halo, feeding both grads.
// CHECK-COUNT-1: tpp.halo.exchange
// CHECK-SAME: tpp.halo = [1, 1]
// CHECK-NOT: tpp.halo.exchange
// CHECK: tpp.grad
// CHECK-SAME: tpp.halo.distributed
// CHECK: tpp.grad
// CHECK-SAME: tpp.halo.distributed
