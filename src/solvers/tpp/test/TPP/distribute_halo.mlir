// RUN: tessera-opt %s -tpp-halo-infer -tpp-distribute-halo | FileCheck %s
//
// DistributeHalo materialises a real `tpp.halo.exchange` op in front of each
// halo-annotated consumer, carrying the exchange plan (widths, mesh axes,
// overlap token).  With a mesh present it is a neighbour comm (local_only =
// false, overlap = comm_q_default) and the consumer is rewritten to read the
// exchanged value.

module attributes {tessera.mesh.axes = ["dp", "tp"]} {
  func.func @grad(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %y = "tpp.grad"(%x) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %y : tensor<32x32xf32>
  }
}

// CHECK: %[[EX:.*]] = tpp.halo.exchange %arg0
// CHECK-SAME: tpp.dist.local_only = false
// CHECK-SAME: tpp.dist.overlap = "comm_q_default"
// CHECK-SAME: tpp.halo = [1, 1]
// CHECK-SAME: tpp.mesh.axes = ["dp", "tp"]
// CHECK: tpp.grad %[[EX]]
// CHECK-SAME: tpp.halo.distributed
