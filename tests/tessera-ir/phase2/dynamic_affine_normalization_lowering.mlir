// RUN: tessera-opt --tessera-to-linalg %s | FileCheck %s

module {
  func.func @dynamic_rmsnorm_affine(
      %x: tensor<?x?xf32>, %gamma: tensor<?xf32>) -> tensor<?x?xf32> {
    // CHECK-LABEL: func.func @dynamic_rmsnorm_affine
    // CHECK-NOT: tessera.rmsnorm
    // CHECK: tensor.dim %arg0
    // CHECK: linalg.reduce
    // CHECK: arith.index_cast
    // CHECK: arith.sitofp
    // CHECK: linalg.generic
    %y = "tessera.rmsnorm"(%x, %gamma) {eps = 1.0e-5 : f64} :
        (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
    return %y : tensor<?x?xf32>
  }

  func.func @dynamic_layernorm_affine(
      %x: tensor<?x?xf32>, %gamma: tensor<?xf32>, %beta: tensor<?xf32>)
      -> tensor<?x?xf32> {
    // CHECK-LABEL: func.func @dynamic_layernorm_affine
    // CHECK-NOT: tessera.layer_norm
    // CHECK: tensor.dim %arg0
    // CHECK: linalg.reduce
    // CHECK: math.sqrt
    // CHECK: linalg.generic
    %y = "tessera.layer_norm"(%x, %gamma, %beta) {eps = 1.0e-5 : f64} :
        (tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
    return %y : tensor<?x?xf32>
  }

  func.func @dynamic_stats_broadcast(%x: tensor<?x?xf32>) -> tensor<?x?xf32> {
    // CHECK-LABEL: func.func @dynamic_stats_broadcast
    // CHECK-NOT: tessera.normalization_stats
    // CHECK-NOT: tessera.broadcast_in_dim
    // CHECK: tensor.dim %arg0
    %center, %inv = "tessera.normalization_stats"(%x) :
        (tensor<?x?xf32>) -> (tensor<?xf32>, tensor<?xf32>)
    %full = "tessera.broadcast_in_dim"(%inv, %x)
        {broadcast_dimensions = [0]} :
        (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %full : tensor<?x?xf32>
  }
}
