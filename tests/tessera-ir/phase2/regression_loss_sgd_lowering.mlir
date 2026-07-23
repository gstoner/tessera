// RUN: tessera-opt --tessera-to-linalg %s | FileCheck %s
//
// Piecewise regression losses and functional SGD lower through one dynamic
// shared Linalg envelope. Boundary predicates are part of the contract:
// MAE uses a zero tie subgradient, Huber includes |e| == delta in its
// quadratic branch, and Smooth-L1 excludes |e| == beta from its quadratic
// branch.

module {
  // CHECK-LABEL: func.func @mae_backward
  // CHECK: tensor.dim
  // CHECK: arith.cmpf ogt
  // CHECK: arith.cmpf olt
  // CHECK: arith.select
  // CHECK: arith.negf
  // CHECK-NOT: tessera.loss.regression_backward
  func.func @mae_backward(
      %prediction: tensor<?x5xf32>, %target: tensor<?x5xf32>,
      %cotangent: tensor<f32>) -> (tensor<?x5xf32>, tensor<?x5xf32>) {
    %dp, %dt = "tessera.loss.regression_backward"(
        %prediction, %target, %cotangent)
        {kind = "mae", parameter = 1.0 : f64, reduction = "mean"} :
        (tensor<?x5xf32>, tensor<?x5xf32>, tensor<f32>) ->
        (tensor<?x5xf32>, tensor<?x5xf32>)
    return %dp, %dt : tensor<?x5xf32>, tensor<?x5xf32>
  }

  // CHECK-LABEL: func.func @huber_none
  // CHECK: math.absf
  // CHECK: arith.cmpf ole
  // CHECK: arith.select
  // CHECK-NOT: tessera.loss.huber
  func.func @huber_none(%prediction: tensor<?x7xf16>,
                        %target: tensor<?x7xf16>) -> tensor<?x7xf16> {
    %loss = "tessera.loss.huber"(%prediction, %target)
        {delta = 0.75 : f64, reduction = "none"} :
        (tensor<?x7xf16>, tensor<?x7xf16>) -> tensor<?x7xf16>
    return %loss : tensor<?x7xf16>
  }

  // CHECK-LABEL: func.func @smooth_l1_backward
  // CHECK: math.absf
  // CHECK: arith.divf
  // CHECK: arith.cmpf olt
  // CHECK-NOT: tessera.loss.regression_backward
  func.func @smooth_l1_backward(
      %prediction: tensor<2x3xbf16>, %target: tensor<2x3xbf16>,
      %cotangent: tensor<2x3xbf16>) ->
      (tensor<2x3xbf16>, tensor<2x3xbf16>) {
    %dp, %dt = "tessera.loss.regression_backward"(
        %prediction, %target, %cotangent)
        {kind = "smooth_l1", parameter = 0.5 : f64, reduction = "none"} :
        (tensor<2x3xbf16>, tensor<2x3xbf16>, tensor<2x3xbf16>) ->
        (tensor<2x3xbf16>, tensor<2x3xbf16>)
    return %dp, %dt : tensor<2x3xbf16>, tensor<2x3xbf16>
  }

  // CHECK-LABEL: func.func @dynamic_sgd
  // CHECK: tensor.dim
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  // CHECK: arith.subf
  // CHECK-NOT: tessera.sgd
  func.func @dynamic_sgd(%param: tensor<?x11xf32>,
                         %grad: tensor<?x11xf32>) -> tensor<?x11xf32> {
    %updated = "tessera.sgd"(%param, %grad) {lr = 0.125 : f64} :
        (tensor<?x11xf32>, tensor<?x11xf32>) -> tensor<?x11xf32>
    return %updated : tensor<?x11xf32>
  }

  // CHECK-LABEL: func.func @dynamic_sgd_backward
  // CHECK-COUNT-2: linalg.generic
  // CHECK: arith.mulf
  // CHECK-NOT: tessera.sgd_backward
  func.func @dynamic_sgd_backward(%dy: tensor<?x11xf32>) ->
      (tensor<?x11xf32>, tensor<?x11xf32>) {
    %dp, %dg = "tessera.sgd_backward"(%dy) {lr = 0.125 : f64} :
        (tensor<?x11xf32>) -> (tensor<?x11xf32>, tensor<?x11xf32>)
    return %dp, %dg : tensor<?x11xf32>, tensor<?x11xf32>
  }
}
