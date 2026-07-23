// RUN: tessera-opt --tessera-training-step-fusion %s | FileCheck %s --check-prefix=FUSE
// RUN: tessera-opt --tessera-training-step-fusion --tessera-to-linalg %s | FileCheck %s --check-prefix=LOWER
//
// The fusion is legal only when the loss prediction gradient has one consumer.
// Target gradients remain observable. Mean scaling uses runtime extents and the
// fused carrier lowers to one Linalg loop instead of a gradient materialization
// followed by a second optimizer loop.

module {
  // FUSE-LABEL: func.func @mse_mean
  // FUSE: %[[NEW:.+]], %[[DT:.+]] = tessera.training.loss_sgd
  // FUSE-SAME: kind = "mse"
  // FUSE-SAME: lr = 1.250000e-01
  // FUSE-NOT: tessera.loss.mse_backward
  // FUSE-NOT: tessera.sgd
  //
  // LOWER-LABEL: func.func @mse_mean
  // LOWER: tensor.dim
  // LOWER-COUNT-1: linalg.generic
  // LOWER: arith.mulf
  // LOWER: arith.negf
  // LOWER: arith.subf
  // LOWER-NOT: tessera.training.loss_sgd
  func.func @mse_mean(
      %prediction: tensor<?x5xf32>, %target: tensor<?x5xf32>,
      %cotangent: tensor<f32>, %param: tensor<?x5xf32>) ->
      (tensor<?x5xf32>, tensor<?x5xf32>) {
    %dp, %dt = "tessera.loss.mse_backward"(
        %prediction, %target, %cotangent) {reduction = "mean"} :
        (tensor<?x5xf32>, tensor<?x5xf32>, tensor<f32>) ->
        (tensor<?x5xf32>, tensor<?x5xf32>)
    %new = "tessera.sgd"(%param, %dp) {lr = 0.125 : f64} :
        (tensor<?x5xf32>, tensor<?x5xf32>) -> tensor<?x5xf32>
    return %new, %dt : tensor<?x5xf32>, tensor<?x5xf32>
  }

  // FUSE-LABEL: func.func @smooth_l1_none
  // FUSE: tessera.training.loss_sgd
  // FUSE-SAME: kind = "smooth_l1"
  // FUSE-SAME: parameter = 5.000000e-01
  // FUSE-SAME: reduction = "none"
  //
  // LOWER-LABEL: func.func @smooth_l1_none
  // LOWER-COUNT-1: linalg.generic
  // LOWER: math.absf
  // LOWER: arith.cmpf olt
  func.func @smooth_l1_none(
      %prediction: tensor<2x3xf32>, %target: tensor<2x3xf32>,
      %cotangent: tensor<2x3xf32>, %param: tensor<2x3xf32>) ->
      (tensor<2x3xf32>, tensor<2x3xf32>) {
    %dp, %dt = "tessera.loss.regression_backward"(
        %prediction, %target, %cotangent)
        {kind = "smooth_l1", parameter = 0.5 : f64, reduction = "none"} :
        (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) ->
        (tensor<2x3xf32>, tensor<2x3xf32>)
    %new = "tessera.sgd"(%param, %dp) {lr = 0.01 : f64} :
        (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %new, %dt : tensor<2x3xf32>, tensor<2x3xf32>
  }

  // BCE target gradients are -logits*dy, not the negated logits gradient.
  // FUSE-LABEL: func.func @bce_mean
  // FUSE: tessera.training.loss_sgd
  // FUSE-SAME: kind = "bce"
  //
  // LOWER-LABEL: func.func @bce_mean
  // LOWER-COUNT-1: linalg.generic
  // LOWER: math.exp
  // LOWER: arith.select
  func.func @bce_mean(
      %logits: tensor<?x7xf32>, %target: tensor<?x7xf32>,
      %cotangent: tensor<f32>, %param: tensor<?x7xf32>) ->
      (tensor<?x7xf32>, tensor<?x7xf32>) {
    %dz, %dt = "tessera.loss.binary_cross_entropy_backward"(
        %logits, %target, %cotangent) {reduction = "mean"} :
        (tensor<?x7xf32>, tensor<?x7xf32>, tensor<f32>) ->
        (tensor<?x7xf32>, tensor<?x7xf32>)
    %new = "tessera.sgd"(%param, %dz) {lr = 0.02 : f64} :
        (tensor<?x7xf32>, tensor<?x7xf32>) -> tensor<?x7xf32>
    return %new, %dt : tensor<?x7xf32>, tensor<?x7xf32>
  }

  // AdamW's parameter and both moment states stay explicit, but the loss
  // prediction gradient is never materialized.
  // FUSE-LABEL: func.func @mse_adamw
  // FUSE: %[[P:.+]], %[[M:.+]], %[[V:.+]], %[[DT:.+]] = tessera.training.loss_adamw
  // FUSE-SAME: beta1 = 8.000000e-01
  // FUSE-SAME: step = 7
  // FUSE-NOT: tessera.loss.mse_backward
  // FUSE-NOT: tessera.adamw
  //
  // LOWER-LABEL: func.func @mse_adamw
  // LOWER: tensor.dim
  // LOWER-COUNT-1: linalg.generic
  // LOWER: math.sqrt
  // LOWER-NOT: tessera.training.loss_adamw
  func.func @mse_adamw(
      %prediction: tensor<?x11xf32>, %target: tensor<?x11xf32>,
      %cotangent: tensor<f32>, %param: tensor<?x11xf32>,
      %moment1: tensor<?x11xf32>, %moment2: tensor<?x11xf32>) ->
      (tensor<?x11xf32>, tensor<?x11xf32>, tensor<?x11xf32>,
       tensor<?x11xf32>) {
    %dp, %dt = "tessera.loss.mse_backward"(
        %prediction, %target, %cotangent) {reduction = "mean"} :
        (tensor<?x11xf32>, tensor<?x11xf32>, tensor<f32>) ->
        (tensor<?x11xf32>, tensor<?x11xf32>)
    %new_p, %new_m, %new_v = "tessera.adamw"(
        %param, %dp, %moment1, %moment2)
        {lr = 2.0e-3 : f64, beta1 = 8.0e-1 : f64,
         beta2 = 9.5e-1 : f64, eps = 1.0e-7 : f64,
         weight_decay = 1.0e-2 : f64, step = 7 : i64} :
        (tensor<?x11xf32>, tensor<?x11xf32>, tensor<?x11xf32>,
         tensor<?x11xf32>) ->
        (tensor<?x11xf32>, tensor<?x11xf32>, tensor<?x11xf32>)
    return %new_p, %new_m, %new_v, %dt :
        tensor<?x11xf32>, tensor<?x11xf32>, tensor<?x11xf32>,
        tensor<?x11xf32>
  }

  // A second prediction-gradient use makes fusion illegal because eliminating
  // the materialized gradient would change the program's observable values.
  // FUSE-LABEL: func.func @shared_gradient
  // FUSE: %[[DP:.+]], %[[DT:.+]] = tessera.loss.mse_backward
  // FUSE: tessera.sgd
  // FUSE-NOT: tessera.training.loss_sgd
  func.func @shared_gradient(
      %prediction: tensor<4xf32>, %target: tensor<4xf32>,
      %cotangent: tensor<f32>, %param: tensor<4xf32>) ->
      (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %dp, %dt = "tessera.loss.mse_backward"(
        %prediction, %target, %cotangent) {reduction = "sum"} :
        (tensor<4xf32>, tensor<4xf32>, tensor<f32>) ->
        (tensor<4xf32>, tensor<4xf32>)
    %new = "tessera.sgd"(%param, %dp) {lr = 0.1 : f64} :
        (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %new, %dp, %dt : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
}
