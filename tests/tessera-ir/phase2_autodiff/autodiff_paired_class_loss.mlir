// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s

module {
  func.func @class_loss(%logits: tensor<?x13x?xf32>,
                        %target: tensor<?x?xi64>) -> tensor<f32>
      attributes {tessera.autodiff = "reverse"} {
    %loss = "tessera.loss.cross_entropy"(%logits, %target)
        {axis = 1 : i64, ignore_index = -9 : i64,
         label_smoothing = 1.5e-1 : f64, reduction = "mean"} :
        (tensor<?x13x?xf32>, tensor<?x?xi64>) -> tensor<f32>
    return %loss : tensor<f32>
  }

  // CHECK-LABEL: func.func @class_loss__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: %[[DLOGITS:.+]] = tessera.loss.cross_entropy_backward
  // CHECK-SAME: axis = 1
  // CHECK-SAME: ignore_index = -9
  // CHECK-SAME: label_smoothing = 1.500000e-01
  // CHECK: return %[[DLOGITS]]
}
