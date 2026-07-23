// RUN: tessera-opt %s -split-input-file -verify-diagnostics

module {
  func.func @valid(%logits: tensor<?x11x?xf32>,
                   %target: tensor<?x?xi64>) -> tensor<f32> {
    %loss = "tessera.loss.cross_entropy"(%logits, %target)
        {axis = 1 : i64, ignore_index = -7 : i64,
         label_smoothing = 2.0e-1 : f64, reduction = "mean"} :
        (tensor<?x11x?xf32>, tensor<?x?xi64>) -> tensor<f32>
    return %loss : tensor<f32>
  }
}

// -----

module {
  func.func @bad_target_shape(%logits: tensor<3x5x7xf32>,
                              %target: tensor<3x5xi64>) -> tensor<f32> {
    // expected-error @+1 {{target shape must equal logits shape with class axis removed}}
    %loss = "tessera.loss.cross_entropy"(%logits, %target)
        {axis = 1 : i64} :
        (tensor<3x5x7xf32>, tensor<3x5xi64>) -> tensor<f32>
    return %loss : tensor<f32>
  }
}

// -----

module {
  func.func @bad_smoothing(%logits: tensor<3x5xf32>,
                           %target: tensor<3xi32>) -> tensor<f32> {
    // expected-error @+1 {{label_smoothing must be in [0, 1)}}
    %loss = "tessera.loss.cross_entropy"(%logits, %target)
        {label_smoothing = 1.0 : f64} :
        (tensor<3x5xf32>, tensor<3xi32>) -> tensor<f32>
    return %loss : tensor<f32>
  }
}
