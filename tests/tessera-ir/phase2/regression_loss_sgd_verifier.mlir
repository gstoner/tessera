// RUN: tessera-opt %s -split-input-file -verify-diagnostics

module {
  func.func @valid(%prediction: tensor<?x5xf32>,
                   %target: tensor<?x5xf32>) -> tensor<f32> {
    %loss = "tessera.loss.huber"(%prediction, %target)
        {delta = 1.5 : f64} :
        (tensor<?x5xf32>, tensor<?x5xf32>) -> tensor<f32>
    return %loss : tensor<f32>
  }
}

// -----

module {
  func.func @bad_huber_delta(%a: tensor<3xf32>, %b: tensor<3xf32>)
      -> tensor<f32> {
    // expected-error @+1 {{requires delta > 0}}
    %loss = "tessera.loss.huber"(%a, %b) {delta = 0.0 : f64} :
        (tensor<3xf32>, tensor<3xf32>) -> tensor<f32>
    return %loss : tensor<f32>
  }
}

// -----

module {
  func.func @bad_smooth_beta(%a: tensor<3xf32>, %b: tensor<3xf32>)
      -> tensor<f32> {
    // expected-error @+1 {{requires beta > 0}}
    %loss = "tessera.loss.smooth_l1"(%a, %b) {beta = -1.0 : f64} :
        (tensor<3xf32>, tensor<3xf32>) -> tensor<f32>
    return %loss : tensor<f32>
  }
}

// -----

module {
  func.func @bad_sgd_shape(%p: tensor<3xf32>, %g: tensor<4xf32>)
      -> tensor<3xf32> {
    // expected-error @+1 {{sgd parameter/gradient shapes must match}}
    %updated = "tessera.sgd"(%p, %g) {lr = 0.1 : f64} :
        (tensor<3xf32>, tensor<4xf32>) -> tensor<3xf32>
    return %updated : tensor<3xf32>
  }
}
