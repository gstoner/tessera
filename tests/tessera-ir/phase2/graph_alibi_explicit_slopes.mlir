// RUN: tessera-opt --split-input-file --verify-diagnostics %s

module {
  func.func @wrong_slope_extent(%slopes: tensor<3xf32>) -> tensor<4x7x7xf32> {
    // expected-error @+1 {{slopes must be rank-1 f32 with num_heads elements}}
    %bias = tessera.alibi %slopes {num_heads = 4 : i64, seq_len = 7 : i64} : (tensor<3xf32>) -> tensor<4x7x7xf32>
    return %bias : tensor<4x7x7xf32>
  }
}

// -----

module {
  func.func @wrong_bias_heads(%slopes: tensor<4xf32>) -> tensor<3x7x7xf32> {
    // expected-error @+1 {{explicit slopes require an f32 [num_heads, seq_len, seq_len] bias}}
    %bias = tessera.alibi %slopes {num_heads = 4 : i64, seq_len = 7 : i64} : (tensor<4xf32>) -> tensor<3x7x7xf32>
    return %bias : tensor<3x7x7xf32>
  }
}

// -----

module {
  func.func @wrong_slope_storage(%slopes: tensor<4xf16>) -> tensor<4x7x7xf32> {
    // expected-error @+1 {{slopes must be rank-1 f32 with num_heads elements}}
    %bias = tessera.alibi %slopes {num_heads = 4 : i64, seq_len = 7 : i64} : (tensor<4xf16>) -> tensor<4x7x7xf32>
    return %bias : tensor<4x7x7xf32>
  }
}
