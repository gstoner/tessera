// E2E-REAL-6 Apple GPU GELU: schedule.activation owns the apple7 contract.
// RUN: tessera-opt --split-input-file --verify-diagnostics %s

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @erf(%x: tensor<5x11xf32>) -> tensor<5x11xf32> {
    // expected-error @+1 {{requires static rank-two apple7 f16/bf16/f32 serial tanh GELU}}
    %0 = schedule.activation %x {accum = "f32", approximation = "erf", arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, kind = "gelu", nan_mode = "propagate", schedule = "serial", storage = "f32", workgroup_size = 1 : i64, zero_mode = "preserve"} : tensor<5x11xf32> -> tensor<5x11xf32>
    return %0 : tensor<5x11xf32>
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @missing_approximation(%x: tensor<5x11xf32>) -> tensor<5x11xf32> {
    // expected-error @+1 {{requires static rank-two apple7 f16/bf16/f32 serial tanh GELU}}
    %0 = schedule.activation %x {accum = "f32", arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, kind = "gelu", nan_mode = "propagate", schedule = "serial", storage = "f32", workgroup_size = 1 : i64, zero_mode = "preserve"} : tensor<5x11xf32> -> tensor<5x11xf32>
    return %0 : tensor<5x11xf32>
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @relu_on_apple(%x: tensor<5x11xf32>) -> tensor<5x11xf32> {
    // expected-error @+1 {{requires static rank-two apple7 f16/bf16/f32 serial tanh GELU}}
    %0 = schedule.activation %x {accum = "f32", approximation = "tanh", arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, kind = "relu", nan_mode = "propagate", schedule = "serial", storage = "f32", workgroup_size = 1 : i64, zero_mode = "preserve"} : tensor<5x11xf32> -> tensor<5x11xf32>
    return %0 : tensor<5x11xf32>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @approximation_on_sm120(%x: tensor<5x11xf32>) -> tensor<5x11xf32> {
    // expected-error @+1 {{approximation is an apple7 GELU attribute}}
    %0 = schedule.activation %x {accum = "f32", approximation = "tanh", arch = "sm_120", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, kind = "relu", nan_mode = "propagate", schedule = "serial", storage = "f32", workgroup_size = 128 : i64, zero_mode = "preserve"} : tensor<5x11xf32> -> tensor<5x11xf32>
    return %0 : tensor<5x11xf32>
  }
}
