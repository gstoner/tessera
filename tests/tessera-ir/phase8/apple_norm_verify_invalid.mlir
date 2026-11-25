// APPLE-NTBI-2 slice 9: schedule.norm owns the apple7 f16/bf16 serial row-norm
// contract; every departure from it is refused.
// RUN: tessera-opt --split-input-file --verify-diagnostics %s

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @f32_storage(%x: tensor<5x11xf32>) -> tensor<5x11xf32> {
    // expected-error @+1 {{requires SM120, Zen 5, apple7 f16/bf16 serial}}
    %0 = schedule.norm %x {accum = "f32", arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, epsilon = 1.0e-05 : f32, kind = "rmsnorm", storage = "f32", workgroup_size = 1 : i64} : tensor<5x11xf32> -> tensor<5x11xf32>
    return %0 : tensor<5x11xf32>
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @wide_workgroup(%x: tensor<5x11xf16>) -> tensor<5x11xf16> {
    // expected-error @+1 {{requires SM120, Zen 5, apple7 f16/bf16 serial}}
    %0 = schedule.norm %x {accum = "f32", arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, epsilon = 1.0e-05 : f32, kind = "rmsnorm", storage = "f16", workgroup_size = 128 : i64} : tensor<5x11xf16> -> tensor<5x11xf16>
    return %0 : tensor<5x11xf16>
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @cooperative(%x: tensor<5x11xf16>) -> tensor<5x11xf16> {
    // expected-error @+1 {{requires SM120, Zen 5, apple7 f16/bf16 serial}}
    %0 = schedule.norm %x {accum = "f32", arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, epsilon = 1.0e-05 : f32, kind = "rmsnorm", schedule = "cooperative_128", storage = "f16", workgroup_size = 1 : i64} : tensor<5x11xf16> -> tensor<5x11xf16>
    return %0 : tensor<5x11xf16>
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @rank_three(%x: tensor<2x5x11xf16>) -> tensor<2x5x11xf16> {
    // expected-error @+1 {{requires SM120, Zen 5, apple7 f16/bf16 serial}}
    %0 = schedule.norm %x {accum = "f32", arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, epsilon = 1.0e-05 : f32, kind = "rmsnorm", storage = "f16", workgroup_size = 1 : i64} : tensor<2x5x11xf16> -> tensor<2x5x11xf16>
    return %0 : tensor<2x5x11xf16>
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @int32_extent(%x: tensor<65536x32768xf16>) -> tensor<65536x32768xf16> {
    // expected-error @+1 {{requires SM120, Zen 5, apple7 f16/bf16 serial}}
    %0 = schedule.norm %x {accum = "f32", arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, epsilon = 1.0e-05 : f32, kind = "rmsnorm", storage = "f16", workgroup_size = 1 : i64} : tensor<65536x32768xf16> -> tensor<65536x32768xf16>
    return %0 : tensor<65536x32768xf16>
  }
}
