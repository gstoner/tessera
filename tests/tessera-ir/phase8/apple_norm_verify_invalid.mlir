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

// APPLE-NTBI-2 slice 10: the weighted form is apple7 only, names both flags,
// and only a layernorm carries a beta.

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @weighted_sm120(%x: tensor<5x11xf16>) -> tensor<5x11xf16> {
    // expected-error @+1 {{a weighted norm requires apple7 with affine = true}}
    %0 = schedule.norm %x {accum = "f32", affine = true, bias = false, arch = "sm_120", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, epsilon = 1.0e-05 : f32, kind = "rmsnorm", storage = "f16", workgroup_size = 128 : i64} : tensor<5x11xf16> -> tensor<5x11xf16>
    return %0 : tensor<5x11xf16>
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @beta_on_rmsnorm(%x: tensor<5x11xf16>) -> tensor<5x11xf16> {
    // expected-error @+1 {{a weighted norm requires apple7 with affine = true}}
    %0 = schedule.norm %x {accum = "f32", affine = true, bias = true, arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, epsilon = 1.0e-05 : f32, kind = "rmsnorm", storage = "f16", workgroup_size = 1 : i64} : tensor<5x11xf16> -> tensor<5x11xf16>
    return %0 : tensor<5x11xf16>
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @affine_without_bias(%x: tensor<5x11xf16>) -> tensor<5x11xf16> {
    // expected-error @+1 {{a weighted norm requires apple7 with affine = true}}
    %0 = schedule.norm %x {accum = "f32", affine = true, arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, epsilon = 1.0e-05 : f32, kind = "rmsnorm", storage = "f16", workgroup_size = 1 : i64} : tensor<5x11xf16> -> tensor<5x11xf16>
    return %0 : tensor<5x11xf16>
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @affine_false(%x: tensor<5x11xf16>) -> tensor<5x11xf16> {
    // expected-error @+1 {{a weighted norm requires apple7 with affine = true}}
    %0 = schedule.norm %x {accum = "f32", affine = false, bias = false, arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, epsilon = 1.0e-05 : f32, kind = "layernorm", storage = "f16", workgroup_size = 1 : i64} : tensor<5x11xf16> -> tensor<5x11xf16>
    return %0 : tensor<5x11xf16>
  }
}

// -----

// A gamma-only layernorm does not decode (layer_norm(x, beta=b) traces the same).
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @gamma_only_layernorm(%x: tensor<5x11xf16>) -> tensor<5x11xf16> {
    // expected-error @+1 {{a weighted norm requires apple7 with affine = true}}
    %0 = schedule.norm %x {accum = "f32", affine = true, arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, bias = false, epsilon = 1.0e-05 : f32, kind = "layernorm", storage = "f16", workgroup_size = 1 : i64} : tensor<5x11xf16> -> tensor<5x11xf16>
    return %0 : tensor<5x11xf16>
  }
}
