// ODS-WIRE-2 negative cases: the GraphToSchedule consumer of
// `tessera.istft_jvp` refuses, with a registered diagnostic, every product it
// cannot bind to an exact profile and a consistent overlap-add geometry
// (Decision #21). It never falls back to a default profile.
//
// RUN: tessera-opt %s --split-input-file --tessera-graph-to-schedule -verify-diagnostics

// No exact profile: an Apple GPU module has no ISTFT JVP package.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @wrong_profile(%x: tensor<4x5xcomplex<f32>>, %w: tensor<8xf32>,
                           %dx: tensor<4x5xcomplex<f32>>, %dw: tensor<8xf32>)
      -> tensor<20xf32> {
    // expected-error @+1 {{SPECTRAL_JVP_SCHEDULE_REFUSED}}
    %t = "tessera.istft_jvp"(%x, %w, %dx, %dw) {hop = 4 : i64, logical_length = 8 : i64}
        : (tensor<4x5xcomplex<f32>>, tensor<8xf32>, tensor<4x5xcomplex<f32>>, tensor<8xf32>)
        -> tensor<20xf32>
    return %t : tensor<20xf32>
  }
}

// -----

// Both tangents are zero splats: there is no product to schedule.
module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @no_active_tangent(%x: tensor<4x5xcomplex<f32>>, %w: tensor<8xf32>)
      -> tensor<20xf32> {
    %dx = arith.constant dense<(0.000000e+00,0.000000e+00)> : tensor<4x5xcomplex<f32>>
    %dw = arith.constant dense<0.000000e+00> : tensor<8xf32>
    // expected-error @+1 {{has no active tangent}}
    %t = "tessera.istft_jvp"(%x, %w, %dx, %dw) {hop = 4 : i64, logical_length = 8 : i64}
        : (tensor<4x5xcomplex<f32>>, tensor<8xf32>, tensor<4x5xcomplex<f32>>, tensor<8xf32>)
        -> tensor<20xf32>
    return %t : tensor<20xf32>
  }
}

// -----

// A declared output_length that the static tangent type contradicts.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @length_mismatch(%x: tensor<4x5xcomplex<f32>>, %w: tensor<8xf32>,
                             %dx: tensor<4x5xcomplex<f32>>, %dw: tensor<8xf32>)
      -> tensor<18xf32> {
    // expected-error @+1 {{output_length disagrees with the static tangent type}}
    %t = "tessera.istft_jvp"(%x, %w, %dx, %dw) {
        hop = 4 : i64, logical_length = 8 : i64, output_length = 16 : i64}
        : (tensor<4x5xcomplex<f32>>, tensor<8xf32>, tensor<4x5xcomplex<f32>>, tensor<8xf32>)
        -> tensor<18xf32>
    return %t : tensor<18xf32>
  }
}

// -----

// A generic ROCm name inherits no chip proof: gfx1200 is refused.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1200"} {
  func.func @unproven_chip(%x: tensor<4x5xcomplex<f32>>, %w: tensor<8xf32>,
                           %dx: tensor<4x5xcomplex<f32>>, %dw: tensor<8xf32>)
      -> tensor<20xf32> {
    // expected-error @+1 {{requires an exact Zen 5 AVX-512, gfx1151, gfx1201 or sm120 profile}}
    %t = "tessera.istft_jvp"(%x, %w, %dx, %dw) {hop = 4 : i64, logical_length = 8 : i64}
        : (tensor<4x5xcomplex<f32>>, tensor<8xf32>, tensor<4x5xcomplex<f32>>, tensor<8xf32>)
        -> tensor<20xf32>
    return %t : tensor<20xf32>
  }
}
