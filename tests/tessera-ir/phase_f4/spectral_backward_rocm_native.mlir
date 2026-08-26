// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --lower-tile-to-rocm='arch=gfx1151' --generate-rocm-spectral-backward-kernel %s | FileCheck %s

module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @conv_backward(
      %dy: tensor<7xf32>, %x: tensor<5xf32>, %filter: tensor<3xf32>)
      -> (tensor<5xf32>, tensor<3xf32>) {
    %dx, %df = "tessera.spectral_backward"(%dy, %x, %filter) {
      kind = "tessera.spectral_conv", axis = -1 : i64,
      logical_length = 8 : i64, normalization = "ortho",
      spectrum_layout = "half_spectrum_nyquist_explicit"
    } : (tensor<7xf32>, tensor<5xf32>, tensor<3xf32>) ->
        (tensor<5xf32>, tensor<3xf32>)
    return %dx, %df : tensor<5xf32>, tensor<3xf32>
  }
}

// CHECK: gpu.module @conv_backward_mod
// CHECK: gpu.func @conv_backward
// CHECK: scf.for
// CHECK-NOT: tessera_rocm.spectral_backward
// CHECK-NOT: tile.spectral_backward_kernel

// -----

module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @stft_backward_centered_axis1(
      %dy: tensor<2x7x10x3xcomplex<f32>>, %x: tensor<2x46x3xf32>,
      %window: tensor<18xf32>) -> (tensor<2x46x3xf32>, tensor<18xf32>) {
    %dx, %dw = "tessera.spectral_backward"(%dy, %x, %window) {
      kind = "tessera.stft", axis = 1 : i64,
      logical_length = 18 : i64, hop = 7 : i64,
      center = true, onesided = true, pad_mode = "reflect",
      normalization = "backward",
      spectrum_layout = "half_spectrum_nyquist_explicit"
    } : (tensor<2x7x10x3xcomplex<f32>>, tensor<2x46x3xf32>, tensor<18xf32>) ->
        (tensor<2x46x3xf32>, tensor<18xf32>)
    return %dx, %dw : tensor<2x46x3xf32>, tensor<18xf32>
  }
}

// CHECK: gpu.module @stft_backward_centered_axis1_mod
// CHECK: gpu.func @stft_backward_centered_axis1
// CHECK: arith.divui
// CHECK: arith.remui
// CHECK-NOT: tessera_rocm.spectral_backward
// CHECK-NOT: tile.spectral_backward_kernel

// -----

module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @istft_backward_centered_axis2(
      %dy: tensor<2x40x3xf32>, %spectrum: tensor<2x7x10x3xcomplex<f32>>,
      %window: tensor<18xf32>) ->
      (tensor<2x7x10x3xcomplex<f32>>, tensor<18xf32>) {
    %ds, %dw = "tessera.spectral_backward"(%dy, %spectrum, %window) {
      kind = "tessera.istft", axis = 2 : i64,
      logical_length = 18 : i64, hop = 7 : i64,
      output_length = 40 : i64, center = true, onesided = true,
      pad_mode = "constant", normalization = "backward",
      spectrum_layout = "half_spectrum_nyquist_explicit"
    } : (tensor<2x40x3xf32>, tensor<2x7x10x3xcomplex<f32>>, tensor<18xf32>) ->
        (tensor<2x7x10x3xcomplex<f32>>, tensor<18xf32>)
    return %ds, %dw : tensor<2x7x10x3xcomplex<f32>>, tensor<18xf32>
  }
}

// CHECK: gpu.module @istft_backward_centered_axis2_mod
// CHECK: gpu.func @istft_backward_centered_axis2
// CHECK: arith.divui
// CHECK: arith.remui
// CHECK-NOT: tessera_rocm.spectral_backward
// CHECK-NOT: tile.spectral_backward_kernel

// -----

module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @stft_backward_centered_reflect(
      %dy: tensor<2x7x10xcomplex<f32>>, %x: tensor<2x46xf32>,
      %window: tensor<18xf32>) -> (tensor<2x46xf32>, tensor<18xf32>) {
    %dx, %dw = "tessera.spectral_backward"(%dy, %x, %window) {
      kind = "tessera.stft", axis = -1 : i64,
      logical_length = 18 : i64, hop = 7 : i64,
      center = true, onesided = true, pad_mode = "reflect",
      normalization = "backward",
      spectrum_layout = "half_spectrum_nyquist_explicit"
    } : (tensor<2x7x10xcomplex<f32>>, tensor<2x46xf32>, tensor<18xf32>) ->
        (tensor<2x46xf32>, tensor<18xf32>)
    return %dx, %dw : tensor<2x46xf32>, tensor<18xf32>
  }
}

// CHECK: gpu.module @stft_backward_centered_reflect_mod
// CHECK: gpu.func @stft_backward_centered_reflect
// CHECK: arith.select
// CHECK: math.cos
// CHECK: math.sin
// CHECK-NOT: tessera_rocm.spectral_backward
// CHECK-NOT: tile.spectral_backward_kernel

// -----

module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @istft_backward_centered_crop(
      %dy: tensor<2x40xf32>, %spectrum: tensor<2x7x10xcomplex<f32>>,
      %window: tensor<18xf32>) ->
      (tensor<2x7x10xcomplex<f32>>, tensor<18xf32>) {
    %ds, %dw = "tessera.spectral_backward"(%dy, %spectrum, %window) {
      kind = "tessera.istft", axis = -1 : i64,
      logical_length = 18 : i64, hop = 7 : i64,
      output_length = 40 : i64, center = true, onesided = true,
      pad_mode = "constant", normalization = "backward",
      spectrum_layout = "half_spectrum_nyquist_explicit"
    } : (tensor<2x40xf32>, tensor<2x7x10xcomplex<f32>>, tensor<18xf32>) ->
        (tensor<2x7x10xcomplex<f32>>, tensor<18xf32>)
    return %ds, %dw : tensor<2x7x10xcomplex<f32>>, tensor<18xf32>
  }
}

// CHECK: gpu.module @istft_backward_centered_crop_mod
// CHECK: gpu.func @istft_backward_centered_crop
// CHECK: scf.if
// CHECK: math.cos
// CHECK: math.sin
// CHECK-NOT: tessera_rocm.spectral_backward
// CHECK-NOT: tile.spectral_backward_kernel

// -----

module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @stft_backward(
      %dy: tensor<3x5x10xcomplex<f32>>, %x: tensor<3x46xf16>,
      %window: tensor<18xf16>) -> (tensor<3x46xf16>, tensor<18xf16>) {
    %dx, %dw = "tessera.spectral_backward"(%dy, %x, %window) {
      kind = "tessera.stft", axis = -1 : i64,
      logical_length = 18 : i64, hop = 7 : i64,
      center = false, onesided = true, pad_mode = "constant",
      normalization = "backward",
      spectrum_layout = "half_spectrum_nyquist_explicit",
      numeric_policy = {storage = "fp16", accum = "fp32"}
    } : (tensor<3x5x10xcomplex<f32>>, tensor<3x46xf16>, tensor<18xf16>) ->
        (tensor<3x46xf16>, tensor<18xf16>)
    return %dx, %dw : tensor<3x46xf16>, tensor<18xf16>
  }
}

// CHECK: gpu.module @stft_backward_mod
// CHECK: gpu.func @stft_backward
// CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: memref<?xf16>
// CHECK: math.cos
// CHECK: math.sin
// CHECK-NOT: tessera_rocm.spectral_backward
// CHECK-NOT: tile.spectral_backward_kernel

// -----

module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @istft_backward(
      %dy: tensor<3x46xbf16>, %spectrum: tensor<3x5x10xcomplex<f32>>,
      %window: tensor<18xbf16>) ->
      (tensor<3x5x10xcomplex<f32>>, tensor<18xbf16>) {
    %ds, %dw = "tessera.spectral_backward"(%dy, %spectrum, %window) {
      kind = "tessera.istft", axis = -1 : i64,
      logical_length = 18 : i64, hop = 7 : i64,
      output_length = 46 : i64, center = false, onesided = true,
      pad_mode = "constant", normalization = "backward",
      spectrum_layout = "half_spectrum_nyquist_explicit",
      numeric_policy = {storage = "bf16", accum = "fp32"}
    } : (tensor<3x46xbf16>, tensor<3x5x10xcomplex<f32>>, tensor<18xbf16>) ->
        (tensor<3x5x10xcomplex<f32>>, tensor<18xbf16>)
    return %ds, %dw : tensor<3x5x10xcomplex<f32>>, tensor<18xbf16>
  }
}

// CHECK: gpu.module @istft_backward_mod
// CHECK: gpu.func @istft_backward
// CHECK-SAME: %arg0: memref<?xbf16>, %arg1: memref<?xf32>, %arg2: memref<?xbf16>, %arg3: memref<?xf32>, %arg4: memref<?xbf16>
// CHECK: math.cos
// CHECK: math.sin
// CHECK-NOT: tessera_rocm.spectral_backward
// CHECK-NOT: tile.spectral_backward_kernel
