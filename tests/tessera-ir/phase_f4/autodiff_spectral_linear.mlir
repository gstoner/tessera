// RUN: tessera-opt %s --tessera-autodiff-paired | FileCheck %s

// AD-TSOL-SPECTRAL-1: spectral transposes are compiler-owned Graph IR. The
// semantic contract survives into the backward function; no Python callback
// or unregistered string pass recreates the transform.

module {
  func.func @fft_backward(%x: tensor<8xcomplex<f32>>)
      -> tensor<8xcomplex<f32>> attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.fft"(%x) {
      axis = -1 : i64, logical_length = 8 : i64,
      normalization = "backward", spectrum_layout = "full_complex",
      hermitian_weight = "none"
    } : (tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>>
    return %y : tensor<8xcomplex<f32>>
  }

  // CHECK-LABEL: func.func @fft_backward__bwd
  // CHECK: tessera.ifft
  // CHECK-SAME: logical_length = 8
  // CHECK-SAME: normalization = "forward"

  func.func @rfft_backward(%x: tensor<8xf32>)
      -> tensor<5xcomplex<f32>> attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.rfft"(%x) {
      axis = -1 : i64, logical_length = 8 : i64,
      normalization = "backward",
      spectrum_layout = "half_spectrum_nyquist_explicit",
      hermitian_weight = "none"
    } : (tensor<8xf32>) -> tensor<5xcomplex<f32>>
    return %y : tensor<5xcomplex<f32>>
  }

  // CHECK-LABEL: func.func @rfft_backward__bwd
  // CHECK: tessera.irfft
  // CHECK-SAME: hermitian_weight = "half_interior"
  // CHECK-SAME: logical_length = 8
  // CHECK-SAME: normalization = "forward"

  func.func @irfft_backward(%x: tensor<5xcomplex<f32>>)
      -> tensor<8xf32> attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.irfft"(%x) {
      axis = -1 : i64, logical_length = 8 : i64,
      normalization = "backward",
      spectrum_layout = "half_spectrum_nyquist_explicit",
      hermitian_weight = "none"
    } : (tensor<5xcomplex<f32>>) -> tensor<8xf32>
    return %y : tensor<8xf32>
  }

  // CHECK-LABEL: func.func @irfft_backward__bwd
  // CHECK: tessera.rfft
  // CHECK-SAME: hermitian_weight = "double_interior"
  // CHECK-SAME: logical_length = 8
  // CHECK-SAME: normalization = "forward"

  func.func @dct_backward(%x: tensor<8xf32>)
      -> tensor<8xf32> attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.dct"(%x) {
      axis = -1 : i64, logical_length = 8 : i64,
      normalization = "ortho", spectrum_layout = "full_real",
      hermitian_weight = "none", type = 3 : i64
    } : (tensor<8xf32>) -> tensor<8xf32>
    return %y : tensor<8xf32>
  }

  // CHECK-LABEL: func.func @dct_backward__bwd
  // CHECK: tessera.dct {{.*}}normalization = "ortho"{{.*}}transpose_basis = true{{.*}}type = 3

  func.func @spectral_filter_backward(
      %x: tensor<8xcomplex<f32>>, %filter: tensor<8xcomplex<f32>>)
      -> tensor<8xcomplex<f32>> attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.spectral_filter"(%x, %filter) {
      axis = -1 : i64, logical_length = 8 : i64,
      normalization = "backward",
      spectrum_layout = "full_complex"
    } : (tensor<8xcomplex<f32>>, tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>>
    return %y : tensor<8xcomplex<f32>>
  }

  // CHECK-LABEL: func.func @spectral_filter_backward__bwd
  // CHECK: tessera.spectral_backward
  // CHECK-SAME: kind = "tessera.spectral_filter"
  // CHECK-SAME: logical_length = 8

  func.func @stft_backward(%x: tensor<16xf32>, %window: tensor<8xf32>)
      -> tensor<3x5xcomplex<f32>> attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.stft"(%x, %window) {
      axis = -1 : i64, logical_length = 8 : i64, hop = 4 : i64,
      normalization = "ortho",
      spectrum_layout = "half_spectrum_nyquist_explicit",
      center = false, onesided = true
    } : (tensor<16xf32>, tensor<8xf32>) -> tensor<3x5xcomplex<f32>>
    return %y : tensor<3x5xcomplex<f32>>
  }

  // CHECK-LABEL: func.func @stft_backward__bwd
  // CHECK: tessera.spectral_backward
  // CHECK-SAME: hop = 4
  // CHECK-SAME: kind = "tessera.stft"
  // CHECK-SAME: normalization = "ortho"

  func.func @istft_backward(
      %x: tensor<3x5xcomplex<f32>>, %window: tensor<8xf32>)
      -> tensor<16xf32> attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.istft"(%x, %window) {
      axis = 1 : i64, logical_length = 8 : i64, hop = 4 : i64,
      normalization = "backward",
      spectrum_layout = "half_spectrum_nyquist_explicit",
      center = false, onesided = true
    } : (tensor<3x5xcomplex<f32>>, tensor<8xf32>) -> tensor<16xf32>
    return %y : tensor<16xf32>
  }

  // CHECK-LABEL: func.func @istft_backward__bwd
  // CHECK: tessera.spectral_backward
  // CHECK-SAME: kind = "tessera.istft"
  // CHECK-SAME: logical_length = 8

  func.func @spectral_conv_backward(%x: tensor<5xf32>, %filter: tensor<3xf32>)
      -> tensor<7xf32> attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.spectral_conv"(%x, %filter) {
      axis = -1 : i64, logical_length = 8 : i64,
      normalization = "backward",
      spectrum_layout = "half_spectrum_nyquist_explicit"
    } : (tensor<5xf32>, tensor<3xf32>) -> tensor<7xf32>
    return %y : tensor<7xf32>
  }

  // CHECK-LABEL: func.func @spectral_conv_backward__bwd
  // CHECK: tessera.spectral_backward
  // CHECK-SAME: kind = "tessera.spectral_conv"
  // CHECK-SAME: logical_length = 8
}
