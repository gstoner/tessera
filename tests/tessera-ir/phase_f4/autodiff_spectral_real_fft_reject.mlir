// RUN: not %tessera_opt %s --tessera-autodiff-paired 2>&1 | FileCheck %s

// A full FFT may accept a real forward input, but it is not complex-linear in
// that input: its transpose must project a full-spectrum cotangent back to the
// real domain. Callers that want compiler spectral autodiff must spell this as
// rfft so the packed Hermitian endpoint/interior policy is part of the IR.

module {
  func.func @real_full_fft(%x: tensor<8xf32>)
      -> tensor<8xcomplex<f32>> attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.fft"(%x) {
      axis = -1 : i64, logical_length = 8 : i64,
      normalization = "backward", spectrum_layout = "full_complex",
      hermitian_weight = "none"
    } : (tensor<8xf32>) -> tensor<8xcomplex<f32>>
    return %y : tensor<8xcomplex<f32>>
  }
}

// CHECK: error: real-input full FFT has no complex-linear transpose; use rfft so Hermitian weighting is explicit
