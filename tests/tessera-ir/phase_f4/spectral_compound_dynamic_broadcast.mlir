// RUN: tessera-opt %s | FileCheck %s

// A dynamic signal batch extent is compatible with a statically sized window
// batch.  The verifier can reject only a pair of unequal static extents; the
// runtime owns the concrete dynamic compatibility check.

module {
  // CHECK-LABEL: func.func @stft_dynamic_signal_batch
  // CHECK: tessera.stft
  func.func @stft_dynamic_signal_batch(
      %x: tensor<?x16x3xf32>, %window: tensor<4x1x8xf32>)
      -> tensor<?x3x5x3xcomplex<f32>> {
    %y = "tessera.stft"(%x, %window) {
      axis = 1 : i64, logical_length = 8 : i64, hop = 4 : i64,
      normalization = "backward",
      spectrum_layout = "half_spectrum_nyquist_explicit",
      center = false, onesided = true, pad_mode = "constant",
      window_broadcast = "batch"
    } : (tensor<?x16x3xf32>, tensor<4x1x8xf32>) ->
        tensor<?x3x5x3xcomplex<f32>>
    return %y : tensor<?x3x5x3xcomplex<f32>>
  }
}
