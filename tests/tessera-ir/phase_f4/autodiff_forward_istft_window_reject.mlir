// RUN: not tessera-opt %s --tessera-autodiff-forward 2>&1 | FileCheck %s

module {
  func.func @istft_active_window(
      %x: tensor<3x5xcomplex<f32>>, %window: tensor<8xf32>)
      -> tensor<16xf32> attributes {tessera.autodiff = "forward"} {
    %y = "tessera.istft"(%x, %window) {
      axis = -1 : i64, logical_length = 8 : i64, hop = 4 : i64,
      normalization = "backward",
      spectrum_layout = "half_spectrum_nyquist_explicit"
    } : (tensor<3x5xcomplex<f32>>, tensor<8xf32>) -> tensor<16xf32>
    return %y : tensor<16xf32>
  }
}

// CHECK: TangentInterface rejected the active operand combination
