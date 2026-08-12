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
