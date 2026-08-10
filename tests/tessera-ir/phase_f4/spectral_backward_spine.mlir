// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s --check-prefixes=X86,ROCM

// AD-TSOL-SPECTRAL-1: compound spectral VJPs retain one content-addressed,
// multi-output artifact boundary and is consumed by bounded architecture-owned
// packages on Zen 5 AVX-512 and gfx1151.

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @filter_backward(
      %dy: tensor<8xcomplex<f32>>, %x: tensor<8xcomplex<f32>>,
      %filter: tensor<8xcomplex<f32>>)
      -> (tensor<8xcomplex<f32>>, tensor<8xcomplex<f32>>) {
    %dx, %df = "tessera.spectral_backward"(%dy, %x, %filter) {
      kind = "tessera.spectral_filter", axis = -1 : i64,
      logical_length = 8 : i64, normalization = "backward",
      spectrum_layout = "full_complex"
    } : (tensor<8xcomplex<f32>>, tensor<8xcomplex<f32>>,
         tensor<8xcomplex<f32>>) ->
        (tensor<8xcomplex<f32>>, tensor<8xcomplex<f32>>)
    return %dx, %df : tensor<8xcomplex<f32>>, tensor<8xcomplex<f32>>
  }
}

// X86: tile.spectral_backward_kernel
// X86-SAME: arch = "zen5-avx512"
// X86-SAME: input_count = 3
// X86-SAME: kind = "tessera.spectral_filter"
// X86-SAME: mutation_lineage = "inputs_immutable_outputs_fresh_v1"
// X86-SAME: output_count = 2
// X86-SAME: tessera.schedule_hash = "[[X86_HASH:[0-9a-f]{64}]]"
// X86-NOT: schedule.spectral_backward
// X86-NOT: tessera.spectral_backward

// -----

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

// ROCM: tile.spectral_backward_kernel
// ROCM-SAME: arch = "gfx1151"
// ROCM-SAME: input_count = 3
// ROCM-SAME: kind = "tessera.spectral_conv"
// ROCM-SAME: normalization = "ortho"
// ROCM-SAME: output_count = 2
// ROCM-SAME: tessera.schedule_hash = "[[ROCM_HASH:[0-9a-f]{64}]]"
// ROCM-NOT: schedule.spectral_backward
// ROCM-NOT: tessera.spectral_backward
