// RUN: tessera-opt --pass-pipeline='builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile,tessera-x86-executable{family=spectral_backward input=tile output=target arch=x86_64_avx512})' %s | FileCheck %s

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

// CHECK: tessera_x86.abi_call
// CHECK: call @tessera_x86_avx512_spectral_filter_bwd_c64
// CHECK-NOT: tile.spectral_backward_kernel
// CHECK-NOT: schedule.spectral_backward
