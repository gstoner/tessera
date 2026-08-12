// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --generate-wmma-gemm-kernel --lower-tile-to-rocm='arch=gfx1151' %s | FileCheck %s

module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @rocm_scheduled_matmul(
      %a: tensor<17x19xf16>, %b: tensor<19x23xf16>)
      -> tensor<17x23xf32> {
    %0 = tessera.matmul %a, %b
        : (tensor<17x19xf16>, tensor<19x23xf16>) -> tensor<17x23xf32>
    return %0 : tensor<17x23xf32>
  }
}

// CHECK-NOT: tessera.matmul
// CHECK-NOT: schedule.
// CHECK-NOT: tile.matmul_kernel
// CHECK: gpu.func @rocm_scheduled_matmul
// CHECK: tessera_rocm.wmma
