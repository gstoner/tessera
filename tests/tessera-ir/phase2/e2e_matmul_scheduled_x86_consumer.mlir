// RUN: tessera-opt --pass-pipeline='builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile,tessera-x86-executable{family=matmul input=tile output=target arch=x86_64_avx512})' %s | FileCheck %s

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @x86_scheduled_matmul(
      %a: tensor<17x19xf32>, %b: tensor<19x23xf32>)
      -> tensor<17x23xf32> {
    %0 = tessera.matmul %a, %b
        : (tensor<17x19xf32>, tensor<19x23xf32>) -> tensor<17x23xf32>
    return %0 : tensor<17x23xf32>
  }
}

// CHECK-LABEL: func.func @x86_scheduled_matmul
// CHECK-NOT: tessera.matmul
// CHECK-NOT: schedule.
// CHECK-NOT: tile.matmul_kernel
// CHECK: call @tessera_x86_avx512_gemm_f32
