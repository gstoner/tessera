// REQUIRES: tessera-x86-target-ir
// RUN: tessera-opt --pass-pipeline='builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile,tessera-x86-executable{family=matmul input=tile output=target arch=x86_64_avx512})' %s | FileCheck %s
module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @f64(%a: tensor<5x17xf64>, %b: tensor<17x9xf64>) -> tensor<5x9xf64> {
    %out = "tessera.matmul"(%a, %b) : (tensor<5x17xf64>, tensor<17x9xf64>) -> tensor<5x9xf64>
    return %out : tensor<5x9xf64>
  }
}
// CHECK: call @tessera_x86_avx512_gemm_f64
// CHECK-NOT: tile.matmul_kernel
