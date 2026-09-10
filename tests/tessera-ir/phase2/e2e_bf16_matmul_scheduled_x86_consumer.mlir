// REQUIRES: tessera-x86-target-ir
// RUN: tessera-opt --pass-pipeline='builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile,tessera-x86-executable{family=matmul input=tile output=target arch=x86_64_avx512})' %s | FileCheck %s
module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @bf16(%a: tensor<5x17xbf16>, %b: tensor<17x9xbf16>) -> tensor<5x9xf32> {
    %out = "tessera.matmul"(%a, %b) : (tensor<5x17xbf16>, tensor<17x9xbf16>) -> tensor<5x9xf32>
    return %out : tensor<5x9xf32>
  }
}
// CHECK: call @tessera_x86_avx512_gemm_bf16
// CHECK-NOT: tile.matmul_kernel
