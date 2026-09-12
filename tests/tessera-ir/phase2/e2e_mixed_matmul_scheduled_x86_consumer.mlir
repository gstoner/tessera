// REQUIRES: tessera-x86-target-ir
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile %s | FileCheck %s --check-prefix=TILE
// RUN: tessera-opt --pass-pipeline='builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile,tessera-x86-executable{family=matmul input=tile output=target arch=x86_64_avx512})' %s | FileCheck %s --check-prefix=TARGET
module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @mixed(%a: tensor<3x17xui8>, %b: tensor<17x19xi8>) -> tensor<3x19xi32> {
    %out = "tessera.matmul"(%a, %b) : (tensor<3x17xui8>, tensor<17x19xi8>) -> tensor<3x19xi32>
    return %out : tensor<3x19xi32>
  }
}
// TILE: tile.matmul_kernel
// TILE-SAME: a = "u8", b = "i8", acc = "i32"
// TARGET: call @tessera_x86_avx512_vnni_gemm_u8s8_s32
// TARGET-NOT: tile.matmul_kernel
