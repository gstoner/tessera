// RUN: tessera-opt --pass-pipeline='builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile,tessera-x86-executable{family=es_low_rank_correction input=tile output=target arch=x86_64_avx512})' %s | FileCheck %s

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @es_rank1(
      %x: tensor<4x3x8xf32>, %members: tensor<4xi64>, %key: tensor<2xi64>)
      -> tensor<4x3x6xf32> {
    %correction = "tessera.es_low_rank_correction"(%x, %members, %key) {
      out_dim = 6 : i64, rank = 1 : i64, epoch = 3 : i64,
      sigma = 2.000000e-02 : f64, antithetic = true,
      numeric_policy = {storage = "fp32", accum = "fp32"}
    } : (tensor<4x3x8xf32>, tensor<4xi64>, tensor<2xi64>) -> tensor<4x3x6xf32>
    return %correction : tensor<4x3x6xf32>
  }
}

// CHECK: tessera.pipeline.family = "es_low_rank_correction"
// CHECK: tessera.pipeline.target_ir_consumer = "tessera_x86"
// CHECK: tessera_x86.abi_call
// CHECK-SAME: symbol = "tessera_x86_avx512_es_low_rank_correction_f32"
// CHECK: call @tessera_x86_avx512_es_low_rank_correction_f32
// CHECK-NOT: tile.es_low_rank_correction_kernel
// CHECK-NOT: schedule.es_low_rank_correction
// CHECK-NOT: tessera.es_low_rank_correction
// REQUIRES: tessera-x86-target-ir
