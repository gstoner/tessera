// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --lower-tile-to-rocm='arch=gfx1151' --generate-rocm-es-low-rank-kernel %s | FileCheck %s

module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
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

// CHECK: gpu.module @es_rank1_mod
// CHECK: gpu.func @es_rank1
// CHECK: scf.for
// CHECK: gpu.shuffle xor
// CHECK: gpu.barrier
// CHECK: math.log
// CHECK: math.sqrt
// CHECK: math.cos
// CHECK: math.sin
// CHECK-NOT: tessera_rocm.es_low_rank_correction
// CHECK-NOT: tile.es_low_rank_correction_kernel
