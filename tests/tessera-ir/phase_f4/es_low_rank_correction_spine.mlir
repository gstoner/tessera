// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile %s | FileCheck %s

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

// CHECK: tile.es_low_rank_correction_kernel
// CHECK-SAME: accum = "f32"
// CHECK-SAME: arch = "gfx1151"
// CHECK-SAME: epoch = 3
// CHECK-SAME: in_dim = 8
// CHECK-SAME: out_dim = 6
// CHECK-SAME: population = 4
// CHECK-SAME: rank = 1
// CHECK-SAME: rng_algorithm = "splitmix64-philox4x32-boxmuller"
// CHECK-SAME: rng_version = 1
// CHECK-SAME: rows_per_member = 3
// CHECK-SAME: tessera.schedule_hash = "[[HASH:[0-9a-f]{64}]]"
// CHECK-NOT: schedule.es_low_rank_correction
// CHECK-NOT: tessera.es_low_rank_correction
