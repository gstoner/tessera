// RUN: tessera-opt %s --tessera-autodiff-forward | FileCheck %s

module {
  func.func @dropout_replay(%x: tensor<8xf32>) -> tensor<8xf32>
      attributes {tessera.autodiff = "forward"} {
    %y = "tessera.dropout"(%x) {
      p = 2.500000e-01 : f64, seed = 7 : i64, counter_base = 13 : i64,
      estimator = "constant_noise", rng_algorithm = "philox4x32-10",
      rng_version = 1 : i64, tessera.effect_kind = "random", training = true
    } : (tensor<8xf32>) -> tensor<8xf32>
    return %y : tensor<8xf32>
  }

  func.func @eggroll_x_only(
      %x: tensor<4x3x8xf32>, %members: tensor<4xi64>, %key: tensor<2xi64>)
      -> tensor<4x3x6xf32>
      attributes {tessera.autodiff = "forward",
                  tessera.autodiff.wrt_indices = [0]} {
    %y = "tessera.es_low_rank_correction"(%x, %members, %key) {
      out_dim = 6 : i64, rank = 1 : i64, epoch = 3 : i64,
      sigma = 2.000000e-02 : f64, antithetic = true,
      numeric_policy = {storage = "fp32", accum = "fp32"}
    } : (tensor<4x3x8xf32>, tensor<4xi64>, tensor<2xi64>) -> tensor<4x3x6xf32>
    return %y : tensor<4x3x6xf32>
  }
}

// CHECK-LABEL: func.func private @dropout_replay__jvp(
// CHECK-COUNT-2: tessera.dropout
// CHECK-SAME: counter_base = 13
// CHECK-SAME: seed = 7
// CHECK: return
// CHECK-LABEL: func.func private @eggroll_x_only__jvp(
// CHECK: %{{.*}}: tensor<4x3x8xf32>, %{{.*}}: tensor<4xi64>, %{{.*}}: tensor<2xi64>, %{{.*}}: tensor<4x3x8xf32>
// CHECK-COUNT-2: tessera.es_low_rank_correction
// CHECK: return
