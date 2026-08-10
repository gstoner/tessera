// RUN: tessera-opt %s | FileCheck %s

// EGGROLL W2 Graph contract: the member-keyed correction is one operation and
// carries every semantic needed to reconstruct its factors physically.
// CHECK-LABEL: func.func @rank1_population
// CHECK: tessera.es_low_rank_correction
// CHECK-SAME: epoch = 7
// CHECK-SAME: accum = "fp32"
// CHECK-SAME: out_dim = 24
// CHECK-SAME: rank = 1
func.func @rank1_population(
    %x: tensor<8x16xf32>, %members: tensor<8xi64>, %key: tensor<2xi64>)
    -> tensor<8x24xf32> {
  %result = "tessera.es_low_rank_correction"(%x, %members, %key) {
    out_dim = 24 : i64, rank = 1 : i64, epoch = 7 : i64,
    sigma = 3.000000e-02 : f64, score = "gaussian", antithetic = true,
    numeric_policy = {storage = "fp32", accum = "fp32"}
  } : (tensor<8x16xf32>, tensor<8xi64>, tensor<2xi64>) -> tensor<8x24xf32>
  return %result : tensor<8x24xf32>
}
