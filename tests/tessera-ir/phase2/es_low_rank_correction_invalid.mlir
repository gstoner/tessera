// RUN: tessera-opt --verify-diagnostics %s

func.func @rank_bucket_is_fail_closed(
    %x: tensor<8x16xf32>, %members: tensor<8xi64>, %key: tensor<2xi64>)
    -> tensor<8x24xf32> {
  // expected-error @+1 {{initial physical bucket requires rank = 1}}
  %result = "tessera.es_low_rank_correction"(%x, %members, %key) {
    out_dim = 24 : i64, rank = 2 : i64, sigma = 3.000000e-02 : f64,
    numeric_policy = {storage = "fp32", accum = "fp32"}
  } : (tensor<8x16xf32>, tensor<8xi64>, tensor<2xi64>) -> tensor<8x24xf32>
  return %result : tensor<8x24xf32>
}

func.func @accum_is_load_bearing(
    %x: tensor<8x16xf32>, %members: tensor<8xi64>, %key: tensor<2xi64>)
    -> tensor<8x24xf32> {
  // expected-error @+1 {{numeric_policy.accum must be fp32}}
  %result = "tessera.es_low_rank_correction"(%x, %members, %key) {
    out_dim = 24 : i64, rank = 1 : i64, sigma = 3.000000e-02 : f64,
    numeric_policy = {storage = "fp32", accum = "bf16"}
  } : (tensor<8x16xf32>, tensor<8xi64>, tensor<2xi64>) -> tensor<8x24xf32>
  return %result : tensor<8x24xf32>
}

func.func @population_identity_must_match(
    %x: tensor<8x16xf32>, %members: tensor<7xi64>, %key: tensor<2xi64>)
    -> tensor<8x24xf32> {
  // expected-error @+1 {{population extent P must match member_ids length}}
  %result = "tessera.es_low_rank_correction"(%x, %members, %key) {
    out_dim = 24 : i64, rank = 1 : i64, sigma = 3.000000e-02 : f64,
    numeric_policy = {storage = "fp32", accum = "fp32"}
  } : (tensor<8x16xf32>, tensor<7xi64>, tensor<2xi64>) -> tensor<8x24xf32>
  return %result : tensor<8x24xf32>
}
