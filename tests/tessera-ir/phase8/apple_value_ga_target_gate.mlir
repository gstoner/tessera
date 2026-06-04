// Stage 16E — only clifford_geometric_product has a promoted Apple GPU value
// executor. Other registered GA/Clifford value-seam IR remains gated.
//
// RUN: %tessera_strict_opt %s -tessera-lower-to-apple_gpu-full --verify-diagnostics -o /dev/null

func.func @clifford_value_target_gate(%a: tensor<2x8xf32>,
                                      %b: tensor<2x8xf32>) -> tensor<2x8xf32> {
  // expected-error @+1 {{apple_gpu value lowering: no value-producing GPU target op for 'tessera.clifford.outer_product'}}
  %0 = tessera.clifford.outer_product %a, %b
    {p = 3 : i64, q = 0 : i64}
    : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}
