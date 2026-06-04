// Stage 16C — GA/Clifford can cross Graph→Tile as registered value-seam IR, but
// Apple value Target execution is still gated until a dedicated GA value
// executor is promoted.
//
// RUN: %tessera_strict_opt %s -tessera-lower-to-apple_gpu-full --verify-diagnostics -o /dev/null

func.func @clifford_value_target_gate(%a: tensor<2x8xf32>,
                                      %b: tensor<2x8xf32>) -> tensor<2x8xf32> {
  // expected-error @+1 {{apple_gpu value lowering: no value-producing GPU target op for 'tessera.clifford.geometric_product'}}
  %0 = tessera.clifford.geometric_product %a, %b
    {p = 3 : i64, q = 0 : i64}
    : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}
