// Stage 12 — standalone Apple value Tile verifier rejects opaque tile.* ops.
//
// RUN: %tessera_strict_opt %s -tessera-verify-apple-value-tile-ir --verify-diagnostics -o /dev/null

func.func @opaque_value_tile(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{apple value Tile IR contains unregistered op 'tile.fake_value_op'}}
  %0 = "tile.fake_value_op"(%a) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
