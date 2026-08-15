// Experiment 2 (c): the §3.2 ODS claim — "a pipeline_advance need not consume a
// pipeline_state". The ODS says Variadic<AnyType>, but TileOps.cpp:1604 has a
// hand-written verifier. This file should be REJECTED if the verifier closes
// the hole (expected: "first operand must be the prior !tile.pipeline_state").
func.func @advance_without_state(%x: tensor<4xf32>) {
  %next = tile.pipeline_advance %x : (tensor<4xf32>) -> !tile.pipeline_state
  return
}
