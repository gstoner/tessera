// CF2 + CF0 composition: in a backend pipeline the control-flow→scf lowering
// runs BEFORE the control-flow target guard. A control_for that lowers to scf.for
// is gone before the guard runs (no diagnostic); a control_if (CF2b, not yet
// lowered) survives and is still caught loudly by the guard. This pins the
// intended pass ordering.
//
// RUN: tessera-opt %s -split-input-file -verify-diagnostics \
// RUN:   --tessera-control-flow-to-scf \
// RUN:   --tessera-control-flow-target-guard=target=rocm
//
// NOTE: the named GPU pipeline (`--tessera-lower-to-gpu`) is intentionally NOT
// exercised here anymore: as of the NVIDIA control-flow completion it lowers
// tessera.control_if to tile.control_if (CF2b) rather than leaving it for the
// target guard to reject, so the "control_if survives and is guarded" contract
// only holds for the explicit control-flow-to-scf pass above. GPU control_if
// lowering is covered by tests/tessera-ir/phase3/control_flow_tile_lowering.mlir.

// control_for lowers → no guard diagnostic (the function verifies clean).
func.func private @body(%c: tensor<1x8xf32>) -> tensor<1x8xf32>
func.func @for_lowers_clean(%init: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %r = "tessera.control_for"(%init) {
    body = @body, start = 0 : i64, stop = 8 : i64, step = 1 : i64
  } : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}

// -----

// control_if is not lowered by CF2 yet → the guard still rejects it.
func.func private @tb()
func.func private @eb()
func.func @if_still_guarded(%flag: tensor<1xf32>) -> tensor<1x8xf32> {
  // expected-error @+1 {{CONTROL_FLOW_UNSUPPORTED_ON_TARGET: 'tessera.control_if'}}
  %r = "tessera.control_if"(%flag) {
    then_branch = @tb, else_branch = @eb, flag_arg_index = 0 : i64
  } : (tensor<1xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}
