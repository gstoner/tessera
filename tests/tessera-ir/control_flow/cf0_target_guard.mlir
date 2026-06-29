// CF0 — control-flow target guard diagnostic (Decision #21).
// docs/spec/CONTROL_FLOW_CONTRACT.md §5: a tessera.control_* op on any backend
// without a control-flow lowering (everything but apple_gpu) must fail loudly
// with a stable CONTROL_FLOW_UNSUPPORTED_ON_TARGET diagnostic, never silently
// fall through to a host loop inside an executable-backend claim.
//
// RUN: tessera-opt %s -split-input-file -verify-diagnostics \
// RUN:   -tessera-control-flow-target-guard=target=rocm \
// RUN:   --allow-unregistered-dialect

func.func private @body(%c: tensor<1x8xf32>) -> tensor<1x8xf32>

func.func @for_on_rocm(%init: tensor<1x8xf32>) -> tensor<1x8xf32> {
  // expected-error @+1 {{CONTROL_FLOW_UNSUPPORTED_ON_TARGET: 'tessera.control_for' is not yet executable on target 'rocm'}}
  %r = "tessera.control_for"(%init) {
    body = @body, start = 0 : i64, stop = 8 : i64, step = 1 : i64
  } : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}

// -----

func.func private @wb(%c: tensor<1x4xf32>) -> tensor<1x4xf32>
func.func private @wc(%c: tensor<1x4xf32>) -> tensor<1x4xf32>

func.func @while_on_rocm(%init: tensor<1x4xf32>) -> tensor<1x4xf32> {
  // expected-error @+1 {{CONTROL_FLOW_UNSUPPORTED_ON_TARGET: 'tessera.control_while' is not yet executable on target 'rocm'}}
  %r = "tessera.control_while"(%init) {
    body = @wb, cond = @wc, carry_arg_index = 0 : i64, max_iters = 8 : i64
  } : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}

// -----

// A control-flow-free program passes the guard untouched (no diagnostic).
func.func @plain_matmul(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c = "tessera.matmul"(%a, %b) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}
