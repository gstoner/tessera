// CF2 in the NAMED backend pipelines users actually invoke (not a hand-spelled
// pass sequence): tessera-lower-to-gpu runs LowerControlFlowToSCF before the
// CF0 guard. A lowerable control_for becomes scf.for and sails past the guard;
// the executable-payload form (which CF2 can't lower without decoding the
// payload) is skipped and still rejected by the guard.

// RUN: tessera-opt %s --tessera-lower-to-gpu --allow-unregistered-dialect | FileCheck %s

// A simple (non-payload) control_for lowers to scf.for inside the named GPU
// pipeline — no CONTROL_FLOW_UNSUPPORTED_ON_TARGET diagnostic.
func.func private @b(%c: tensor<1x8xf32>) -> tensor<1x8xf32>
// CHECK-LABEL: func.func @lowers_in_named_pipeline
// CHECK: scf.for
// CHECK-NOT: tessera.control_for
func.func @lowers_in_named_pipeline(%init: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %r = "tessera.control_for"(%init) {
    body = @b, start = 0 : i64, stop = 8 : i64, step = 1 : i64
  } : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}
