// RUN: tessera-opt --pass-pipeline='builtin.module(tessera-control-flow-to-scf,tessera-autodiff-paired)' %s | FileCheck %s

// W4-PRODUCT-1: a SAVE scan produces a compact interior-state tape, exposes it
// from the paired forward, accepts it in the backward ABI, and consumes it in
// the reverse loop instead of constructing the quadratic per-step replay loop.

module {
  func.func @step(%carry: tensor<4xf32>, %xt: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %next = "tessera.add"(%carry, %xt) :
        (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %next, %next : tensor<4xf32>, tensor<4xf32>
  }

  func.func @saved(%init: tensor<4xf32>, %xs: tensor<3x4xf32>)
      -> (tensor<4xf32>, tensor<3x4xf32>) attributes {
        tessera.autodiff = "reverse"} {
    %carry, %ys = "tessera.control_scan"(%init, %xs) {
      body = @step, carry_arg_index = 0 : i64, trip = 3 : i64,
      tessera.autodiff.checkpoint_policy = "save",
      tessera.autodiff.checkpoint_indices = array<i64: 1, 2>,
      tessera.autodiff.residual_schema = "tessera.region_residual_abi.v1",
      tessera.autodiff.residual_digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      tessera.structured_cfg.digest = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    } : (tensor<4xf32>, tensor<3x4xf32>) ->
        (tensor<4xf32>, tensor<3x4xf32>)
    return %carry, %ys : tensor<4xf32>, tensor<3x4xf32>
  }
}

// CHECK-LABEL: func.func @saved(
// CHECK-SAME: -> (tensor<4xf32>, tensor<3x4xf32>, tensor<2x4xf32>)
// CHECK-SAME: tessera.autodiff.residual_sources = ["control_scan:state_tape"]
// CHECK: %[[FWD:.+]]:3 = scf.for
// CHECK: } {tessera.autodiff.activity = "active", tessera.autodiff.checkpoint_indices = array<i64: 1, 2>
// CHECK-SAME: tessera.autodiff.residual_materialized = true
// CHECK-SAME: tessera.autodiff.residual_owner = "control_scan"
// CHECK-SAME: tessera.autodiff.residual_result_indices = array<i64: 2>
// CHECK: return %[[FWD]]#0, %[[FWD]]#1, %[[FWD]]#2

// CHECK-LABEL: func.func @saved__bwd(
// CHECK-SAME: %[[TAPE:[^:]+]]: tensor<2x4xf32>
// CHECK-SAME: -> (tensor<4xf32>, tensor<3x4xf32>)
// CHECK-SAME: tessera.autodiff.residual_sources = ["control_scan:state_tape"]
// CHECK-COUNT-2: scf.for
// CHECK: tensor.extract_slice %[[TAPE]]
// CHECK-NOT: scf.for
// CHECK: return
