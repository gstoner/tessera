// RUN: tessera-opt --pass-pipeline='builtin.module(tessera-control-flow-to-scf,tessera-autodiff-paired)' %s | FileCheck %s

// W4.3: HYBRID materializes only selected checkpoints, carries the compact
// tape through the paired ABI, and replays from the nearest retained state.

module {
  func.func @step(%carry: tensor<4xf32>, %xt: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %next = "tessera.add"(%carry, %xt) :
        (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %next, %next : tensor<4xf32>, tensor<4xf32>
  }

  func.func @hybrid(%init: tensor<4xf32>, %xs: tensor<3x4xf32>)
      -> (tensor<4xf32>, tensor<3x4xf32>) attributes {
        tessera.autodiff = "reverse"} {
    %carry, %ys = "tessera.control_scan"(%init, %xs) {
      body = @step, carry_arg_index = 0 : i64, trip = 3 : i64,
      tessera.autodiff.checkpoint_policy = "hybrid",
      tessera.autodiff.checkpoint_indices = array<i64: 1>,
      tessera.autodiff.residual_schema = "tessera.region_residual_abi.v1",
      tessera.autodiff.residual_digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      tessera.structured_cfg.digest = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    } : (tensor<4xf32>, tensor<3x4xf32>) ->
        (tensor<4xf32>, tensor<3x4xf32>)
    return %carry, %ys : tensor<4xf32>, tensor<3x4xf32>
  }
}

// CHECK-LABEL: func.func @hybrid(
// CHECK-SAME: -> (tensor<4xf32>, tensor<3x4xf32>, tensor<1x4xf32>)
// CHECK-SAME: tessera.autodiff.residual_policy = "hybrid"
// CHECK-SAME: tessera.autodiff.residual_sources = ["control_scan:state_tape"]
//
// CHECK-LABEL: func.func @hybrid__bwd(
// CHECK-SAME: %[[TAPE:[^:]+]]: tensor<1x4xf32>
// CHECK-SAME: tessera.autodiff.residual_policy = "hybrid"
// CHECK: arith.cmpi ule
// CHECK: tensor.extract_slice %[[TAPE]]
// CHECK: scf.for {{.*}} to
// CHECK: return
