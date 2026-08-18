// RUN: tessera-opt --tessera-control-flow-to-scf %s | FileCheck %s

// W4-PRODUCT-1: SAVE/HYBRID selection crosses the Graph -> SCF boundary as
// one lossless, digest-bound contract.  This is deliberately stricter than a
// policy marker: lowering must reject a missing CFG identity or a checkpoint
// set whose cardinality contradicts the policy.

module {
  func.func @step(%carry: tensor<4xf32>, %xt: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %next = "tessera.add"(%carry, %xt) :
        (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %next, %next : tensor<4xf32>, tensor<4xf32>
  }

  func.func @saved(%init: tensor<4xf32>, %xs: tensor<3x4xf32>)
      -> (tensor<4xf32>, tensor<3x4xf32>) {
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

  func.func @hybrid(%init: tensor<4xf32>, %xs: tensor<3x4xf32>)
      -> (tensor<4xf32>, tensor<3x4xf32>) {
    %carry, %ys = "tessera.control_scan"(%init, %xs) {
      body = @step, carry_arg_index = 0 : i64, trip = 3 : i64,
      tessera.autodiff.checkpoint_policy = "hybrid",
      tessera.autodiff.checkpoint_indices = array<i64: 1>,
      tessera.autodiff.residual_schema = "tessera.region_residual_abi.v1",
      tessera.autodiff.residual_digest = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      tessera.structured_cfg.digest = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
    } : (tensor<4xf32>, tensor<3x4xf32>) ->
        (tensor<4xf32>, tensor<3x4xf32>)
    return %carry, %ys : tensor<4xf32>, tensor<3x4xf32>
  }
}

// CHECK-LABEL: func.func @saved
// CHECK: scf.for
// CHECK: } {tessera.autodiff.checkpoint_indices = array<i64: 1, 2>
// CHECK-SAME: tessera.autodiff.checkpoint_policy = "save"
// CHECK-SAME: tessera.autodiff.residual_digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
// CHECK-SAME: tessera.autodiff.residual_schema = "tessera.region_residual_abi.v1"
// CHECK-SAME: tessera.structured_cfg.digest = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
// CHECK-NOT: tessera.control_scan

// CHECK-LABEL: func.func @hybrid
// CHECK: scf.for
// CHECK: } {tessera.autodiff.checkpoint_indices = array<i64: 1>
// CHECK-SAME: tessera.autodiff.checkpoint_policy = "hybrid"
// CHECK-NOT: tessera.control_scan
