// RUN: not tessera-opt --tessera-control-flow-to-scf %s 2>&1 | FileCheck %s

module {
  func.func @step(%carry: tensor<4xf32>, %xt: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    return %carry, %xt : tensor<4xf32>, tensor<4xf32>
  }

  func.func @unsupported_save(%init: tensor<4xf32>, %xs: tensor<3x4xf32>)
      -> (tensor<4xf32>, tensor<3x4xf32>) {
    %carry, %ys = "tessera.control_scan"(%init, %xs) {
      body = @step, carry_arg_index = 0 : i64, trip = 3 : i64,
      tessera.autodiff.checkpoint_policy = "save"
    } : (tensor<4xf32>, tensor<3x4xf32>) ->
        (tensor<4xf32>, tensor<3x4xf32>)
    return %carry, %ys : tensor<4xf32>, tensor<3x4xf32>
  }
}

// CHECK: save control region requires residual_schema=v1, CFG/residual SHA-256
// CHECK-SAME: digests, and explicit checkpoint_indices
