// Stage 13/14 — PPO policy loss lowers to the strict or extended Apple GPU
// value executor depending on optional mask/ref/entropy operands.
//
// RUN: %tessera_strict_opt %s -tessera-lower-to-apple_gpu-full | FileCheck %s

// CHECK-LABEL: func.func @ppo_value
// CHECK: tessera_apple.gpu.kernel_call {{.*}}framework = "MPSGraph"{{.*}}op_kind = "ppo_policy_loss"{{.*}}status = "executable"{{.*}}symbol = "tessera_apple_gpu_ppo_policy_loss_f32"
// CHECK-NOT: tile.ppo_policy_loss
// CHECK-NOT: ub.poison
func.func @ppo_value(%n: tensor<2x3x5xf32>, %o: tensor<2x3x5xf32>,
                     %a: tensor<2x3x5xf32>) -> tensor<f32> {
  %0 = tessera.rl.ppo_policy_loss %n, %o, %a
      {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
       clip_epsilon = 2.000000e-01 : f64, reduction = "mean"}
      : (tensor<2x3x5xf32>, tensor<2x3x5xf32>, tensor<2x3x5xf32>)
        -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func.func @ppo_value_full
// CHECK: tessera_apple.gpu.kernel_call {{.*}}abi = "mpsgraph_ppo_policy_loss_ex_f32"{{.*}}has_entropy = true{{.*}}has_mask = true{{.*}}has_ref_kl = true{{.*}}op_kind = "ppo_policy_loss"{{.*}}symbol = "tessera_apple_gpu_ppo_policy_loss_ex_f32"
// CHECK-NOT: tile.ppo_policy_loss
// CHECK-NOT: ub.poison
func.func @ppo_value_full(%n: tensor<2x3x5xf32>, %o: tensor<2x3x5xf32>,
                          %a: tensor<2x3x5xf32>, %m: tensor<2x3x5xf32>,
                          %r: tensor<2x3x5xf32>, %e: tensor<2x3x5xf32>)
    -> tensor<f32> {
  %0 = tessera.rl.ppo_policy_loss %n, %o, %a, %m, %r, %e
      {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>,
       clip_epsilon = 2.000000e-01 : f64, kl_coef = 1.000000e-02 : f64,
       entropy_coef = 2.000000e-02 : f64, reduction = "mean"}
      : (tensor<2x3x5xf32>, tensor<2x3x5xf32>, tensor<2x3x5xf32>,
         tensor<2x3x5xf32>, tensor<2x3x5xf32>, tensor<2x3x5xf32>)
        -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func.func @ppo_value_ref_only
// CHECK: tessera_apple.gpu.kernel_call {{.*}}has_entropy = false{{.*}}has_mask = false{{.*}}has_ref_kl = true{{.*}}symbol = "tessera_apple_gpu_ppo_policy_loss_ex_f32"
func.func @ppo_value_ref_only(%n: tensor<2x3x5xf32>, %o: tensor<2x3x5xf32>,
                              %a: tensor<2x3x5xf32>,
                              %r: tensor<2x3x5xf32>) -> tensor<f32> {
  %0 = tessera.rl.ppo_policy_loss %n, %o, %a, %r
      {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>,
       clip_epsilon = 2.000000e-01 : f64, kl_coef = 1.000000e-02 : f64,
       reduction = "mean"}
      : (tensor<2x3x5xf32>, tensor<2x3x5xf32>, tensor<2x3x5xf32>,
         tensor<2x3x5xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}
