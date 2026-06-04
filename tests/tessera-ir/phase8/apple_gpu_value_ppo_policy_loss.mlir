// Stage 13 — PPO policy loss lowers to the narrow Apple GPU value executor.
//
// RUN: %tessera_strict_opt %s -tessera-lower-to-apple_gpu-full | FileCheck %s

// CHECK-LABEL: func.func @ppo_value
// CHECK: tessera_apple.gpu.kernel_call {{.*}}framework = "MPSGraph"{{.*}}op_kind = "ppo_policy_loss"{{.*}}status = "executable"{{.*}}symbol = "tessera_apple_gpu_ppo_policy_loss_f32"
// CHECK-NOT: tile.ppo_policy_loss
// CHECK-NOT: ub.poison
func.func @ppo_value(%n: tensor<2x3x5xf32>, %o: tensor<2x3x5xf32>,
                     %a: tensor<2x3x5xf32>) -> tensor<f32> {
  %0 = tessera.rl.ppo_policy_loss %n, %o, %a
      {clip_epsilon = 2.000000e-01 : f64, reduction = "mean"}
      : (tensor<2x3x5xf32>, tensor<2x3x5xf32>, tensor<2x3x5xf32>)
        -> tensor<f32>
  return %0 : tensor<f32>
}
