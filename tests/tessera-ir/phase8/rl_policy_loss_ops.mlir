// Stage 13 — registered RL policy-loss Graph IR ops plus the compiler-visible
// decomposition pass.
//
// RUN: %tessera_strict_opt %s -tessera-rl-loss-decompose | FileCheck %s

// CHECK-LABEL: func.func @ppo
// CHECK: tessera.rl.ppo_policy_loss {{.*}}tessera.rl.compiler_decomposed_reference = true{{.*}}tessera.rl.compiler_visible = true{{.*}}tessera.rl.variant = "ppo"
func.func @ppo(%n: tensor<2x3x5xf32>, %o: tensor<2x3x5xf32>,
               %a: tensor<2x3x5xf32>) -> tensor<f32> {
  %0 = tessera.rl.ppo_policy_loss %n, %o, %a
      {clip_epsilon = 2.000000e-01 : f64, reduction = "mean"}
      : (tensor<2x3x5xf32>, tensor<2x3x5xf32>, tensor<2x3x5xf32>)
        -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func.func @grpo
// CHECK: tessera.rl.grpo_policy_loss {{.*}}tessera.rl.compiler_visible = true{{.*}}tessera.rl.decomposition_status = "compiler_visible_non_executable"{{.*}}tessera.rl.variant = "grpo"
func.func @grpo(%n: tensor<2x3x5xf32>, %o: tensor<2x3x5xf32>,
                %r: tensor<2x3x5xf32>) -> tensor<f32> {
  %0 = tessera.rl.grpo_policy_loss %n, %o, %r
      {group_axis = 1 : i64, reduction = "mean"}
      : (tensor<2x3x5xf32>, tensor<2x3x5xf32>, tensor<2x3x5xf32>)
        -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func.func @cispo
// CHECK: tessera.rl.cispo_policy_loss {{.*}}tessera.rl.compiler_visible = true{{.*}}tessera.rl.decomposition_status = "compiler_visible_non_executable"{{.*}}tessera.rl.variant = "cispo"
func.func @cispo(%n: tensor<2x3x5xf32>, %o: tensor<2x3x5xf32>,
                 %r: tensor<2x3x5xf32>) -> tensor<f32> {
  %0 = tessera.rl.cispo_policy_loss %n, %o, %r
      {group_axis = 1 : i64, epsilon_high = 5.000000e+00 : f64,
       reduction = "mean"}
      : (tensor<2x3x5xf32>, tensor<2x3x5xf32>, tensor<2x3x5xf32>)
        -> tensor<f32>
  return %0 : tensor<f32>
}
