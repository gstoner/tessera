// Stage 13 — RL policy-loss verifier diagnostics.
//
// RUN: %tessera_strict_opt %s --verify-diagnostics -o /dev/null

func.func @bad_reduction(%n: tensor<2x3xf32>, %o: tensor<2x3xf32>,
                         %a: tensor<2x3xf32>) -> tensor<f32> {
  // expected-error @+1 {{reduction must be one of "none", "mean", "sum"}}
  %0 = tessera.rl.ppo_policy_loss %n, %o, %a
      {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
       reduction = "median"}
      : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func.func @bad_shape(%n: tensor<2x3xf32>, %o: tensor<2x4xf32>,
                     %a: tensor<2x3xf32>) -> tensor<f32> {
  // expected-error @+1 {{ppo_policy_loss log-prob shapes must match}}
  %0 = tessera.rl.ppo_policy_loss %n, %o, %a
      {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>}
      : (tensor<2x3xf32>, tensor<2x4xf32>, tensor<2x3xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func.func @bad_kl_without_ref(%n: tensor<2x3xf32>, %o: tensor<2x3xf32>,
                              %a: tensor<2x3xf32>) -> tensor<f32> {
  // expected-error @+1 {{kl_coef requires ref_logp operand}}
  %0 = tessera.rl.ppo_policy_loss %n, %o, %a
      {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
       kl_coef = 1.000000e-02 : f64}
      : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func.func @bad_entropy_coef_without_entropy(
    %n: tensor<2x3xf32>, %o: tensor<2x3xf32>,
    %a: tensor<2x3xf32>) -> tensor<f32> {
  // expected-error @+1 {{entropy_coef requires entropy operand}}
  %0 = tessera.rl.ppo_policy_loss %n, %o, %a
      {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
       entropy_coef = 1.000000e-02 : f64}
      : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func.func @bad_group_axis(%n: tensor<2x3xf32>, %o: tensor<2x3xf32>,
                          %r: tensor<2x3xf32>) -> tensor<f32> {
  // expected-error @+1 {{group_axis must be within the operand rank}}
  %0 = tessera.rl.grpo_policy_loss %n, %o, %r {group_axis = 7 : i64}
      : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}
