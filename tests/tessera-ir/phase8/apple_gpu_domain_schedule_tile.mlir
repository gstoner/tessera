// E2E-REAL-6 Apple GPU domain value families (2026-11-04): an opt-in Apple
// GPU module (launch bindings) records its PPO / EBM / Clifford op as a
// content-addressed Schedule contract, lowers it to the registered Tile value
// op with every semantic value the delegate reads written explicitly, and the
// Apple GPU value lowering turns that into one verified
// tessera_apple.gpu.kernel_call carrying the schedule hash
// (NativeAppleGPUDomain.h).
// REQUIRES: tessera-apple-backend
// RUN: tessera-opt --tessera-graph-to-schedule --split-input-file %s | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s --check-prefix=TILE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --tile-to-apple_gpu=value=true --split-input-file %s | FileCheck %s --check-prefix=TARGET

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["new", "old", "adv", "ref", "loss"]} {
  func.func @ppo_ref(%n: tensor<2x3xf32>, %o: tensor<2x3xf32>, %a: tensor<2x3xf32>, %r: tensor<2x3xf32>) -> tensor<f32> {
    %0 = tessera.rl.ppo_policy_loss %n, %o, %a, %r {kl_coef = 5.000000e-02, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}

// The side tensor is read from its ODS segment (ref_logp, not mask), and the
// absent clip/entropy/reduction values are materialized at the delegate's.
// SCHEDULE-LABEL: func.func @ppo_ref
// SCHEDULE: tessera.rl.ppo_policy_loss{{.*}}schedule.artifact_hash = "[[HASH:[0-9a-f]+]]"
// SCHEDULE: schedule.artifact {
// SCHEDULE-SAME: abi = "mpsgraph_ppo_policy_loss_ex_f32"
// SCHEDULE-SAME: scalars = {N = 6 : i64}
// SCHEDULE-SAME: semantics = {clip_epsilon = 2.000000e-01 : f64, entropy_coef = 0.000000e+00 : f64, has_entropy = false, has_mask = false, has_ref_kl = true, kl_coef = 5.000000e-02 : f64, reduction = "mean"}
// SCHEDULE-SAME: symbol = "tessera_apple_gpu_ppo_policy_loss_ex_f32"
// SCHEDULE-SAME: hash = "[[HASH]]"
// SCHEDULE-SAME: shape_key = "family=apple_gpu_domain;kind=ppo_policy_loss"

// TILE-LABEL: func.func @ppo_ref
// TILE-NOT: tessera.rl.ppo_policy_loss
// TILE-NOT: schedule.
// TILE: tile.ppo_policy_loss
// TILE-SAME: has_mask = false, has_ref_kl = true
// TILE-SAME: source = "tessera.rl.ppo_policy_loss"
// TILE-SAME: tessera.apple_gpu_domain_contract =

// TARGET-LABEL: func.func @ppo_ref
// TARGET-NOT: tile.
// TARGET: tessera_apple.gpu.kernel_call
// TARGET-SAME: abi = "mpsgraph_ppo_policy_loss_ex_f32"
// TARGET-SAME: has_mask = false, has_ref_kl = true
// TARGET-SAME: op_kind = "ppo_policy_loss"
// TARGET-SAME: symbol = "tessera_apple_gpu_ppo_policy_loss_ex_f32"
// TARGET-SAME: tessera.schedule_hash = "{{[0-9a-f]+}}"

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["x", "y", "e"]} {
  func.func @energy(%x: tensor<4x5xf32>, %y: tensor<4x5xf32>) -> tensor<4xf32> {
    %0 = tessera.ebm.energy_quadratic %x, %y {tessera.effect_kind = "pure"} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// SCHEDULE-LABEL: func.func @energy
// SCHEDULE: schedule.artifact {
// SCHEDULE-SAME: scalars = {B = 4 : i64, D = 5 : i64}
// SCHEDULE-SAME: shape_key = "family=apple_gpu_domain;kind=ebm_energy_quadratic"
// TILE-LABEL: func.func @energy
// TILE: tile.ebm_energy_quadratic
// TARGET-LABEL: func.func @energy
// TARGET: tessera_apple.gpu.kernel_call
// TARGET-SAME: symbol = "tessera_apple_gpu_ebm_energy_quadratic_value_f32"

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["y", "g", "z", "out"]} {
  func.func @langevin(%y: tensor<3x4xf32>, %g: tensor<3x4xf32>, %z: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %0 = tessera.ebm.langevin_step %y, %g, %z {eta = 1.250000e-01} : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}

// SCHEDULE-LABEL: func.func @langevin
// SCHEDULE: semantics = {eta = 1.250000e-01 : f64, has_noise = true, noise_scale = 0.000000e+00 : f64}
// TILE-LABEL: func.func @langevin
// TILE: tile.ebm_langevin_step
// TILE-SAME: has_noise = true, noise_scale = 0.000000e+00 : f64
// TARGET-LABEL: func.func @langevin
// TARGET: tessera_apple.gpu.kernel_call
// TARGET-SAME: eta = 1.250000e-01 : f64
// TARGET-SAME: noise_scale = 0.000000e+00 : f64
// TARGET-SAME: symbol = "tessera_apple_gpu_ebm_langevin_step_value_f32"

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["y", "g", "out"]} {
  func.func @refine(%y: tensor<6xf32>, %g: tensor<6xf32>) -> tensor<6xf32> {
    %0 = tessera.ebm.refinement %y, %g {eta = 2.500000e-01, steps = 3} : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }
}

// SCHEDULE-LABEL: func.func @refine
// SCHEDULE: semantics = {eta = 2.500000e-01 : f64, steps = 3 : i64}
// TARGET-LABEL: func.func @refine
// TARGET: tessera_apple.gpu.kernel_call
// TARGET-SAME: steps = 3 : i64
// TARGET-SAME: symbol = "tessera_apple_gpu_ebm_refinement_value_f32"

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["energies", "z"]} {
  func.func @partition(%e: tensor<7xf32>) -> tensor<f32> {
    %0 = tessera.ebm.partition_exact %e : (tensor<7xf32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}

// SCHEDULE-LABEL: func.func @partition
// SCHEDULE: semantics = {reduction = "logsumexp", temperature = 1.000000e+00 : f64}
// TARGET-LABEL: func.func @partition
// TARGET: tessera_apple.gpu.kernel_call
// TARGET-SAME: reduction = "logsumexp"
// TARGET-SAME: temperature = 1.000000e+00 : f64

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["a", "b", "c"]} {
  func.func @clifford(%a: tensor<2x3x8xf32>, %b: tensor<2x3x8xf32>) -> tensor<2x3x8xf32> {
    %0 = tessera.clifford.geometric_product %a, %b {p = 3, q = 0} : (tensor<2x3x8xf32>, tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
    return %0 : tensor<2x3x8xf32>
  }
}

// SCHEDULE-LABEL: func.func @clifford
// SCHEDULE: scalars = {Batch = 6 : i64}
// SCHEDULE-SAME: semantics = {coefficient_layout = "blade_last", p = 3 : i64, q = 0 : i64}
// TARGET-LABEL: func.func @clifford
// TARGET: tessera_apple.gpu.kernel_call
// TARGET-SAME: coefficient_layout = "blade_last"
// TARGET-SAME: p = 3 : i64, q = 0 : i64
// TARGET-SAME: symbol = "tessera_apple_gpu_clifford_geo_product_cl30_value_f32"

// -----

// Without launch bindings the module is not opted in: the op passes
// Graph -> Schedule untouched.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @not_owned(%x: tensor<4x5xf32>, %y: tensor<4x5xf32>) -> tensor<4xf32> {
    %0 = tessera.ebm.energy_quadratic %x, %y : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// SCHEDULE-LABEL: func.func @not_owned
// SCHEDULE-NOT: schedule.artifact
// SCHEDULE: tessera.ebm.energy_quadratic
// SCHEDULE-NOT: schedule.artifact_hash
