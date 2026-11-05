// E2E-REAL-6 Apple GPU domain value families (2026-11-04): an opted-in Apple
// GPU module fails closed on any semantic the MSL / MPSGraph delegate does not
// implement, instead of letting the delegate compute a different result
// (NativeAppleGPUDomain.h, Decision #21a).
// RUN: not tessera-opt --tessera-graph-to-schedule --split-input-file %s 2>&1 | FileCheck %s

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["y", "g", "z", "out"]} {
  func.func @langevin_temperature(%y: tensor<3x4xf32>, %g: tensor<3x4xf32>, %z: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %0 = tessera.ebm.langevin_step %y, %g, %z {eta = 1.250000e-01, temperature = 2.000000e+00} : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}

// CHECK: error: Apple GPU native domain kernel: ebm_langevin_step does not implement temperature

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["y", "g", "out"]} {
  func.func @langevin_without_noise(%y: tensor<3x4xf32>, %g: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %0 = tessera.ebm.langevin_step %y, %g {eta = 1.250000e-01} : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}

// CHECK: error: Apple GPU native domain kernel: requires an isolated entry

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["y", "g", "out"]} {
  func.func @noisy_refinement(%y: tensor<6xf32>, %g: tensor<6xf32>) -> tensor<6xf32> {
    %0 = tessera.ebm.refinement %y, %g {eta = 2.500000e-01, steps = 3, temperature = 0.000000e+00} : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }
}

// CHECK: error: Apple GPU native domain kernel: ebm_refinement implements the deterministic loop only

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["energies", "z"]} {
  func.func @partition_sum(%e: tensor<7xf32>) -> tensor<f32> {
    %0 = tessera.ebm.partition_exact %e {reduction = "sum"} : (tensor<7xf32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}

// CHECK: error: Apple GPU native domain kernel: ebm_partition_exact implements reduction = "logsumexp" only

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["a", "b", "c"]} {
  func.func @clifford_blade_major(%a: tensor<2x8xf32>, %b: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %0 = tessera.clifford.geometric_product %a, %b {p = 3, q = 0, coefficient_layout = "blade_major"} : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
}

// CHECK: error: Apple GPU native domain kernel: clifford_geometric_product implements coefficient_layout = "blade_last" only

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["a", "b", "c"]} {
  func.func @clifford_signature(%a: tensor<2x8xf32>, %b: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %0 = tessera.clifford.geometric_product %a, %b {p = 3, q = 0, signature = [1, -1, 1]} : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
}

// CHECK: error: Apple GPU native domain kernel: clifford_geometric_product implements signature [1,1,1] only

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["new", "old", "adv", "loss"]} {
  func.func @ppo_sum(%n: tensor<2x3xf32>, %o: tensor<2x3xf32>, %a: tensor<2x3xf32>) -> tensor<f32> {
    %0 = tessera.rl.ppo_policy_loss %n, %o, %a {reduction = "sum", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>} : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}

// CHECK: error: Apple GPU native domain kernel: ppo_policy_loss implements reduction = "mean" only

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["x", "y", "e"]} {
  func.func @energy_f16(%x: tensor<4x5xf16>, %y: tensor<4x5xf16>) -> tensor<4xf16> {
    %0 = tessera.ebm.energy_quadratic %x, %y : (tensor<4x5xf16>, tensor<4x5xf16>) -> tensor<4xf16>
    return %0 : tensor<4xf16>
  }
}

// CHECK: error: Apple GPU native domain kernel: requires static non-empty f32 operands within the int32 ABI

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["x", "y", "e"]} {
  func.func @energy_unknown_attr(%x: tensor<4x5xf32>, %y: tensor<4x5xf32>) -> tensor<4xf32> {
    %0 = tessera.ebm.energy_quadratic %x, %y {scale = 2.000000e+00} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK: error: Apple GPU native domain kernel: has an unsupported policy attribute 'scale'

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple8", tessera.launch_bindings = ["x", "y", "e"]} {
  func.func @wrong_arch(%x: tensor<4x5xf32>, %y: tensor<4x5xf32>) -> tensor<4xf32> {
    %0 = tessera.ebm.energy_quadratic %x, %y : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK: error: Apple GPU native domain kernel: requires the apple7 Apple GPU target
