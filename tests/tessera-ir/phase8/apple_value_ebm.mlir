// EBM Value Target IR — strict fp32 static Apple GPU envelopes.
//
// REQUIRES: tessera-apple-backend
//
// RUN: %tessera_strict_opt %s -tessera-lower-to-apple_gpu-full | FileCheck %s

// CHECK-LABEL: func.func @ebm_energy_quadratic_value
// CHECK: tessera_apple.gpu.kernel_call
// CHECK-SAME: abi = "msl_ebm_energy_quadratic_value_f32"
// CHECK-SAME: framework = "Metal"
// CHECK-SAME: op_kind = "ebm_energy_quadratic"
// CHECK-SAME: status = "executable"
// CHECK-SAME: symbol = "tessera_apple_gpu_ebm_energy_quadratic_value_f32"
// CHECK-SAME: : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2xf32>
// CHECK-NOT: tile.ebm_energy_quadratic
func.func @ebm_energy_quadratic_value(%x: tensor<2x3xf32>,
                                      %y: tensor<2x3xf32>) -> tensor<2xf32> {
  %0 = tessera.ebm.energy_quadratic %x, %y
    : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func.func @ebm_langevin_step_value
// CHECK: tessera_apple.gpu.kernel_call
// CHECK-SAME: abi = "msl_ebm_langevin_step_value_f32"
// CHECK-SAME: eta = 1.250000e-01
// CHECK-SAME: framework = "Metal"
// CHECK-SAME: noise_scale = 2.500000e-01
// CHECK-SAME: op_kind = "ebm_langevin_step"
// CHECK-SAME: status = "executable"
// CHECK-SAME: symbol = "tessera_apple_gpu_ebm_langevin_step_value_f32"
// CHECK-SAME: : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NOT: tile.ebm_langevin_step
func.func @ebm_langevin_step_value(%y: tensor<2x3xf32>,
                                   %g: tensor<2x3xf32>,
                                   %n: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = tessera.ebm.langevin_step %y, %g, %n
    {eta = 1.250000e-01 : f64, noise_scale = 2.500000e-01 : f64}
    : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @ebm_refinement_value
// CHECK: tessera_apple.gpu.kernel_call
// CHECK-SAME: abi = "msl_ebm_refinement_value_f32"
// CHECK-SAME: eta = 1.250000e-01
// CHECK-SAME: framework = "Metal"
// CHECK-SAME: op_kind = "ebm_refinement"
// CHECK-SAME: status = "executable"
// CHECK-SAME: steps = 4
// CHECK-SAME: symbol = "tessera_apple_gpu_ebm_refinement_value_f32"
// CHECK-SAME: : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NOT: tile.ebm_refinement
func.func @ebm_refinement_value(%y: tensor<2x3xf32>,
                                %g: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = tessera.ebm.refinement %y, %g
    {eta = 1.250000e-01 : f64, steps = 4 : i64}
    : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @ebm_partition_exact_value
// CHECK: tessera_apple.gpu.kernel_call
// CHECK-SAME: abi = "msl_ebm_partition_exact_value_f32"
// CHECK-SAME: framework = "Metal"
// CHECK-SAME: op_kind = "ebm_partition_exact"
// CHECK-SAME: reduction = "logsumexp"
// CHECK-SAME: status = "executable"
// CHECK-SAME: symbol = "tessera_apple_gpu_ebm_partition_exact_value_f32"
// CHECK-SAME: temperature = 7.500000e-01
// CHECK-SAME: : (tensor<2x3xf32>) -> tensor<f32>
// CHECK-NOT: tile.ebm_partition_exact
func.func @ebm_partition_exact_value(%e: tensor<2x3xf32>) -> tensor<f32> {
  %0 = tessera.ebm.partition_exact %e
    {temperature = 7.500000e-01 : f64, reduction = "logsumexp"}
    : (tensor<2x3xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}
