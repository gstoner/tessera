// REQUIRES: tessera-apple-backend
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu-runtime)' | FileCheck %s

// CHECK-DAG: func.func private @tessera_apple_gpu_ebm_langevin_step_philox_graph_f32_status(i64, i64, i64, i64, f32, f32, i64, i32) -> i32
// CHECK-LABEL: func.func @philox_langevin
// CHECK-NOT: tessera.ebm.langevin_step_philox
// CHECK: call @tessera_apple_gpu_ebm_langevin_step_philox_graph_f32_status
// CHECK: cf.assert
func.func @philox_langevin(%y: tensor<2x8xf32>, %grad: tensor<2x8xf32>,
                           %seed: tensor<1xi64>, %counter: tensor<4xi64>)
                           -> tensor<2x8xf32> {
  %0 = tessera.ebm.langevin_step_philox %y, %grad, %seed, %counter
      {eta = 0.125 : f64, temperature = 0.5 : f64}
      : (tensor<2x8xf32>, tensor<2x8xf32>, tensor<1xi64>, tensor<4xi64>)
      -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}
