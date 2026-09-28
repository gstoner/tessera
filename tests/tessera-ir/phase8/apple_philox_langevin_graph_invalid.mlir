// REQUIRES: tessera-apple-backend
// RUN: not tessera-opt %s --tessera-philox-langevin-to-apple_gpu 2>&1 | FileCheck %s

// CHECK: Apple Philox Langevin is outside the bounded Metal ABI
func.func @underflow(%y: tensor<8xf32>, %grad: tensor<8xf32>,
                     %seed: tensor<1xi64>, %counter: tensor<4xi64>)
                     -> tensor<8xf32> {
  %0 = tessera.ebm.langevin_step_philox %y, %grad, %seed, %counter
      {eta = 1.0e-99 : f64, temperature = 0.5 : f64}
      : (tensor<8xf32>, tensor<8xf32>, tensor<1xi64>, tensor<4xi64>)
      -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
