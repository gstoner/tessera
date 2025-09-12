// RUN: tessera-opt %s -tessera-solver-legalize -tessera-mixed-precision-schedule | FileCheck %s
func.func @spd_solve(%A: tensor<128x128xf32>, %b: tensor<128xf32>) -> tensor<128xf32> {
  // Convert to fp8 (per-tile scale) and factorize
  %Af8 = "tessera.quantize"(%A) : (tensor<128x128xf32>) -> tensor<128x128x!tessera.fp8.e4m3>
  %L   = "tessera.solver.potrf"(%Af8) { policy = #tessera.solver.precision<"fp8:e4m3","f32","f32","per_tile"> } : (tensor<128x128x!tessera.fp8.e4m3>) -> tensor<128x128x!tessera.fp8.e4m3>
  %x0  = "tessera.solver.potrs"(%L, %b) : (tensor<128x128x!tessera.fp8.e4m3>, tensor<128xf32>) -> tensor<128xf32>
  // CHECK: tessera.solver.potrf
  // CHECK-SAME: policy
  return %x0 : tensor<128xf32>
}
