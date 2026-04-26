// RUN: tessera-opt %s -tpp-space-time -lower-tpp-to-target-ir | FileCheck %s
func.func @shallow_water(%u: tensor<64x64xf32>, %v: tensor<64x64xf32>, %h: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %gu = "tpp.grad"(%h) : (tensor<64x64xf32>) -> tensor<64x64xf32>
  %u1 = "tpp.bc.enforce"(%gu) { bc = "periodic" } : (tensor<64x64xf32>) -> tensor<64x64xf32>
  // CHECK: lowered.bc.masked
  return %u1 : tensor<64x64xf32>
}
