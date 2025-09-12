// RUN: tessera-opt %s -lower-tpp-to-target-ir | FileCheck %s
func.func @bc(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %y = "tpp.bc.enforce"(%x) { bc = "periodic" } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: lowered.bc.masked
  return %y : tensor<32x32xf32>
}
