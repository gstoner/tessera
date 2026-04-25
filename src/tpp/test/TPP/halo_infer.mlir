// RUN: tessera-opt %s -tpp-halo-infer | FileCheck %s
func.func @halo_example(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // tpp.grad would cause halo inference
  %y = "tpp.grad"(%x) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: "tpp.grad"{{.*}} tpp.halo = "1,1,0"
  return %y : tensor<32x32xf32>
}
