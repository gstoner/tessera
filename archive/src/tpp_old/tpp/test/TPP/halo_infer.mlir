// RUN: tessera-opt %s -tpp-halo-infer | FileCheck %s
// CHECK: // (halo inferred)
func.func @halo_example(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // tpp.grad would cause halo=(1,1); placeholder body
  return %x : tensor<32x32xf32>
}
