// RUN: tessera-opt %s --allow-unregistered-dialect --tessera-autodiff-forward=emit-storage-child=true | FileCheck %s
module attributes {tessera.frontend.authority = "tracer"} {
  func.func @square(%x: tensor<32xf32>) -> tensor<32xf32>
      attributes {tessera.autodiff = "forward"} {
    %y = tessera.mul %x, %x : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    return %y : tensor<32xf32>
  }
}
// CHECK: tessera.native_jvp_inputs = 2 : i64
// CHECK: tessera.native_jvp_width = 32 : i64
// CHECK: gpu.func @paired_child
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: tile.alloc_shared
// CHECK: gpu.barrier
// CHECK: llvm.store
// CHECK: tile.alloc_shared
// CHECK: gpu.barrier
// CHECK: llvm.store
