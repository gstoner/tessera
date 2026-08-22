// RUN: %tnv --allow-unregistered-dialect --tessera-lower-to-hopper %s | FileCheck %s
// RUN: %tnv --allow-unregistered-dialect --tessera-lower-to-nvidia-sm90 %s | FileCheck %s

module {
  func.func @kernel(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %dst: !llvm.ptr, %src: !llvm.ptr, %bytes: i64) {
    %m = "tile.mma"(%a, %b) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %tok = "tile.async_copy"(%dst, %src, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    "tile.wait_async"() : () -> ()
    return
  }
}

// CHECK: llvm.call @llvm.nvvm.wgmma.contract
// CHECK: llvm.call @llvm.nvvm.cp.async.bulk.tensor.contract
// CHECK: llvm.call @llvm.nvvm.mbarrier.contract
// CHECK-NOT: tessera_nvidia.
// CHECK-NOT: tile.
