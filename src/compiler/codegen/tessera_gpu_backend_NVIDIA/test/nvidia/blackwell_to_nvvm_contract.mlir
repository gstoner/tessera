// RUN: %tnv --allow-unregistered-dialect --tessera-lower-to-blackwell %s | FileCheck %s
// RUN: %tnv --allow-unregistered-dialect --tessera-lower-to-nvidia-sm100 %s | FileCheck %s

module {
  func.func @kernel(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>) {
    %m = "tile.mma"(%a, %b) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    return
  }
}

// CHECK: llvm.call @llvm.nvvm.tmem.alloc.contract
// CHECK: llvm.call @llvm.nvvm.tcgen05.mma.contract
// CHECK-NOT: tessera_nvidia.
// CHECK-NOT: tile.mma
