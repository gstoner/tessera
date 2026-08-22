// RUN: %tnv --allow-unregistered-dialect --lower-tile-to-nvidia='sm=100' %s | FileCheck %s

module {
  func.func @kernel(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>) {
    %m = "tile.mma"(%a, %b) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    return
  }
}

// CHECK: tessera_nvidia.tmem_alloc
// CHECK-SAME: arch = "sm_100a"
// CHECK: tessera_nvidia.tcgen05_mma
// CHECK-SAME: accum = "tmem_f32"
// CHECK-SAME: cta_group = 2
// CHECK-SAME: shape = "m128n128k32"
