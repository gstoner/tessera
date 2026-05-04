// RUN: %tnv --allow-unregistered-dialect --lower-tile-to-nvidia='sm=100' %s | FileCheck %s

module {
  func.func @kernel(%a: f32, %b: f32) {
    %m = "tile.mma"(%a, %b) : (f32, f32) -> f32
    return
  }
}

// CHECK: tessera_nvidia.tmem_alloc
// CHECK-SAME: arch = "sm_100a"
// CHECK: tessera_nvidia.tcgen05_mma
// CHECK-SAME: accum = "tmem_f32"
// CHECK-SAME: cta_group = 2
// CHECK-SAME: shape = "m128n128k32"
