// RUN: %tnv --allow-unregistered-dialect --lower-tile-to-nvidia='sm=90' %s | FileCheck %s

module {
  func.func @kernel(%a: f32, %b: f32, %dst: !llvm.ptr, %src: !llvm.ptr, %bytes: i64) {
    %m = "tile.mma"(%a, %b) : (f32, f32) -> f32
    %tok = "tile.async_copy"(%dst, %src, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    "tile.wait_async"() : () -> ()
    return
  }
}

// CHECK: tessera_nvidia.wgmma
// CHECK-SAME: arch = "sm_90a"
// CHECK-SAME: shape = "m64n64k16"
// CHECK-SAME: warpgroup = 4
// CHECK: tessera_nvidia.tma_async_copy
// CHECK-SAME: dst_space = "shared"
// CHECK-SAME: src_space = "global"
// CHECK: tessera_nvidia.mbarrier
