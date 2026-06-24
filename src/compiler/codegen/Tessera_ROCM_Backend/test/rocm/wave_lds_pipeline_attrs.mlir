// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-pipeline,rocm-wave-lds-legality,lower-tile-to-rocm{arch=gfx1151})' %s | FileCheck %s

module {
  func.func @pipeline_attrs(%dst: !llvm.ptr, %src: !llvm.ptr, %bytes: i64, %a: f16, %b: f16) -> f16 {
    %tok = "tile.async_copy"(%dst, %src, %bytes) {
      tile_rows = 64 : i64,
      tile_cols = 32 : i64,
      numeric_policy = {storage = "f16", accum = "f32"}
    } : (!llvm.ptr, !llvm.ptr, i64) -> !tessera_rocm.token
    "tile.wait_async"() : () -> ()
    %m = "tile.mma"(%a, %b) : (f16, f16) -> f16
    return %m : f16
  }
}

// CHECK: tessera_rocm.async_copy
// CHECK-SAME: buffer = "rocm.lds.0"
// CHECK-SAME: dst_space = "lds"
// CHECK-SAME: layout_storage = "lds"
// CHECK-SAME: numeric_policy = {accum = "f32", storage = "f16"}
// CHECK-SAME: tile.layout = #tile.layout<shard = [64, 32] : [32, 1] on ["lds", "waveid"]
// CHECK-SAME: tile.pipeline_depths = #tile.pipeline_depths<q = 1, kv = 2, tmem = 1>
// CHECK-SAME: uses_tile_layout = true
// CHECK: tessera_rocm.wait
// CHECK-SAME: counter = "vmcnt"
// CHECK: tessera_rocm.wmma
// CHECK-SAME: arch = "gfx1151"
// CHECK-SAME: tile.pipeline_depths = #tile.pipeline_depths<q = 1, kv = 2, tmem = 1>
