// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-pipeline)' %s | FileCheck %s --check-prefix=PLAN
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-pipeline,rocm-wave-lds-legality,lower-tile-to-rocm{arch=gfx1151})' %s | FileCheck %s --check-prefix=LOWER

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

// PLAN: %[[CONSUMER_ROLE:.*]] = tile.role
// PLAN-SAME: kind = "consumer"
// PLAN-SAME: name = "rocm.wave_lds.consumer"
// PLAN: %[[CONSUMER:.*]] = tile.pipeline_init %[[CONSUMER_ROLE]]
// PLAN-SAME: role = "consumer"
// PLAN: %[[PRODUCER_ROLE:.*]] = tile.role
// PLAN-SAME: kind = "producer"
// PLAN-SAME: name = "rocm.wave_lds.producer"
// PLAN: %[[PRODUCER:.*]] = tile.pipeline_init %[[PRODUCER_ROLE]]
// PLAN-SAME: role = "producer"
// PLAN: %[[BUFFER:.*]] = tile.alloc
// PLAN-SAME: bytes = 4096
// PLAN-SAME: space = "smem"
// PLAN: %[[COPY:.*]]:2 = tile.async_copy
// PLAN-SAME: %[[BUFFER]], %[[PRODUCER]]
// PLAN-SAME: -> (!tessera_rocm.token, !tile.async_token)
// PLAN: tile.pipeline_advance %[[PRODUCER]]
// PLAN: tile.wait_async %[[COPY]]#1, %[[BUFFER]], %[[CONSUMER]]
// PLAN: tile.pipeline_advance %[[CONSUMER]]
// PLAN: tile.mma
// PLAN-SAME: %[[COPY]]#1, %[[BUFFER]]
// PLAN: tile.dealloc %[[BUFFER]]
// PLAN-NOT: tile.buf = #tile.buffer_ref

// LOWER: tessera_rocm.async_copy
// LOWER-SAME: buffer_bytes = 4096
// LOWER-SAME: buffer_identity = "ssa_allocation"
// LOWER-SAME: buffer_space = "lds"
// LOWER-SAME: dst_space = "lds"
// LOWER-SAME: layout_storage = "lds"
// LOWER-SAME: numeric_policy = {accum = "f32", storage = "f16"}
// LOWER-SAME: tile.layout = #tile.layout<shard = [64, 32] : [32, 1] on ["lds", "waveid"]
// LOWER-SAME: tile.pipeline_depths = #tile.pipeline_depths<q = 1, kv = 2, tmem = 1>
// LOWER-SAME: uses_tile_layout = true
// LOWER: tessera_rocm.wait
// LOWER-SAME: counter = "vmcnt"
// LOWER: tessera_rocm.wmma
// LOWER-SAME: arch = "gfx1151"
// LOWER-SAME: tile.pipeline_depths = #tile.pipeline_depths<q = 1, kv = 2, tmem = 1>
// LOWER-NOT: tile.alloc
// LOWER-NOT: tile.pipeline_
