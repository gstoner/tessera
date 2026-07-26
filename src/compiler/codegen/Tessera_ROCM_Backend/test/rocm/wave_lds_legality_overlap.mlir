// RUN: not %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-legality)' %s 2>&1 | FileCheck %s

module {
  func.func @overlapping_lds_writes(%dst: !llvm.ptr, %src: !llvm.ptr, %bytes: i64) {
    %tok0 = "tile.async_copy"(%dst, %src, %bytes) {
      tile.buf = #tile.buffer_ref<name = "stage.lds.0", space = "lds", access = "write">,
      tile.layout = #tile.layout<shard = [64, 32] : [32, 1] on ["lds", "waveid"], replica = [] : [] on [], offset = 0>
    } : (!llvm.ptr, !llvm.ptr, i64) -> !tessera_rocm.token
    %tok1 = "tile.async_copy"(%dst, %src, %bytes) {
      tile.buf = #tile.buffer_ref<name = "stage.lds.0", space = "lds", access = "write">,
      tile.layout = #tile.layout<shard = [64, 32] : [32, 1] on ["lds", "waveid"], replica = [] : [] on [], offset = 0>
    } : (!llvm.ptr, !llvm.ptr, i64) -> !tessera_rocm.token
    return
  }

  func.func @overlapping_ssa_lds_writes(%dst: !llvm.ptr, %src: !llvm.ptr, %bytes: i64) {
    %buffer = tile.alloc {
      bytes = 4096 : i64,
      layout = #tile.layout<shard = [64, 32] : [32, 1] on ["lds", "waveid"], replica = [] : [] on [], offset = 0>,
      space = "smem"
    } : !tile.buffer
    %tok0 = "tile.async_copy"(%dst, %src, %bytes, %buffer) {
      tile.layout = #tile.layout<shard = [64, 32] : [32, 1] on ["lds", "waveid"], replica = [] : [] on [], offset = 0>
    } : (!llvm.ptr, !llvm.ptr, i64, !tile.buffer) -> !tessera_rocm.token
    %tok1 = "tile.async_copy"(%dst, %src, %bytes, %buffer) {
      tile.layout = #tile.layout<shard = [64, 32] : [32, 1] on ["lds", "waveid"], replica = [] : [] on [], offset = 0>
    } : (!llvm.ptr, !llvm.ptr, i64, !tile.buffer) -> !tessera_rocm.token
    tile.dealloc %buffer : !tile.buffer
    return
  }

  func.func @wait_retires_only_named_ssa_buffer(%dst: !llvm.ptr, %src: !llvm.ptr, %bytes: i64) {
    %buffer0 = tile.alloc {
      bytes = 4096 : i64,
      layout = #tile.layout<shard = [64, 32] : [32, 1] on ["lds", "waveid"], replica = [] : [] on [], offset = 0>,
      space = "smem"
    } : !tile.buffer
    %buffer1 = tile.alloc {
      bytes = 4096 : i64,
      layout = #tile.layout<shard = [64, 32] : [32, 1] on ["lds", "waveid"], replica = [] : [] on [], offset = 0>,
      space = "smem"
    } : !tile.buffer
    %tok0 = "tile.async_copy"(%dst, %src, %bytes, %buffer0) {
      tile.layout = #tile.layout<shard = [64, 32] : [32, 1] on ["lds", "waveid"], replica = [] : [] on [], offset = 0>
    } : (!llvm.ptr, !llvm.ptr, i64, !tile.buffer) -> !tessera_rocm.token
    %tok1 = "tile.async_copy"(%dst, %src, %bytes, %buffer1) {
      tile.layout = #tile.layout<shard = [64, 32] : [32, 1] on ["lds", "waveid"], replica = [] : [] on [], offset = 0>
    } : (!llvm.ptr, !llvm.ptr, i64, !tile.buffer) -> !tessera_rocm.token
    "tile.wait_async"(%buffer0) : (!tile.buffer) -> ()
    %tok2 = "tile.async_copy"(%dst, %src, %bytes, %buffer1) {
      tile.layout = #tile.layout<shard = [64, 32] : [32, 1] on ["lds", "waveid"], replica = [] : [] on [], offset = 0>
    } : (!llvm.ptr, !llvm.ptr, i64, !tile.buffer) -> !tessera_rocm.token
    tile.dealloc %buffer0 : !tile.buffer
    tile.dealloc %buffer1 : !tile.buffer
    return
  }
}

// CHECK: ROCM_WAVE_LDS_OVERLAPPING_WRITE
// CHECK: ROCM_WAVE_LDS_OVERLAPPING_WRITE
// CHECK: ROCM_WAVE_LDS_OVERLAPPING_WRITE
