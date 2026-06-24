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
}

// CHECK: ROCM_WAVE_LDS_OVERLAPPING_WRITE
