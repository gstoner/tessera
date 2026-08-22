// RUN: %tnv --lower-tile-to-nvidia='sm=120' --lower-tessera-nvidia-to-nvvm %s | FileCheck %s
// RUN: %tnv --lower-tile-to-nvidia='sm=120' %s | FileCheck %s --check-prefix=ABI
//
// Complete compiler path for one m16n8k16 f16 tile with f32 accumulation:
// pointer-backed A/B loads, real MMA, accumulator unpack, row-major D stores.

!fa = !tile.fragment<m = 16, n = 8, k = 16, elem = "f16", acc = "f32", role = "a", layout = "row_major", family = "mma_sync">
!fb = !tile.fragment<m = 16, n = 8, k = 16, elem = "f16", acc = "f32", role = "b", layout = "col_major", family = "mma_sync">
!fc = !tile.fragment<m = 16, n = 8, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "mma_sync">

module {
  llvm.func @pointer_fragment_store(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr,
                                    %d_ptr: !llvm.ptr, %zero: i64)
      attributes {nvvm.kernel} {
    %a_tile = tile.view %a_ptr, %zero, %zero {
      tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>
    } : (!llvm.ptr, i64, i64) -> !tile.tile
    %b_tile = tile.view %b_ptr, %zero, %zero {
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 16>
    } : (!llvm.ptr, i64, i64) -> !tile.tile
    %a = tile.fragment_pack %a_tile {
      role = "a",
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.tile) -> !fa
    %b = tile.fragment_pack %b_tile {
      role = "b",
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.tile) -> !fb
    %c = tile.fragment_zero {
      role = "acc",
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : !fc
    %d = tile.mma %a, %b, %c {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!fa, !fb, !fc) -> !fc
    %out = tile.fragment_unpack %d {
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!fc) -> !tile.tile
    "tile.store"(%out, %d_ptr, %zero, %zero) {
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 8>
    } : (!tile.tile, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @pointer_fragment_store
// CHECK: nvvm.read.ptx.sreg.tid.x
// CHECK: nvvm.mma.sync A[
// CHECK-SAME: C[{{.*}}] {layoutA
// CHECK: llvm.extractvalue
// CHECK-NEXT: llvm.store
// CHECK: llvm.extractvalue
// CHECK-NEXT: llvm.store
// CHECK: llvm.extractvalue
// CHECK-NEXT: llvm.store
// CHECK: llvm.extractvalue
// CHECK-NEXT: llvm.store
// CHECK-NOT: tile.fragment
// CHECK-NOT: tile.store

// ABI: tessera_nvidia.mma_sync
// ABI-SAME: a_registers_per_lane = 4
// ABI-SAME: accumulator_registers_per_lane = 4
// ABI-SAME: b_registers_per_lane = 2
// ABI-SAME: instruction_family = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
