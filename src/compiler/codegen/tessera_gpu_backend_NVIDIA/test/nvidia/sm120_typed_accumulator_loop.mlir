// RUN: %tnv --lower-tile-to-nvidia='sm=120' --lower-tessera-nvidia-to-nvvm %s | FileCheck %s
//
// The typed fragment path must structurally convert an scf.for fragment
// iter-arg and feed the converted accumulator registers into every MMA in the
// K loop.  A zero accumulator is only the loop initializer; the body must not
// synthesize a fresh zero for each iteration.

!fa = !tile.fragment<m = 16, n = 8, k = 16, elem = "f16", acc = "f32", role = "a", layout = "row_major", family = "mma_sync">
!fb = !tile.fragment<m = 16, n = 8, k = 16, elem = "f16", acc = "f32", role = "b", layout = "col_major", family = "mma_sync">
!fc = !tile.fragment<m = 16, n = 8, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "mma_sync">

module {
  llvm.func @typed_accumulator_loop(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr,
                                    %d_ptr: !llvm.ptr, %zero: i64,
                                    %k_bound: i64) attributes {nvvm.kernel} {
    %c0_index = arith.constant 0 : index
    %c1_index = arith.constant 1 : index
    %k_bound_index = arith.index_cast %k_bound : i64 to index
    %c0 = tile.fragment_zero {
      role = "acc",
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : !fc
    %acc = scf.for %k = %c0_index to %k_bound_index step %c1_index iter_args(%carry = %c0) -> (!fc) {
      %k_i64 = arith.index_cast %k : index to i64
      %a_tile = tile.view %a_ptr, %zero, %k_i64 {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>
      } : (!llvm.ptr, i64, i64) -> !tile.tile
      %b_tile = tile.view %b_ptr, %k_i64, %zero {
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
      %next = tile.mma %a, %b, %carry {
        mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!fa, !fb, !fc) -> !fc
      scf.yield %next : !fc
    }
    %out = tile.fragment_unpack %acc {
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

// CHECK-LABEL: llvm.func @typed_accumulator_loop
// CHECK: scf.for {{.*}} iter_args
// CHECK: nvvm.mma.sync A[
// CHECK-SAME: C[
// CHECK: llvm.extractvalue
// CHECK-NOT: tile.mma
// CHECK-NOT: tile.fragment
