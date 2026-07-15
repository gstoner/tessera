// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' %s | FileCheck %s
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151},lower-tessera-target-to-rocdl)' %s | FileCheck %s --check-prefix=ROCDL

module {
  gpu.module @fragment_mod {
    gpu.func @fragment_store(%a_mem: memref<256xf16>,
                             %b_mem: memref<256xf16>,
                             %d_mem: memref<256xf32>) kernel {
      %zero = arith.constant 0 : index
      %a_tile = tile.view %a_mem, %zero, %zero {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>
      } : (memref<256xf16>, index, index) -> !tile.tile
      %b_tile = tile.view %b_mem, %zero, %zero {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        tile.memory = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 16>
      } : (memref<256xf16>, index, index) -> !tile.tile
      %a = tile.fragment_pack %a_tile {
        role = "a",
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!tile.tile) -> !tile.fragment
      %b = tile.fragment_pack %b_tile {
        role = "b",
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!tile.tile) -> !tile.fragment
      %c = tile.fragment_zero {
        role = "acc",
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : !tile.fragment
      %d = tile.mma %a, %b, %c {
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!tile.fragment, !tile.fragment, !tile.fragment) -> !tile.fragment
      %out = tile.fragment_unpack %d {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!tile.fragment) -> !tile.tile
      "tile.store"(%out, %d_mem, %zero, %zero) {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>
      } : (!tile.tile, memref<256xf32>, index, index) -> ()
      gpu.return
    }
  }
}

// CHECK-LABEL: gpu.func @fragment_store
// CHECK: gpu.thread_id x
// CHECK: vector.load
// CHECK: tessera_rocm.wmma
// CHECK-SAME: arch = "gfx1151"
// CHECK-SAME: shape = "m16n16k16"
// CHECK-SAME: source = "tile.fragment_pack"
// CHECK: vector.extract
// CHECK: memref.store
// CHECK-NOT: tile.fragment
// CHECK-NOT: tile.store

// ROCDL: rocdl.wmma.f32.16x16x16.f16
// ROCDL-NOT: tile.fragment
