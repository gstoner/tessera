// Dynamic per-batch offsets must survive fragment materialization and stores.
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1201})' %s | FileCheck %s
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1201},lower-tessera-target-to-rocdl)' %s | FileCheck %s --check-prefix=ROCDL

!frag_a = !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "a", layout = "row_major", family = "auto">
!frag_b = !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "b", layout = "col_major", family = "auto">
!frag_acc = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">

module {
  gpu.module @batched_fragment_mod {
    gpu.func @batched_fragment_offsets(%a_mem: memref<256xf8E4M3FN>,
                             %b_mem: memref<256xf8E4M3FN>,
                             %d_mem: memref<256xf32>, %offset: index) kernel {
      %zero = arith.constant 0 : index
      %a_batch = memref.reinterpret_cast %a_mem to offset: [%offset], sizes: [256], strides: [1] : memref<256xf8E4M3FN> to memref<256xf8E4M3FN, strided<[1], offset: ?>>
      %b_batch = memref.reinterpret_cast %b_mem to offset: [%offset], sizes: [256], strides: [1] : memref<256xf8E4M3FN> to memref<256xf8E4M3FN, strided<[1], offset: ?>>
      %d_batch = memref.reinterpret_cast %d_mem to offset: [%offset], sizes: [256], strides: [1] : memref<256xf32> to memref<256xf32, strided<[1], offset: ?>>
      %a_tile = tile.view %a_batch, %zero, %zero {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>
      } : (memref<256xf8E4M3FN, strided<[1], offset: ?>>, index, index) -> !tile.tile
      %b_tile = tile.view %b_batch, %zero, %zero {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        tile.memory = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 16>
      } : (memref<256xf8E4M3FN, strided<[1], offset: ?>>, index, index) -> !tile.tile
      %a = tile.fragment_pack %a_tile {
        role = "a",
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!tile.tile) -> !frag_a
      %b = tile.fragment_pack %b_tile {
        role = "b",
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!tile.tile) -> !frag_b
      %c = tile.fragment_zero {
        role = "acc",
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : !frag_acc
      %d = tile.mma %a, %b, %c {
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!frag_a, !frag_b, !frag_acc) -> !frag_acc
      %out = tile.fragment_unpack %d {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!frag_acc) -> !tile.tile
      "tile.store"(%out, %d_batch, %zero, %zero) {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>
      } : (!tile.tile, memref<256xf32, strided<[1], offset: ?>>, index, index) -> ()
      gpu.return
    }
    gpu.func @ragged_batched_fragment_offsets(%a_mem: memref<256xf8E4M3FN>,
                             %b_mem: memref<256xf8E4M3FN>,
                             %d_mem: memref<256xf32>, %offset: index) kernel {
      %zero = arith.constant 0 : index
      %bound = arith.constant 13 : index
      %a_batch = memref.reinterpret_cast %a_mem to offset: [%offset], sizes: [256], strides: [1] : memref<256xf8E4M3FN> to memref<256xf8E4M3FN, strided<[1], offset: ?>>
      %b_batch = memref.reinterpret_cast %b_mem to offset: [%offset], sizes: [256], strides: [1] : memref<256xf8E4M3FN> to memref<256xf8E4M3FN, strided<[1], offset: ?>>
      %d_batch = memref.reinterpret_cast %d_mem to offset: [%offset], sizes: [256], strides: [1] : memref<256xf32> to memref<256xf32, strided<[1], offset: ?>>
      %a_tile = tile.view %a_batch, %zero, %zero, %bound, %bound {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>
      } : (memref<256xf8E4M3FN, strided<[1], offset: ?>>, index, index, index, index) -> !tile.tile
      %b_tile = tile.view %b_batch, %zero, %zero, %bound, %bound {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        tile.memory = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 16>
      } : (memref<256xf8E4M3FN, strided<[1], offset: ?>>, index, index, index, index) -> !tile.tile
      %a = tile.fragment_pack %a_tile {
        role = "a",
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!tile.tile) -> !frag_a
      %b = tile.fragment_pack %b_tile {
        role = "b",
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!tile.tile) -> !frag_b
      %c = tile.fragment_zero {
        role = "acc",
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : !frag_acc
      %d = tile.mma %a, %b, %c {
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!frag_a, !frag_b, !frag_acc) -> !frag_acc
      %out = tile.fragment_unpack %d {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!frag_acc) -> !tile.tile
      "tile.store"(%out, %d_batch, %zero, %zero) {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>
      } : (!tile.tile, memref<256xf32, strided<[1], offset: ?>>, index, index) -> ()
      gpu.return
    }
  }
}

// CHECK-LABEL: gpu.func @batched_fragment_offsets
// CHECK: %[[A:.*]] = memref.reinterpret_cast %arg0 to offset: [%arg3]
// CHECK: %[[B:.*]] = memref.reinterpret_cast %arg1 to offset: [%arg3]
// CHECK: %[[D:.*]] = memref.reinterpret_cast %arg2 to offset: [%arg3]
// CHECK: vector.load %[[A]]
// CHECK: vector.load %[[B]]
// CHECK: tessera_rocm.wmma
// CHECK: memref.store {{.*}}, %[[D]]
// CHECK-NOT: tile.fragment
// CHECK-LABEL: gpu.func @ragged_batched_fragment_offsets
// CHECK: %[[RA:.*]] = memref.reinterpret_cast %arg0 to offset: [%arg3]
// CHECK: %[[RB:.*]] = memref.reinterpret_cast %arg1 to offset: [%arg3]
// CHECK: %[[RD:.*]] = memref.reinterpret_cast %arg2 to offset: [%arg3]
// CHECK: arith.select
// CHECK: memref.load %[[RA]]
// CHECK: memref.load %[[RB]]
// CHECK: tessera_rocm.wmma
// CHECK: memref.store {{.*}}, %[[RD]]
// CHECK-NOT: tile.fragment
// ROCDL: rocdl.wmma
// ROCDL: memref.store {{.*}}, {{.*}} : memref<256xf32, strided<[1], offset: ?>>
