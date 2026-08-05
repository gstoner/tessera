// RUN: not tessera-opt --split-input-file %s 2>&1 | FileCheck %s

#lay = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                    replica = [] : [] on [], offset = 0>
#dyn = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>

func.func @view_missing_dynamic_stride(%src: memref<?xf32>, %r: index,
                                       %c: index) {
  // CHECK: TILE_VIEW_POINTER_ARITY
  // CHECK-SAME: got 3 operands
  %tile = tile.view %src, %r, %c {tile.layout = #lay, tile.memory = #dyn}
      : (memref<?xf32>, index, index) -> !tile.tile
  return
}

// -----

#lay = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                    replica = [] : [] on [], offset = 0>
#dyn = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>

func.func @store_missing_dynamic_stride(%src: memref<?xf32>,
                                        %dst: memref<?xf32>, %r: index,
                                        %c: index, %ld: index) {
  %tile = tile.view %src, %r, %c, %ld
      {tile.layout = #lay, tile.memory = #dyn}
      : (memref<?xf32>, index, index, index) -> !tile.tile
  // CHECK: TILE_STORE_POINTER_ARITY
  // CHECK-SAME: got 4 operands
  tile.store %tile, %dst, %r, %c {tile.layout = #lay, tile.memory = #dyn}
      : !tile.tile, memref<?xf32>, index, index
  return
}
