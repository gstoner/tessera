// RUN: tessera-opt %s | FileCheck %s
//
// A problem-size-generic Tile producer cannot encode N/K in a static
// #tile.memory_layout. leading_dim=0 therefore means the final SSA operand of
// tile.view/tile.store supplies the element stride at runtime. Bounds remain
// immediately before that final operand, so the unbounded and bounded forms
// are unambiguous.

#lay = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                    replica = [] : [] on [], offset = 0>
#dyn = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>

// CHECK-LABEL: func.func @dynamic_leading_dim
// CHECK: tile.view {{.*}}leading_dim = 0
// CHECK: tile.view {{.*}}leading_dim = 0
// CHECK: tile.store {{.*}}leading_dim = 0
// CHECK: tile.store {{.*}}leading_dim = 0
func.func @dynamic_leading_dim(%src: memref<?xf32>, %dst: memref<?xf32>,
                               %r: index, %c: index, %rows: index,
                               %cols: index, %ld: index) {
  %plain = tile.view %src, %r, %c, %ld
      {tile.layout = #lay, tile.memory = #dyn}
      : (memref<?xf32>, index, index, index) -> !tile.tile
  %bounded = tile.view %src, %r, %c, %rows, %cols, %ld
      {tile.layout = #lay, tile.memory = #dyn}
      : (memref<?xf32>, index, index, index, index, index) -> !tile.tile
  tile.store %plain, %dst, %r, %c, %ld
      {tile.layout = #lay, tile.memory = #dyn}
      : !tile.tile, memref<?xf32>, index, index, index
  tile.store %bounded, %dst, %r, %c, %rows, %cols, %ld
      {tile.layout = #lay, tile.memory = #dyn}
      : !tile.tile, memref<?xf32>, index, index, index, index, index
  return
}
