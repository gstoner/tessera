// RUN: not tessera-opt --tessera-tile-buffer-arena %s 2>&1 | FileCheck %s
//
// TILE-LATENT-DEFECTS-2026-09-27. TileBufferArenaPass rechecks every supplied
// reuse group against the shared lifetime proof before it assigns two
// allocations one offset. Two registered TMEM allocations handed the same
// `tile.buffer_group` have no disjointness proof (Tile IR carries no TMEM
// completion fact), so the arena must refuse to alias them -- here %a is even
// still live when %b is written. Before the fix the arena matched only the
// unregistered "tile.tmem.alloc" spelling, never saw these ops, and emitted no
// TMEM arena at all.

func.func @tmem_forced_shared_group(%x: f32, %i: index) -> f32 {
  %a = tile.tmem.allocate {bytes = 256 : i64, alignment = 128 : i64,
                           tile.buffer_group = 0 : i64} : !tile.tmem
  %b = tile.tmem.allocate {bytes = 256 : i64, alignment = 128 : i64,
                           tile.buffer_group = 0 : i64} : !tile.tmem
  tile.tmem.store %x, %b : f32, !tile.tmem
  %v = tile.tmem.load %a, %i : (!tile.tmem, index) -> f32
  return %v : f32
}

// CHECK: TILE_BARRIER_REUSE_MISSING_BARRIER: arena reuse group lacks a disjoint lifetime proof
// CHECK-NOT: tile.tmem_offset
