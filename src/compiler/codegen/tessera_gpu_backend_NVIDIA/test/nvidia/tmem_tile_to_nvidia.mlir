// RUN: %tnv --lower-tile-to-nvidia='sm=100' %s | FileCheck %s

// TILE-LATENT-DEFECTS-2026-09-27. The three registered Tile TMEM ops lower by
// op identity, and the results they define are WIRED, not dropped: the load
// result feeds arith and the store, and the allocation handle feeds both
// accesses. Before the fix this branch erased each op without replacing its
// uses, so this fixture aborted an assertions-ON tessera-nvidia-opt
// ("operation destroyed but still has uses"). Datacenter sm_100 hardware is
// not in the fleet: this is IR/lowering evidence only, never execution.

module {
  func.func @tmem_roundtrip(%i: index, %j: index, %bias: f32) {
    %tmem = tile.tmem.allocate {bytes = 16384 : i64, alignment = 128 : i64,
                             tile.buffer_group = 0 : i64, tile.tmem_offset = 0 : i64}
        : !tile.tmem
    %v = tile.tmem.load %tmem, %i, %j : (!tile.tmem, index, index) -> f32
    %w = arith.addf %v, %bias : f32
    tile.tmem.store %w, %tmem [%i, %j] : f32, !tile.tmem, index, index
    return
  }
}

// CHECK-LABEL: func.func @tmem_roundtrip
// CHECK-SAME: (%[[I:.*]]: index, %[[J:.*]]: index, %[[BIAS:.*]]: f32)
// CHECK: %[[ADDR:.*]] = tessera_nvidia.tmem_alloc
// CHECK-SAME: alignment = 128 : i64
// CHECK-SAME: arch = "sm_100a"
// CHECK-SAME: bytes = 16384 : i64
// The arena placement planned before this pass survives the boundary (#32).
// CHECK-SAME: tile.buffer_group = 0 : i64, tile.tmem_offset = 0 : i64
// CHECK-SAME: : () -> i32
// CHECK: %[[I64:.*]] = arith.index_cast %[[I]] : index to i64
// CHECK: %[[J64:.*]] = arith.index_cast %[[J]] : index to i64
// CHECK: %[[V:.*]] = tessera_nvidia.tmem_load %[[ADDR]], %[[I64]], %[[J64]]
// CHECK-SAME: : (i32, i64, i64) -> f32
// CHECK: %[[W:.*]] = arith.addf %[[V]], %[[BIAS]] : f32
// CHECK: tessera_nvidia.tmem_store %[[W]], %[[ADDR]]
// CHECK-SAME: : (f32, i32, i64, i64) -> ()
// CHECK-NOT: tile.tmem
// CHECK: return

