// P2 code review (2026-08-29) — TileBufferArenaPass crashed on a non-scalar
// element type, and WarpSpecLegalityPass lost a staged operand behind an
// intervening op.
//
// RUN: tessera-opt --tessera-tile-buffer-arena --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s

// staticByteSize called getIntOrFloatBitWidth unconditionally, which asserts on
// a vector element type. Every neighbouring unknown (dynamic shape, missing
// size) leaves the group unplaced, so this one must too rather than take the
// tool down.
// CHECK-LABEL: func.func @vector_element_leaves_group_unplaced
// CHECK-SAME: tile.smem_arena_bytes = 0
// CHECK-NOT: tile.smem_arena_materialized
func.func @vector_element_leaves_group_unplaced(
    %a: memref<64x8xvector<4xf16>, 3>) {
  "tile.alloc_shared"(%a) {tile.buffer_group = 0 : i64}
      : (memref<64x8xvector<4xf16>, 3>) -> ()
  return
}

// A scalar element type is still placed, so the guard did not cost the
// supported path: 64 * 8 * 2 bytes.
// CHECK-LABEL: func.func @scalar_element_is_still_placed
// CHECK-SAME: tile.smem_arena_bytes = 1024
func.func @scalar_element_is_still_placed(%a: memref<64x8xf16, 3>) {
  "tile.alloc_shared"(%a) {tile.buffer_group = 0 : i64}
      : (memref<64x8xf16, 3>) -> ()
  return
}
