// RUN: tessera-opt --tessera-tile-buffer-reuse --allow-unregistered-dialect %s | FileCheck %s
//
// TileBufferReusePass (Workstream H / W3, 2026-07-08): global buffer assignment/
// reuse for Tile IR. Shared-memory staging buffers (tile.alloc_shared / TMEM)
// with DISJOINT live ranges + identical memref type share one reuse group
// (tile.buffer_group), cutting the statically-allocated shared-memory footprint.
// The assignment half of shared-memory planning; TileBarrierReuseLegalityPass is
// the verifier. Output is IR metadata a shared-memory-aware backend consumes.

// ── Disjoint live ranges reuse; an overlapping buffer gets its own group. ────
// %arg0 is live over ops [0..4]; %arg1 (group 1) overlaps it; %arg2 starts only
// after %arg0's last use, so it REUSES group 0. 3 buffers → 2 groups; the static
// footprint drops from 3 tiles to 2 (512 B of 16x16xf16 saved).
// CHECK-LABEL: func.func @reuse_disjoint
// CHECK-SAME: tile.buffer_reuse.bytes_after = 1024
// CHECK-SAME: tile.buffer_reuse.bytes_before = 1536
// CHECK-SAME: tile.buffer_reuse.groups = 2
// CHECK: "tile.alloc_shared"(%arg0) {tile.buffer_group = 0
// CHECK: "tile.alloc_shared"(%arg1) {tile.buffer_group = 1
// CHECK: "tile.alloc_shared"(%arg2) {tile.buffer_group = 0
func.func @reuse_disjoint(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>,
                          %arg2: memref<16x16xf16>) {
  "tile.alloc_shared"(%arg0) : (memref<16x16xf16>) -> ()
  "tile.async_copy"(%arg0) : (memref<16x16xf16>) -> ()
  "tile.alloc_shared"(%arg1) : (memref<16x16xf16>) -> ()
  "tile.async_copy"(%arg1) : (memref<16x16xf16>) -> ()
  "tile.async_copy"(%arg0) : (memref<16x16xf16>) -> ()
  "tile.alloc_shared"(%arg2) : (memref<16x16xf16>) -> ()
  "tile.async_copy"(%arg2) : (memref<16x16xf16>) -> ()
  return
}

// -----

// ── Different memref types never share a group (backing size differs). ───────
// Even though their live ranges are disjoint, an f16 tile and an f32 tile of
// different byte size must not alias — each is its own group.
// CHECK-LABEL: func.func @distinct_types_not_shared
// CHECK-SAME: tile.buffer_reuse.groups = 2
// CHECK: "tile.alloc_shared"(%arg0) {tile.buffer_group = 0
// CHECK: "tile.alloc_shared"(%arg1) {tile.buffer_group = 1
func.func @distinct_types_not_shared(%arg0: memref<8x8xf16>,
                                     %arg1: memref<8x8xf32>) {
  "tile.alloc_shared"(%arg0) : (memref<8x8xf16>) -> ()
  "tile.async_copy"(%arg0) : (memref<8x8xf16>) -> ()
  "tile.alloc_shared"(%arg1) : (memref<8x8xf32>) -> ()
  "tile.async_copy"(%arg1) : (memref<8x8xf32>) -> ()
  return
}

// -----

// ── TMEM allocations are planned too, and a chain of fully-disjoint buffers
// collapses to a single group. ──────────────────────────────────────────────
// Three same-type buffers used strictly one-after-another → all reuse group 0.
// CHECK-LABEL: func.func @tmem_chain_collapses
// CHECK-SAME: tile.buffer_reuse.bytes_after = 512
// CHECK-SAME: tile.buffer_reuse.bytes_before = 1536
// CHECK-SAME: tile.buffer_reuse.groups = 1
// CHECK: "tile.tmem.alloc"(%arg0) {tile.buffer_group = 0
// CHECK: "tile.tmem.alloc"(%arg1) {tile.buffer_group = 0
// CHECK: "tile.tmem.alloc"(%arg2) {tile.buffer_group = 0
func.func @tmem_chain_collapses(%arg0: memref<16x16xf16>,
                                %arg1: memref<16x16xf16>,
                                %arg2: memref<16x16xf16>) {
  "tile.tmem.alloc"(%arg0) : (memref<16x16xf16>) -> ()
  "tile.tmem.alloc"(%arg1) : (memref<16x16xf16>) -> ()
  "tile.tmem.alloc"(%arg2) : (memref<16x16xf16>) -> ()
  return
}
