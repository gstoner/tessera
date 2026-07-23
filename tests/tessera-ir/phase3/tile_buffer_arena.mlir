// RUN: tessera-opt --tessera-tile-buffer-reuse --tessera-tile-buffer-arena --allow-unregistered-dialect %s | FileCheck %s
//
// TileBufferArenaPass (Workstream H / W3 follow-on, 2026-07-08): the first
// CONSUMER of TileBufferReusePass's tile.buffer_group. It realizes the reuse plan
// into a concrete per-space arena — a byte offset per alloc (same-group buffers
// share an offset = the promised aliasing) + the arena size on the func. This is
// the form a shared-memory backend emits directly
// (`__shared__ char arena[N]; T* buf = arena + offset`).

// ── Disjoint reuse realized: buf0 and buf2 (same group) alias at offset 0. ───
// buf1 overlaps buf0 → distinct group → offset 512. Arena = 1024 B (= the reuse
// bytes_after: two live tiles, not three).
// CHECK-LABEL: func.func @arena_disjoint
// CHECK-SAME: tile.smem_arena_bytes = 1024
// CHECK: memref.alloca() {alignment = 16 : i64} : memref<1024xi8, 3>
// CHECK: arith.constant 0 : index
// CHECK: memref.view{{.*}}to memref<16x16xf16, 3>
// CHECK: arith.constant 512 : index
// CHECK: memref.view{{.*}}to memref<16x16xf16, 3>
// CHECK: arith.constant 0 : index
// CHECK: memref.view{{.*}}to memref<16x16xf16, 3>
// CHECK-NOT: tile.alloc_shared
func.func @arena_disjoint(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>,
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

// ── SMEM and TMEM are laid out in SEPARATE arenas (distinct spaces). ─────────
// Each is the sole member of its space → both at offset 0, each arena = 512 B.
// CHECK-LABEL: func.func @arena_smem_tmem_separate
// CHECK-SAME: tile.smem_arena_bytes = 512
// CHECK-SAME: tile.tmem_arena_bytes = 512
// CHECK: memref.alloca() {alignment = 16 : i64} : memref<512xi8, 3>
// CHECK: memref.view{{.*}}to memref<16x16xf16, 3>
// CHECK: "tile.tmem.alloc"(%arg1) {tile.buffer_group = 1 : i64, tile.tmem_offset = 0 : i64}
func.func @arena_smem_tmem_separate(%arg0: memref<16x16xf16>,
                                    %arg1: memref<16x16xf16>) {
  "tile.alloc_shared"(%arg0) : (memref<16x16xf16>) -> ()
  "tile.tmem.alloc"(%arg1) : (memref<16x16xf16>) -> ()
  return
}

// -----

// ── A fully-serial chain collapses to one group → all alias at offset 0. ─────
// Three same-space same-type buffers used one-after-another → one group, one
// arena slot (512 B), every alloc at offset 0.
// CHECK-LABEL: func.func @arena_chain_collapses
// CHECK-SAME: tile.smem_arena_bytes = 512
// CHECK: memref.alloca() {alignment = 16 : i64} : memref<512xi8, 3>
// CHECK-COUNT-3: memref.view{{.*}}to memref<16x16xf16, 3>
func.func @arena_chain_collapses(%arg0: memref<16x16xf16>,
                                 %arg1: memref<16x16xf16>,
                                 %arg2: memref<16x16xf16>) {
  "tile.alloc_shared"(%arg0) : (memref<16x16xf16>) -> ()
  "tile.alloc_shared"(%arg1) : (memref<16x16xf16>) -> ()
  "tile.alloc_shared"(%arg2) : (memref<16x16xf16>) -> ()
  return
}

// -----

// ── Offsets are padded up to each group's element alignment. ─────────────────
// An i8 group (1 B) then an f32 group (4 B): the f32 must land at a 4-aligned
// offset (4, NOT the raw cumulative 1) so `arena + offset` cast to f32* is legal;
// the arena size (8) includes the padding.
// CHECK-LABEL: func.func @arena_alignment_padding
// CHECK-SAME: tile.smem_arena_bytes = 8
// CHECK: memref.alloca() {alignment = 16 : i64} : memref<8xi8, 3>
// CHECK: arith.constant 0 : index
// CHECK: memref.view{{.*}}to memref<1xi8, 3>
// CHECK: arith.constant 4 : index
// CHECK: memref.view{{.*}}to memref<1xf32, 3>
func.func @arena_alignment_padding(%arg0: memref<1xi8>, %arg1: memref<1xf32>) {
  "tile.alloc_shared"(%arg0) : (memref<1xi8>) -> ()
  "tile.alloc_shared"(%arg1) : (memref<1xf32>) -> ()
  return
}
