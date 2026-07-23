// RUN: tessera-opt --tessera-tile-buffer-reuse --tessera-tile-buffer-arena --allow-unregistered-dialect %s | FileCheck %s
// RUN: tessera-opt --tessera-tile-buffer-arena --allow-unregistered-dialect %s | FileCheck %s --check-prefix=DYNAMIC
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
// CHECK: memref.get_global @__tessera_smem_arena_arena_disjoint
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
// CHECK: memref.get_global @__tessera_smem_arena_arena_smem_tmem_separate
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
// CHECK: memref.get_global @__tessera_smem_arena_arena_chain_collapses
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
// CHECK: memref.get_global @__tessera_smem_arena_arena_alignment_padding
// CHECK: arith.constant 0 : index
// CHECK: memref.view{{.*}}to memref<1xi8, 3>
// CHECK: arith.constant 4 : index
// CHECK: memref.view{{.*}}to memref<1xf32, 3>
func.func @arena_alignment_padding(%arg0: memref<1xi8>, %arg1: memref<1xf32>) {
  "tile.alloc_shared"(%arg0) : (memref<1xi8>) -> ()
  "tile.alloc_shared"(%arg1) : (memref<1xf32>) -> ()
  return
}

// -----

// Dynamic extents are sized once at function entry. Disjoint buffers in the
// same reuse group share the same runtime offset and the group reserves the
// maximum member size.
// CHECK-LABEL: func.func @arena_dynamic
// CHECK-SAME: tile.smem_arena_dynamic
// CHECK-SAME: tile.smem_arena_materialized
// CHECK: memref.dim %arg0, %c0
// CHECK: memref.dim %arg1, %c0
// CHECK: memref.alloca(%{{.*}}) {alignment = 16 : i64} : memref<?xi8, 3>
// CHECK-COUNT-2: memref.view {{.*}} : memref<?xi8, 3> to memref<?x16xf16, 3>
// CHECK-NOT: tile.alloc_shared
// DYNAMIC-LABEL: func.func @arena_dynamic
// DYNAMIC-SAME: tile.smem_arena_dynamic
// DYNAMIC: arith.maxui
// DYNAMIC: memref.alloca(%{{.*}}) {alignment = 16 : i64} : memref<?xi8, 3>
// DYNAMIC-COUNT-2: memref.view {{.*}} : memref<?xi8, 3> to memref<?x16xf16, 3>
func.func @arena_dynamic(%arg0: memref<?x16xf16>,
                         %arg1: memref<?x16xf16>) {
  "tile.alloc_shared"(%arg0) {tile.buffer_group = 0 : i64}
      : (memref<?x16xf16>) -> ()
  "tile.alloc_shared"(%arg1) {tile.buffer_group = 0 : i64}
      : (memref<?x16xf16>) -> ()
  return
}

// -----

// A descriptor created inside a nested dominance region owns a region-local
// arena. It must not be hoisted to function entry, where %local and its runtime
// extent do not exist.
// DYNAMIC-LABEL: func.func @arena_dynamic_branch
// DYNAMIC-SAME: tile.smem_arena_dynamic
// DYNAMIC-SAME: tile.smem_arena_materialized
// DYNAMIC-SAME: tile.smem_arena_regions = 1
// DYNAMIC-NOT: tile.smem_arena_dynamic_unresolved
// DYNAMIC: scf.if %{{.*}} {
// DYNAMIC: %[[LOCAL:.*]] = memref.alloca(%{{.*}}) : memref<?xf16>
// DYNAMIC: memref.dim %[[LOCAL]], %c0
// DYNAMIC: %[[ARENA:.*]] = memref.alloca(%{{.*}}) {alignment = 16 : i64} : memref<?xi8, 3>
// DYNAMIC: memref.view %[[ARENA]]
// DYNAMIC-NOT: tile.alloc_shared
func.func @arena_dynamic_branch(%cond: i1, %n: index) {
  scf.if %cond {
    %local = memref.alloca(%n) : memref<?xf16>
    "tile.alloc_shared"(%local) {tile.buffer_group = 0 : i64}
        : (memref<?xf16>) -> ()
    "tile.async_copy"(%local) : (memref<?xf16>) -> ()
  }
  return
}

// A later descriptor starts a second cohort instead of forcing an illegal
// hoist above its definition.
// DYNAMIC-LABEL: func.func @arena_dynamic_sequential_cohorts
// DYNAMIC-SAME: tile.smem_arena_regions = 2
// DYNAMIC-COUNT-2: memref.alloca(%{{.*}}) {alignment = 16 : i64} : memref<?xi8, 3>
// DYNAMIC-NOT: tile.alloc_shared
func.func @arena_dynamic_sequential_cohorts(%arg0: memref<?xf16>, %n: index) {
  "tile.alloc_shared"(%arg0) {tile.buffer_group = 0 : i64}
      : (memref<?xf16>) -> ()
  "tile.async_copy"(%arg0) : (memref<?xf16>) -> ()
  %local = memref.alloca(%n) : memref<?xf16>
  "tile.alloc_shared"(%local) {tile.buffer_group = 1 : i64}
      : (memref<?xf16>) -> ()
  "tile.async_copy"(%local) : (memref<?xf16>) -> ()
  return
}
