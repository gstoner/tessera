// REQUIRES: tessera-apple-backend
//
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-apple-threadgroup-pipeline)' | FileCheck %s
//
// The placement must also hold through the real target pipeline, not only when
// the pass is driven directly — the physical record is decided once, early,
// and every later Apple lowering reads it rather than re-deriving it.
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu)' \
// RUN:   --allow-unregistered-dialect | FileCheck %s

// APPLE-PIPE-1 (2026-07-26) — Apple is a real consumer of the shared Tile
// physical-allocation and staged-pipeline SSA vocabulary.
//
// The fixture supplies only portable Tile IR: `!tile.buffer` allocations and
// `!tile.pipeline_state` producer/consumer rings, exactly as the AMD wave/LDS
// planner and the NVIDIA warp-specialized path receive them. Apple owns the
// physical answer — 16-byte-aligned placement inside one per-function
// threadgroup arena, and ping-pong staging as the Metal realization of a
// depth-2 ring. Nothing here names a Metal API; that stays in the emitter.

// The function-level arena is the sum of the placed allocations, so a reader
// can see the total threadgroup-memory demand without walking every alloc.
// CHECK-LABEL: func.func @staged_gemm_tiles
// CHECK-SAME: tessera_apple.threadgroup_arena_bytes = 4096
func.func @staged_gemm_tiles() {
  // First allocation anchors the arena at offset 0.
  // CHECK: tile.alloc
  // CHECK-SAME: tessera_apple.address_space = "threadgroup"
  // CHECK-SAME: tessera_apple.threadgroup_bytes = 2048
  // CHECK-SAME: tessera_apple.threadgroup_capacity_bytes = 32768
  // CHECK-SAME: tessera_apple.threadgroup_offset = 0
  %a = tile.alloc {space = "smem", bytes = 2048 : i64,
                   layout = #tile.layout<shard = [32, 16] : [16, 1] on ["m", "m"], replica = [] : [] on [], offset = 0>}
      : !tile.buffer

  // The staged B tile is placed after A, not aliased onto it.
  // CHECK: tile.alloc
  // CHECK-SAME: tessera_apple.threadgroup_bytes = 2048
  // CHECK-SAME: tessera_apple.threadgroup_offset = 2048
  %b = tile.alloc {space = "smem", bytes = 2048 : i64,
                   layout = #tile.layout<shard = [16, 32] : [32, 1] on ["m", "m"], replica = [] : [] on [], offset = 0>}
      : !tile.buffer

  // A depth-2 producer ring is exactly Metal ping-pong staging.
  // CHECK: tile.pipeline_init
  // CHECK-SAME: tessera_apple.stage_buffering = "ping_pong"
  // CHECK-SAME: tessera_apple.stage_depth = 2
  // CHECK-SAME: tessera_apple.stage_role = "producer"
  %prod = tile.pipeline_init {depth = 2 : i64, stage = 0 : i64, phase = 1 : i64,
                              role = "producer"} : !tile.pipeline_state

  // The consumer ring is a distinct SSA state, claimed on its own terms.
  // CHECK: tile.pipeline_init
  // CHECK-SAME: tessera_apple.stage_role = "consumer"
  %cons = tile.pipeline_init {depth = 2 : i64, stage = 0 : i64, phase = 0 : i64,
                              role = "consumer"} : !tile.pipeline_state

  // Advances are claimed only because they carry real SSA state — identity
  // travels through def-use, never through a name.
  // CHECK: tile.pipeline_advance
  // CHECK-SAME: tessera_apple.stage_buffering = "ping_pong"
  %prod1 = tile.pipeline_advance %prod
      : (!tile.pipeline_state) -> !tile.pipeline_state
  // CHECK: tile.pipeline_advance
  // CHECK-SAME: tessera_apple.stage_buffering = "ping_pong"
  %cons1 = tile.pipeline_advance %cons
      : (!tile.pipeline_state) -> !tile.pipeline_state

  tile.dealloc %b : !tile.buffer
  tile.dealloc %a : !tile.buffer
  return
}

// A single-buffered ring is legal and is named as such, so the emitter never
// has to infer buffering mode from the depth attribute.
// CHECK-LABEL: func.func @single_buffered_ring
func.func @single_buffered_ring() {
  // CHECK: tile.pipeline_init
  // CHECK-SAME: tessera_apple.stage_buffering = "single"
  %s = tile.pipeline_init {depth = 1 : i64, stage = 0 : i64, phase = 1 : i64,
                           role = "producer"} : !tile.pipeline_state
  return
}

// A function with no Tile allocation gets no arena attribute: the pass states
// a demand it measured, never a zero it assumed.
// CHECK-LABEL: func.func @no_threadgroup_demand
// CHECK-NOT: tessera_apple.threadgroup_arena_bytes
func.func @no_threadgroup_demand() {
  return
}
