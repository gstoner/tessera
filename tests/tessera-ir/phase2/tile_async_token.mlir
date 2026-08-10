// RUN: tessera-opt %s | FileCheck %s
//
// W1.1 steps 1-2 — the typed `tile.async_copy` form.
//
// `!tile.async_token` was declared in the Tile dialect and used NOWHERE: a live
// Decision #29 violation ("a declaration must have a consumer"). It was not an
// oversight — the type exists for the NV warp-spec dependency model that
// replaced program-order reasoning — it simply had no producer or consumer.
//
// The typed form gives it both: `tile.async_copy` yields the token and
// `tile.wait_async` consumes it, so the copy/wait dependency lives in SSA
// def-use rather than in program order. This mirrors `tessera_rocm.async_copy`,
// which already returns a `!tessera_rocm.token` threaded into its `wait` — a
// cross-level precedent that already worked.

// CHECK-LABEL: func.func @typed_async_copy_threads_a_token
func.func @typed_async_copy_threads_a_token() {
  // CHECK: tile.alloc
  %dst = tile.alloc {space = "smem", bytes = 4096 : i64, layout = #tile.layout<shard = [256] : [1] on ["tlane"], replica = [] : [] on [], offset = 0>}
      : !tile.buffer
  %src = tile.alloc {space = "gmem", bytes = 4096 : i64, layout = #tile.layout<shard = [256] : [1] on ["tlane"], replica = [] : [] on [], offset = 0>}
      : !tile.buffer

  // The dependency is the VALUE, not the line order.
  // CHECK: %[[TOK:.*]] = tile.async_copy
  // CHECK-SAME: -> !tile.async_token
  %tok = tile.async_copy %dst, %src
      : (!tile.buffer, !tile.buffer) -> !tile.async_token

  // CHECK: tile.wait_async %[[TOK]]
  tile.wait_async %tok : (!tile.async_token) -> ()

  tile.dealloc %src : !tile.buffer
  tile.dealloc %dst : !tile.buffer
  return
}

// The legacy form stays valid — ten C++ producers still emit it, and ordering
// there is carried by `tile.barrier_id` / `tile.depends_on` (still load-bearing
// on ROCm). W1.1 deletes the permissive branch only after those producers
// migrate; breaking them now would be starting at the end.
//
// CHECK-LABEL: func.func @legacy_untyped_form_still_accepted
func.func @legacy_untyped_form_still_accepted(%a: memref<128xf16>) {
  // CHECK: tile.async_copy
  // CHECK-NOT: !tile.async_token
  "tile.async_copy"(%a) {tile.barrier_id = 0 : i64}
      : (memref<128xf16>) -> ()
  "tile.wait_async"() {tile.barrier_id = 0 : i64} : () -> ()
  return
}

// The other legacy grouping key: an integer `stage`, consumed by
// TileBufferReusePass (lifetime matching) and the Python reference spine. It
// is OPTIONAL — the declared contract (reconciled 2026-08-10) requires only
// that a present key be well formed (>= 0; see the invalid fixture for the
// rejection). A required-stage model was a dead third contract and is gone.
//
// CHECK-LABEL: func.func @legacy_stage_grouping_key_accepted
func.func @legacy_stage_grouping_key_accepted(%a: memref<128xf16>) {
  // CHECK: tile.async_copy
  // CHECK-SAME: stage = 0
  "tile.async_copy"(%a) {stage = 0 : i64} : (memref<128xf16>) -> ()
  // CHECK: tile.wait_async
  "tile.wait_async"() {stage = 0 : i64} : () -> ()
  return
}
