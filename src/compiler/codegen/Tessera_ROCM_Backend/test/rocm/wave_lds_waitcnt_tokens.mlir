// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-pipeline)' %s | FileCheck %s --check-prefix=PLAN
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-pipeline,lower-tile-to-rocm{arch=gfx1100})' %s | FileCheck %s --check-prefix=LOWER
//
// Barrier-id waitcnt contract (replaces the scalar lastAsyncToken model). Two
// async copies followed by two waits: the planner stamps per-copy barrier ids
// and per-wait waitcnt thresholds (oldest-retired-first, so vmcnt(1) then
// vmcnt(0)); the lowering retires the matching token per wait, NOT "the last".

// Each copy mints a !tile.async_token result (the SSA completion edge); each
// wait consumes the token of the copy it retires (oldest first), not "the last".
// PLAN: %[[C0:.*]]:2 = tile.async_copy
// PLAN-SAME: tile.barrier = #tile.barrier<kind = "waitcnt", expect = 0>
// PLAN-SAME: tile.barrier_id = "rocm.waitcnt.0"
// PLAN-SAME: -> (i32, !tile.async_token)
// PLAN: %[[C1:.*]]:2 = tile.async_copy
// PLAN-SAME: tile.barrier_id = "rocm.waitcnt.1"
// PLAN-SAME: -> (i32, !tile.async_token)
// The first wait consumes C0's token + retires the OLDEST id (rocm.waitcnt.0)
// with one still outstanding (threshold 1); the second consumes C1's token
// (threshold 0).
// PLAN: tile.wait_async %[[C0]]#1
// PLAN-SAME: tile.barrier_id = "rocm.waitcnt.0"
// PLAN-SAME: tile.waitcnt_threshold = 1
// PLAN: tile.wait_async %[[C1]]#1
// PLAN-SAME: tile.barrier_id = "rocm.waitcnt.1"
// PLAN-SAME: tile.waitcnt_threshold = 0

// LOWER: %[[A0:.*]] = tessera_rocm.async_copy
// LOWER: %[[A1:.*]] = tessera_rocm.async_copy
// Each wait gates a DISTINCT copy (oldest first), with the threshold metadata.
// LOWER: tessera_rocm.wait %[[A0]] {barrier_id = "rocm.waitcnt.0", counter = "vmcnt", threshold = 1 : i64}
// LOWER: tessera_rocm.wait %[[A1]] {barrier_id = "rocm.waitcnt.1", counter = "vmcnt", threshold = 0 : i64}
func.func @double_buffer(%d0: memref<64xf16>, %d1: memref<64xf16>,
                         %s: memref<64xf16>, %b: index) {
  %t0 = "tile.async_copy"(%d0, %s, %b) : (memref<64xf16>, memref<64xf16>, index) -> i32
  %t1 = "tile.async_copy"(%d1, %s, %b) : (memref<64xf16>, memref<64xf16>, index) -> i32
  "tile.wait_async"() : () -> ()
  "tile.wait_async"() : () -> ()
  return
}
