// RUN: %tessera_strict_opt --tessera-tile-barrier-reuse-legality -split-input-file -verify-diagnostics %s

func.func @earlier_disjoint_write_is_not_forgotten() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  // expected-note @+1 {{previous write to the same allocation root}}
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 16>} : !tile.buffer
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  return
}

// -----

func.func @typed_wait_releases_only_its_copy() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %b = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %src = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %ta = tile.async_copy %a, %src : (!tile.buffer, !tile.buffer) -> !tile.async_token
  // expected-note @+1 {{previous write to the same allocation root}}
  %tb = tile.async_copy %b, %src : (!tile.buffer, !tile.buffer) -> !tile.async_token
  tile.wait_async %ta : (!tile.async_token) -> ()
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
  tile.buffer_write %b {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  return
}

// -----

func.func @typed_wait_completes_own_allocation() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %src = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %t = tile.async_copy %a, %src : (!tile.buffer, !tile.buffer) -> !tile.async_token
  tile.wait_async %t : (!tile.async_token) -> ()
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.cta_sync
  tile.dealloc %a : !tile.buffer
  tile.dealloc %src : !tile.buffer
  return
}

// -----

func.func @wait_in_one_branch_does_not_release_other(%cond: i1) {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  // expected-note @+1 {{previous write to the same allocation root}}
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  scf.if %cond {
    tile.cta_sync
  }
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  return
}

// -----

func.func @wait_in_both_branches_releases(%cond: i1) {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  scf.if %cond {
    tile.cta_sync
  } else {
    tile.cta_sync
  }
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  return
}

// -----

func.func @exclusive_branch_writes_do_not_conflict(%cond: i1) {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  scf.if %cond {
    tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  } else {
    tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  }
  return
}

// -----

func.func @loop_backedge_requires_completion() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
  // expected-note @+2 {{previous write to the same allocation root}}
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  }
  return
}

// -----

func.func @loop_backedge_completed() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.cta_sync
  }
  return
}

// -----

func.func @dealloc_before_consumer_completion() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  // expected-note @+1 {{previous write to the same allocation root}}
  %t = tile.async_copy %a : (!tile.buffer) -> !tile.async_token
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
  tile.dealloc %a : !tile.buffer
  return
}

// -----

func.func @freed_on_one_path(%cond: i1) {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  scf.if %cond {
  tile.dealloc %a : !tile.buffer
  }
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  return
}

// -----

func.func @cfg_join_keeps_unreleased_path(%cond: i1) {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  // expected-note @+1 {{previous write to the same allocation root}}
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  cf.cond_br %cond, ^yes, ^no
  ^yes:
  tile.cta_sync
  cf.br ^join
  ^no:
  cf.br ^join
  ^join:
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  return
}

// -----

func.func @stage_wait_preserves_other_stage() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  // expected-note @+1 {{previous write to the same allocation root}}
  tile.async_copy %a {stage = 1 : i64} : (!tile.buffer) -> ()
  tile.wait_async {stage = 0 : i64} : () -> ()
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  return
}

// -----

func.func @thread_barrier_does_not_complete_dma() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  // expected-note @+1 {{previous write to the same allocation root}}
  %t = tile.async_copy %a : (!tile.buffer) -> !tile.async_token
  tile.cta_sync
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
  tile.dealloc %a : !tile.buffer
  return
}

// -----

func.func @zero_trip_loop_has_no_access() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c0 step %c1 {
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  }
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  return
}

// -----

func.func @one_trip_loop_has_no_backedge() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  }
  return
}

// -----

func.func @two_async_writers_conflict() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %src = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  // expected-note @+1 {{previous write to the same allocation root}}
  %t0 = tile.async_copy %a, %src : (!tile.buffer, !tile.buffer) -> !tile.async_token
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
  %t1 = tile.async_copy %a, %src : (!tile.buffer, !tile.buffer) -> !tile.async_token
  return
}

// -----

func.func @unknown_buffer_argument_fails_closed(%a: !tile.buffer) {
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  return
}

// -----

func.func @loop_with_direct_tokens_completes() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %src = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
  %t = tile.async_copy %a, %src : (!tile.buffer, !tile.buffer) -> !tile.async_token
  tile.wait_async %t : (!tile.async_token) -> ()
  }
  tile.dealloc %a : !tile.buffer
  tile.dealloc %src : !tile.buffer
  return
}

// -----

func.func @conditional_loop_alias_resolution_terminates(%cond: i1) {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %out = scf.for %i = %c0 to %c2 step %c1 iter_args(%b = %a) -> (!tile.buffer) {
  tile.buffer_write %b {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.cta_sync
  %v = scf.if %cond -> (!tile.buffer) {
  scf.yield %b : !tile.buffer
  } else {
  scf.yield %b : !tile.buffer
  }
  scf.yield %v : !tile.buffer
  }
  tile.buffer_write %out {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.cta_sync
  tile.dealloc %a : !tile.buffer
  return
}

// -----

func.func @role_local_wait_does_not_release_inherited_dma() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  // expected-note @+1 {{previous write to the same allocation root}}
  %t = tile.async_copy %a : (!tile.buffer) -> !tile.async_token
  "schedule.warp"() ({
    tile.wait_async : () -> ()
    "schedule.yield"() : () -> ()
  }) {role = "consumer"} : () -> ()
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
  tile.dealloc %a : !tile.buffer
  return
}

// -----

func.func @arrival_wait_is_not_allocation_completion() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %bar = tile.mbarrier.init {slots = 1 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  %token = tile.mbarrier.arrive_expect_tx %bar {slot = 0 : i64, bytes = 64 : i64} : !tile.mbarrier -> !tile.mbarrier_token
  // expected-note @+1 {{previous write to the same allocation root}}
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.mbarrier.wait %bar, %token {operandSegmentSizes = array<i32: 1, 1, 0>, slot = 0 : i64} : !tile.mbarrier, !tile.mbarrier_token
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  return
}

// -----

func.func @negative_stride_footprint_overlaps() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  // expected-note @+1 {{previous write to the same allocation root}}
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [4] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
  tile.buffer_write %a {tile.layout = #tile.layout<shard = [8] : [-1] on ["m"], replica = [] : [] on [], offset = 7>} : !tile.buffer
  return
}
