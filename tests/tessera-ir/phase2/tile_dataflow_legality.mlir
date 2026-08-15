// RUN: tessera-opt --tessera-tile-dataflow-legality -split-input-file -verify-diagnostics %s
//
// P1a / CAKE Phase 1 §5.3 — derived Tile sync legality (W2.4's
// TileDataflowLegalityPass, first increment, 2026-08-15). Every negative case
// here was SILENT before this pass existed (measured:
// compiler_enhancement.md §5.1.1; probes preserved in research/tile_sync/).
// The first function is the positive control: the canonical loop-carried
// pipeline shape, fully well-formed, must produce NO diagnostic — a derivation
// pass that only ever rejects proves nothing (Decision #10a analog).

// Well-formed: barrier + pipeline state carried as iter_args; arrive and wait
// agree on slot and barrier; the ring advances through pipeline_advance; the
// copy and its descriptor agree on expect bytes.
func.func @well_formed_loop_pipeline(%src: tensor<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %bar0 = tile.mbarrier.init {slots = 2 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  %ps0 = tile.pipeline_init {depth = 2 : i64, stage = 0 : i64, phase = 1 : i64, role = "producer"} : !tile.pipeline_state
  %barN, %psN = scf.for %i = %c0 to %c8 step %c1
      iter_args(%bar = %bar0, %ps = %ps0) -> (!tile.mbarrier, !tile.pipeline_state) {
    %desc = tile.tma.descriptor %src {tile_rows = 64 : i64, tile_cols = 64 : i64, slot = 0 : i64, expect_tx = 8192 : i64} : tensor<64x64xf16> -> !tile.tma_descriptor
    tile.tma.copy_async %desc, %bar {operandSegmentSizes = array<i32: 1, 1, 0>, mbarrier_slot = 0 : i64, expect_tx = 8192 : i64} : (!tile.tma_descriptor, !tile.mbarrier) -> ()
    %tok = tile.mbarrier.arrive_expect_tx %bar {slot = 0 : i64, bytes = 8192 : i64} : !tile.mbarrier -> !tile.mbarrier_token
    tile.mbarrier.wait %bar, %tok {operandSegmentSizes = array<i32: 1, 1, 0>, slot = 0 : i64} : !tile.mbarrier, !tile.mbarrier_token
    %next = tile.pipeline_advance %ps : (!tile.pipeline_state) -> !tile.pipeline_state
    scf.yield %bar, %next : !tile.mbarrier, !tile.pipeline_state
  }
  return
}

// -----

// Arrive and wait name different slots on a loop-carried barrier: this wait
// never releases on hardware. Was silent (shape-only checks).
func.func @wait_slot_mismatch(%src: tensor<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %bar0 = tile.mbarrier.init {slots = 3 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  %barN = scf.for %i = %c0 to %c8 step %c1 iter_args(%bar = %bar0) -> (!tile.mbarrier) {
    // expected-note @+1 {{token arrives here}}
    %tok = tile.mbarrier.arrive_expect_tx %bar {slot = 1 : i64, bytes = 8192 : i64} : !tile.mbarrier -> !tile.mbarrier_token
    // expected-error @+1 {{TILE_WAIT_SLOT_MISMATCH}}
    tile.mbarrier.wait %bar, %tok {operandSegmentSizes = array<i32: 1, 1, 0>, slot = 0 : i64} : !tile.mbarrier, !tile.mbarrier_token
    scf.yield %bar : !tile.mbarrier
  }
  return
}

// -----

// The wait consumes a token that arrived on a DIFFERENT barrier.
func.func @wait_barrier_disagrees() {
  %a = tile.mbarrier.init {slots = 1 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  %b = tile.mbarrier.init {slots = 1 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  // expected-note @+1 {{token arrives on this barrier}}
  %tok = tile.mbarrier.arrive_expect_tx %a {slot = 0 : i64, bytes = 4096 : i64} : !tile.mbarrier -> !tile.mbarrier_token
  // expected-error @+1 {{TILE_WAIT_BARRIER_DISAGREES}}
  tile.mbarrier.wait %b, %tok {operandSegmentSizes = array<i32: 1, 1, 0>, slot = 0 : i64} : !tile.mbarrier, !tile.mbarrier_token
  return
}

// -----

// The stale ring: the loop yields the un-advanced pipeline state, so the ring
// never moves. Was silent (nothing walked the ring; §5.1.1 experiment 2).
func.func @stale_pipeline_ring() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %ps0 = tile.pipeline_init {depth = 3 : i64, stage = 0 : i64, phase = 1 : i64, role = "producer"} : !tile.pipeline_state
  // expected-error @+1 {{TILE_PIPELINE_RING_STALE}}
  %psN = scf.for %i = %c0 to %c8 step %c1 iter_args(%ps = %ps0) -> (!tile.pipeline_state) {
    %next = tile.pipeline_advance %ps : (!tile.pipeline_state) -> !tile.pipeline_state
    scf.yield %ps : !tile.pipeline_state
  }
  return
}

// -----

// The yielded state is not derived from advancing THIS loop's iter_arg — a
// fresh init inside the body restarts the ring every iteration.
func.func @underived_pipeline_ring() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %ps0 = tile.pipeline_init {depth = 3 : i64, stage = 0 : i64, phase = 1 : i64, role = "producer"} : !tile.pipeline_state
  // expected-error @+1 {{TILE_PIPELINE_RING_UNDERIVED}}
  %psN = scf.for %i = %c0 to %c8 step %c1 iter_args(%ps = %ps0) -> (!tile.pipeline_state) {
    %fresh = tile.pipeline_init {depth = 3 : i64, stage = 0 : i64, phase = 1 : i64, role = "producer"} : !tile.pipeline_state
    scf.yield %fresh : !tile.pipeline_state
  }
  return
}

// -----

// A barrier whose origin cannot be derived (a function argument) fails
// closed: an unprovable origin is unproven, not permitted (CAKE §5.5 gate 2).
func.func @barrier_origin_unresolved(%bar: !tile.mbarrier) {
  // expected-error @+1 {{TILE_BARRIER_ORIGIN_UNRESOLVED}}
  %tok = tile.mbarrier.arrive_expect_tx %bar {slot = 0 : i64, bytes = 4096 : i64} : !tile.mbarrier -> !tile.mbarrier_token
  return
}

// -----

// SSA-keyed expect agreement: one descriptor, one copy, two different
// transaction byte counts. The string-keyed arrival check missed this when
// the two ops carried different tile.barrier_id strings (§5.1 probe).
func.func @tma_expect_mismatch(%src: tensor<64x64xf16>) {
  %bar = tile.mbarrier.init {slots = 1 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  // expected-note @+1 {{descriptor declared here}}
  %desc = tile.tma.descriptor %src {tile_rows = 64 : i64, tile_cols = 64 : i64, slot = 0 : i64, expect_tx = 8192 : i64} : tensor<64x64xf16> -> !tile.tma_descriptor
  // expected-error @+1 {{TILE_TMA_EXPECT_MISMATCH}}
  tile.tma.copy_async %desc, %bar {operandSegmentSizes = array<i32: 1, 1, 0>, mbarrier_slot = 0 : i64, expect_tx = 4096 : i64} : (!tile.tma_descriptor, !tile.mbarrier) -> ()
  return
}
