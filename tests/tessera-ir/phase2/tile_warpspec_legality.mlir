// RUN: tessera-opt --tessera-warpspec-legality -split-input-file -verify-diagnostics %s | FileCheck %s
//
// C6 (2026-06-23, TIRx review / COMPILER_AUDIT item C6): structural warp-spec
// diagnostics from the "Debugging Warp-Specialized Kernels" appendix,
// complementing C3's phase-asymmetry check. A warp-role region is modeled by an
// ancestor carrying `tile.warp_role`.
//
// Ported to the REGISTERED vocabulary (P1a second increment, CAKE §5.4,
// 2026-08-15): no `--allow-unregistered-dialect`, no husk spellings, and the
// legality predicates that recognize these ops are typed (`isa<...>`), not
// name substrings. The producer/consumer loop markers are `scf.for` loops
// carrying the `tile.pipeline` / `tile.trip_count` attributes the C6
// trip-count check reads.

// A well-formed warp-specialized kernel: barrier init at CTA top level, the
// cta_sync collective at top level, matching producer/consumer trip counts,
// and a visibility fence before the TMA store.
// CHECK-LABEL: func.func @well_formed
func.func @well_formed() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %bar = tile.mbarrier.init {slots = 1 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  tile.cta_sync
  scf.for %i = %c0 to %c8 step %c1 {
    scf.yield
  } {tile.pipeline = "kv", tile.trip_count = 8 : i64}
  scf.execute_region {
    scf.for %j = %c0 to %c8 step %c1 {
      scf.yield
    } {tile.pipeline = "kv", tile.trip_count = 8 : i64}
    scf.yield
  } {tile.warp_role = "consumer"}
  tile.fence {scope = "shared::cta"}
  tile.tma.store
  return
}

// -----

// Barrier init nested inside a producer-role region → never initializes for the
// other roles → hang.
func.func @init_under_guard() {
  scf.execute_region {
    // expected-error @+1 {{WARPSPEC_INIT_UNDER_GUARD}}
    %bar = tile.mbarrier.init {slots = 1 : i64, phase_bits = 1 : i64} : !tile.mbarrier
    scf.yield
  } {tile.warp_role = "producer"}
  return
}

// -----

// A cta_sync inside a warp-role branch → partial participation hangs.
func.func @collective_in_branch() {
  scf.execute_region {
    // expected-error @+1 {{WARPSPEC_COLLECTIVE_IN_DIVERGENT_BRANCH}}
    tile.cta_sync
    scf.yield
  } {tile.warp_role = "producer"}
  return
}

// -----

// Producer TMA loop count (8) disagrees with consumer MMA loop count (7).
func.func @loop_count_disagree() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  // expected-note @+1 {{pipeline "kv" first trip count here}}
  scf.for %i = %c0 to %c8 step %c1 {
    scf.yield
  } {tile.pipeline = "kv", tile.trip_count = 8 : i64}
  // expected-error @+1 {{WARPSPEC_LOOP_COUNT_DISAGREE}}
  scf.for %j = %c0 to %c7 step %c1 {
    scf.yield
  } {tile.pipeline = "kv", tile.trip_count = 7 : i64}
  return
}

// -----

// TMA store with no prior visibility fence in its block.
func.func @missing_fence() {
  // expected-error @+1 {{WARPSPEC_MISSING_VISIBILITY_FENCE}}
  tile.tma.store
  return
}

// -----

// One barrier id whose arrive count (4096) disagrees with its init count
// (8192) — the wait would never release. Fed in real lowering by
// NVTMADescriptorPass's typed #tile.barrier emission on descriptor +
// copy_async. This fixture uses the registered operations so the legality
// proof cannot silently depend on a stale string-only carrier.
func.func @arrival_count_mismatch(%src: tensor<64x64xf16>) {
  // expected-note @+1 {{barrier "mbar.0" init count here}}
  %desc = "tile.tma.descriptor"(%src) {tile_rows = 64 : i64, tile_cols = 64 : i64, tile.barrier_id = "mbar.0", tile.barrier = #tile.barrier<kind = "tma", expect = 8192>} : (tensor<64x64xf16>) -> !tile.tma_descriptor
  // expected-error @+1 {{WARPSPEC_ARRIVAL_COUNT_MISMATCH}}
  "tile.tma.copy_async"(%desc) {operandSegmentSizes = array<i32: 1, 0, 0>, mbarrier_slot = 0 : i64, tile.barrier_id = "mbar.0", tile.barrier = #tile.barrier<kind = "tma", expect = 4096>} : (!tile.tma_descriptor) -> ()
  return
}

// -----

// A buffer freed during writeback with no prior cta_sync — a warp may still be
// reading it. Emitted in real lowering by WarpSpecialization's dealloc epilogue
// (which DOES precede the frees with a cta_sync, so correct lowering is clean).
func.func @use_after_free() {
  %buffer = tile.alloc {
    bytes = 256 : i64, space = "smem",
    layout = #tile.layout<shard = [256] : [1] on ["m"],
                          replica = [] : [] on [], offset = 0>
  } : !tile.buffer
  // expected-error @+1 {{WARPSPEC_USE_AFTER_FREE}}
  tile.dealloc %buffer : !tile.buffer
  return
}
