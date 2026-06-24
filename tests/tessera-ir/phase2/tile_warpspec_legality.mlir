// RUN: tessera-opt --allow-unregistered-dialect --tessera-warpspec-legality -split-input-file -verify-diagnostics %s | FileCheck %s
//
// C6 (2026-06-23, TIRx review / COMPILER_AUDIT item C6): structural warp-spec
// diagnostics from the "Debugging Warp-Specialized Kernels" appendix,
// complementing C3's phase-asymmetry check. A warp-role region is modeled by an
// ancestor carrying `tile.warp_role`.

// A well-formed warp-specialized kernel: barrier init at CTA top level, the
// cta_sync collective at top level, matching producer/consumer trip counts, and
// a visibility fence before the TMA store.
// CHECK-LABEL: func.func @well_formed
func.func @well_formed() {
  "tile.mbarrier_init"() {count = 1 : i64} : () -> ()
  "tile.cta_sync"() : () -> ()
  "tile.tma_loop"() {tile.pipeline = "kv", tile.trip_count = 8 : i64} : () -> ()
  scf.execute_region {
    "tile.mma_loop"() {tile.pipeline = "kv", tile.trip_count = 8 : i64} : () -> ()
    scf.yield
  } {tile.warp_role = "consumer"}
  "tile.fence"() {scope = "shared::cta"} : () -> ()
  "tile.tma_store"() : () -> ()
  return
}

// -----

// Barrier init nested inside a producer-role region → never initializes for the
// other roles → hang.
func.func @init_under_guard() {
  scf.execute_region {
    // expected-error @+1 {{WARPSPEC_INIT_UNDER_GUARD}}
    "tile.mbarrier_init"() {count = 1 : i64} : () -> ()
    scf.yield
  } {tile.warp_role = "producer"}
  return
}

// -----

// A cta_sync inside a warp-role branch → partial participation hangs.
func.func @collective_in_branch() {
  scf.execute_region {
    // expected-error @+1 {{WARPSPEC_COLLECTIVE_IN_DIVERGENT_BRANCH}}
    "tile.cta_sync"() : () -> ()
    scf.yield
  } {tile.warp_role = "producer"}
  return
}

// -----

// Producer TMA loop count (8) disagrees with consumer MMA loop count (7).
func.func @loop_count_disagree() {
  // expected-note @+1 {{pipeline "kv" first trip count here}}
  "tile.tma_loop"() {tile.pipeline = "kv", tile.trip_count = 8 : i64} : () -> ()
  // expected-error @+1 {{WARPSPEC_LOOP_COUNT_DISAGREE}}
  "tile.mma_loop"() {tile.pipeline = "kv", tile.trip_count = 7 : i64} : () -> ()
  return
}

// -----

// TMA store with no prior visibility fence in its block.
func.func @missing_fence() {
  // expected-error @+1 {{WARPSPEC_MISSING_VISIBILITY_FENCE}}
  "tile.tma_store"() : () -> ()
  return
}

// -----

// One barrier id whose arrive count (4096) disagrees with its init count
// (8192) — the wait would never release. Fed in real lowering by
// NVTMADescriptorPass's typed #tile.barrier emission on setup + copy_async.
func.func @arrival_count_mismatch() {
  // expected-note @+1 {{barrier "mbar.0" init count here}}
  "tile.tma.setup_descriptor"() {tile.barrier_id = "mbar.0", tile.barrier = #tile.barrier<kind = "tma", expect = 8192>} : () -> ()
  // expected-error @+1 {{WARPSPEC_ARRIVAL_COUNT_MISMATCH}}
  "tile.tma.copy_async"() {tile.barrier_id = "mbar.0", tile.barrier = #tile.barrier<kind = "tma", expect = 4096>} : () -> ()
  return
}

// -----

// A buffer freed during writeback with no prior cta_sync — a warp may still be
// reading it. Emitted in real lowering by WarpSpecialization's dealloc epilogue
// (which DOES precede the frees with a cta_sync, so correct lowering is clean).
func.func @use_after_free() {
  // expected-error @+1 {{WARPSPEC_USE_AFTER_FREE}}
  "tile.buffer_free"() {tile.access = "free", tile.buffer = "warpspec.0.smem.0"} : () -> ()
  return
}
