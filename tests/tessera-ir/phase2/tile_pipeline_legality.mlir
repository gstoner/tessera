// RUN: tessera-opt --allow-unregistered-dialect --tessera-tile-pipeline-legality -split-input-file -verify-diagnostics %s | FileCheck %s
//
// C3 (2026-06-23): cross-op pipeline legality — producer phase=1 / consumer
// phase=0 asymmetry (the off-by-one ring-deadlock fix) and per-barrier-id kind
// consistency.

// A correctly-initialized warp-specialized pipeline: producer ring starts at
// phase 1, consumer ring at phase 0, and the kv barrier keeps one kind.
// CHECK-LABEL: func.func @well_formed_pipeline
func.func @well_formed_pipeline() {
  "tile.tma_load"() {tile.pipeline = "kv", tile.barrier_id = "tma2mma",
    tile.barrier = #tile.barrier<kind = "tma", expect = 16384>,
    tile.pipeline_state = #tile.pipeline_state<depth = 3, stage = 0, phase = 1, role = "producer">} : () -> ()
  "tile.mma"() {tile.pipeline = "kv", tile.barrier_id = "tma2mma",
    tile.barrier = #tile.barrier<kind = "tma", expect = 16384>,
    tile.pipeline_state = #tile.pipeline_state<depth = 3, stage = 0, phase = 0, role = "consumer">} : () -> ()
  return
}

// -----

// Producer initialized at phase 0 → its first wait blocks forever (classic
// off-by-one ring deadlock).
func.func @producer_phase_zero() {
  // expected-error @+1 {{TILE_PIPELINE_PHASE_ASYMMETRY}}
  "tile.tma_load"() {tile.pipeline = "kv",
    tile.pipeline_state = #tile.pipeline_state<depth = 2, stage = 0, phase = 0, role = "producer">} : () -> ()
  "tile.mma"() {tile.pipeline = "kv",
    tile.pipeline_state = #tile.pipeline_state<depth = 2, stage = 0, phase = 0, role = "consumer">} : () -> ()
  return
}

// -----

// One barrier id signaled with two different completion semantics — a latent
// hang (engine byte-count vs MMA-completion).
func.func @barrier_kind_mismatch() {
  // expected-note @+1 {{first use of barrier "b0" here}}
  "tile.tma_load"() {tile.barrier_id = "b0",
    tile.barrier = #tile.barrier<kind = "tma", expect = 256>} : () -> ()
  // expected-error @+1 {{TILE_PIPELINE_BARRIER_KIND_MISMATCH}}
  "tile.mma"() {tile.barrier_id = "b0",
    tile.barrier = #tile.barrier<kind = "tcgen05", expect = 1>} : () -> ()
  return
}
