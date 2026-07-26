// RUN: tessera-opt --allow-unregistered-dialect --tessera-tile-pipeline-legality -split-input-file -verify-diagnostics %s | FileCheck %s
//
// C3 (2026-06-23): cross-op pipeline legality — producer phase=1 / consumer
// phase=0 asymmetry (the off-by-one ring-deadlock fix) and per-barrier-id kind
// consistency.

// A correctly-initialized SSA pipeline: producer ring starts at phase 1,
// consumer ring at phase 0, and the kv barrier keeps one kind.
// CHECK-LABEL: func.func @well_formed_pipeline
func.func @well_formed_pipeline() {
  %producer = tile.pipeline_init {depth = 3 : i64, stage = 0 : i64,
    phase = 1 : i64, role = "producer"} : !tile.pipeline_state
  %consumer = tile.pipeline_init {depth = 3 : i64, stage = 0 : i64,
    phase = 0 : i64, role = "consumer"} : !tile.pipeline_state
  "tile.tma_load"(%producer) {tile.barrier_id = "tma2mma",
    tile.barrier = #tile.barrier<kind = "tma", expect = 16384>} : (!tile.pipeline_state) -> ()
  "tile.mma"(%consumer) {tile.barrier_id = "tma2mma",
    tile.barrier = #tile.barrier<kind = "tma", expect = 16384>} : (!tile.pipeline_state) -> ()
  return
}

// -----

// Annotation-only state no longer establishes an executable dependency.
func.func @legacy_pipeline_metadata() {
  // expected-error @+1 {{TILE_PIPELINE_LEGACY_METADATA}}
  "tile.tma_load"() {
    tile.pipeline_state = #tile.pipeline_state<depth = 2, stage = 0, phase = 0, role = "producer">} : () -> ()
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
