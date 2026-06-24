// RUN: tessera-opt --allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s
//
// C3 (2026-06-23, TIRx review / COMPILER_AUDIT item C3): typed barrier domains
// (#tile.barrier, kind = tma/tcgen05/mbarrier) + the ring-buffer PipelineState
// (#tile.pipeline_state, depth/stage/phase/role). Round-trip + per-attribute
// verifier bounds.

// CHECK-LABEL: func.func @typed_barriers
func.func @typed_barriers() {
  // CHECK: #tile.barrier<kind = "tma", expect = 16384>
  "test.bar"() {b = #tile.barrier<kind = "tma", expect = 16384>} : () -> ()
  // CHECK: #tile.barrier<kind = "tcgen05", expect = 1>
  "test.bar"() {b = #tile.barrier<kind = "tcgen05", expect = 1>} : () -> ()
  // CHECK: #tile.barrier<kind = "mbarrier", expect = 128>
  "test.bar"() {b = #tile.barrier<kind = "mbarrier", expect = 128>} : () -> ()
  // AMD is first-class: s_barrier (workgroup arrival) + waitcnt (async counter)
  // are the AMD completion-semantics domains alongside the NVIDIA ones.
  // CHECK: #tile.barrier<kind = "s_barrier", expect = 256>
  "test.bar"() {b = #tile.barrier<kind = "s_barrier", expect = 256>} : () -> ()
  // CHECK: #tile.barrier<kind = "waitcnt", expect = 0>
  "test.bar"() {b = #tile.barrier<kind = "waitcnt", expect = 0>} : () -> ()
  return
}

// CHECK-LABEL: func.func @pipeline_state
func.func @pipeline_state() {
  // CHECK: #tile.pipeline_state<depth = 3, stage = 0, phase = 1, role = "producer">
  "test.ps"() {p = #tile.pipeline_state<depth = 3, stage = 0, phase = 1, role = "producer">} : () -> ()
  // CHECK: #tile.pipeline_state<depth = 3, stage = 0, phase = 0, role = "consumer">
  "test.ps"() {p = #tile.pipeline_state<depth = 3, stage = 0, phase = 0, role = "consumer">} : () -> ()
  return
}

// C5 scaffold — independent per-ring depths (Q=2, KV=3, TMEM=2 book defaults).
// CHECK-LABEL: func.func @pipeline_depths
func.func @pipeline_depths() {
  // CHECK: #tile.pipeline_depths<q = 2, kv = 3, tmem = 2>
  "test.pd"() {d = #tile.pipeline_depths<q = 2, kv = 3, tmem = 2>} : () -> ()
  return
}

// -----
func.func @bad_barrier_kind() {
  // expected-error @+1 {{TILE_BARRIER_UNKNOWN_KIND}}
  "test.bar"() {b = #tile.barrier<kind = "bogus", expect = 0>} : () -> ()
  return
}

// -----
func.func @bad_barrier_expect() {
  // expected-error @+1 {{TILE_BARRIER_NEGATIVE_EXPECT}}
  "test.bar"() {b = #tile.barrier<kind = "tma", expect = -1>} : () -> ()
  return
}

// -----
func.func @bad_pipeline_depth() {
  // expected-error @+1 {{TILE_PIPELINE_BAD_DEPTH}}
  "test.ps"() {p = #tile.pipeline_state<depth = 0, stage = 0, phase = 1, role = "producer">} : () -> ()
  return
}

// -----
func.func @bad_pipeline_stage() {
  // expected-error @+1 {{TILE_PIPELINE_STAGE_OOB}}
  "test.ps"() {p = #tile.pipeline_state<depth = 2, stage = 5, phase = 0, role = "consumer">} : () -> ()
  return
}

// -----
func.func @bad_pipeline_phase() {
  // expected-error @+1 {{TILE_PIPELINE_BAD_PHASE}}
  "test.ps"() {p = #tile.pipeline_state<depth = 2, stage = 0, phase = 2, role = "producer">} : () -> ()
  return
}

// -----
func.func @bad_pipeline_role() {
  // expected-error @+1 {{TILE_PIPELINE_BAD_ROLE}}
  "test.ps"() {p = #tile.pipeline_state<depth = 2, stage = 0, phase = 0, role = "bogus">} : () -> ()
  return
}

// -----
func.func @bad_ring_depth() {
  // expected-error @+1 {{TILE_PIPELINE_DEPTHS_NONPOSITIVE}}
  "test.pd"() {d = #tile.pipeline_depths<q = 0, kv = 3, tmem = 2>} : () -> ()
  return
}
