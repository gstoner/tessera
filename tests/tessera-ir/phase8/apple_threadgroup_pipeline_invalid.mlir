// REQUIRES: tessera-apple-backend
//
// RUN: tessera-opt %s --allow-unregistered-dialect \
// RUN:   --pass-pipeline='builtin.module(tessera-apple-threadgroup-pipeline)' \
// RUN:   -split-input-file -verify-diagnostics

// APPLE-PIPE-1 rejection contract. Apple's position on the shared Tile
// physical vocabulary is a *declared* capability boundary: every construct
// Metal cannot realize fails with a stable diagnostic naming the operation and
// the target, never a silent no-op (Architecture Decision #21).

// Metal has no tensor memory. `tmem` is datacenter-Blackwell vocabulary.
func.func @rejects_tensor_memory() {
  // expected-error @+1 {{APPLE_THREADGROUP_SPACE_UNSUPPORTED}}
  %t = tile.alloc {space = "tmem", bytes = 1024 : i64,
                   layout = #tile.layout<shard = [32, 8] : [8, 1] on ["m", "m"], replica = [] : [] on [], offset = 0>}
      : !tile.buffer
  return
}

// -----

// Device memory is a real allocation, but it is not threadgroup memory and
// this pass does not own it. Saying so is the honest answer.
func.func @rejects_global_allocation() {
  // expected-error @+1 {{APPLE_THREADGROUP_SPACE_UNSUPPORTED}}
  %g = tile.alloc {space = "gmem", bytes = 1024 : i64,
                   layout = #tile.layout<shard = [32, 8] : [8, 1] on ["m", "m"], replica = [] : [] on [], offset = 0>}
      : !tile.buffer
  return
}

// -----

// The threadgroup arena is capacity-bounded. This is the IR-level peer of the
// `APPLE_FRAGMENT_THREADGROUP_MEMORY_EXCEEDED` check the TILE-1 simdgroup
// materializer already applies to its staged bytes.
func.func @rejects_over_capacity_arena() {
  %a = tile.alloc {space = "smem", bytes = 30000 : i64,
                   layout = #tile.layout<shard = [32, 8] : [8, 1] on ["m", "m"], replica = [] : [] on [], offset = 0>}
      : !tile.buffer
  // expected-error @+1 {{APPLE_THREADGROUP_MEMORY_EXCEEDED}}
  %b = tile.alloc {space = "smem", bytes = 4096 : i64,
                   layout = #tile.layout<shard = [32, 8] : [8, 1] on ["m", "m"], replica = [] : [] on [], offset = 0>}
      : !tile.buffer
  return
}

// -----

// A single allocation may not exceed capacity either — the bound is on the
// arena, so the first allocation is checked on the same path.
func.func @rejects_single_oversized_allocation() {
  // expected-error @+1 {{APPLE_THREADGROUP_MEMORY_EXCEEDED}}
  %a = tile.alloc {space = "smem", bytes = 65536 : i64,
                   layout = #tile.layout<shard = [32, 8] : [8, 1] on ["m", "m"], replica = [] : [] on [], offset = 0>}
      : !tile.buffer
  return
}

// -----

// Metal stages through a threadgroup ping-pong pair. A deeper ring has no
// Metal realization, and quietly narrowing it to 2 would silently change the
// program's synchronization structure.
func.func @rejects_deep_stage_ring() {
  // expected-error @+1 {{APPLE_STAGE_DEPTH_UNSUPPORTED}}
  %s = tile.pipeline_init {depth = 4 : i64, stage = 0 : i64, phase = 1 : i64,
                           role = "producer"} : !tile.pipeline_state
  return
}

// -----

// Identity must travel through SSA. A name-based #tile.buffer_ref is readable
// migration metadata, not an Apple physical allocation.
func.func @rejects_name_based_buffer_identity(%arg0: tensor<64x64xf16>) {
  // expected-error @+1 {{APPLE_THREADGROUP_LEGACY_METADATA}}
  "test.staged_write"(%arg0) {tile.buffer_ref = "stage.0"} : (tensor<64x64xf16>) -> ()
  return
}

// -----

// TMA is an NVIDIA copy engine. Apple has no descriptor-driven asynchronous
// tensor movement, so the vocabulary is rejected rather than approximated.
func.func @rejects_tma_descriptor(%arg0: tensor<64x64xf16>) {
  // expected-error @+1 {{APPLE_TILE_UNSUPPORTED_VOCABULARY}}
  %d = tile.tma.descriptor %arg0 {tile_rows = 64 : i64, tile_cols = 64 : i64}
      : tensor<64x64xf16> -> !tile.tma_descriptor
  return
}

// -----

// Transaction barriers are NVIDIA-only. Metal's threadgroup_barrier is a
// different primitive and must not be silently substituted.
func.func @rejects_mbarrier() {
  // expected-error @+1 {{APPLE_TILE_UNSUPPORTED_VOCABULARY}}
  %b = tile.mbarrier.init {slots = 2 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  return
}
