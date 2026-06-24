// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-legality)' -split-input-file -verify-diagnostics %s
//
// Per-id waitcnt legality (replaces the function-global pendingAsync bool). A
// matrix op is legal iff every barrier id it depends on has been retired; an mma
// may run while *unrelated* prefetch ids remain outstanding (double buffering).

// Valid double buffer: the mma depends on the already-waited stage b0; b1 is a
// live prefetch the mma does not consume, so it is fine to leave outstanding.
func.func @double_buffer_ok(%x: tensor<16x16xf16>) {
  "tile.async_copy"() {tile.barrier_id = "b0"} : () -> ()
  "tile.wait_async"() {tile.barrier_id = "b0"} : () -> ()
  "tile.async_copy"() {tile.barrier_id = "b1"} : () -> ()
  %c = "tile.mma"(%x, %x) {tile.depends_on = ["b0"]} : (tensor<16x16xf16>, tensor<16x16xf16>) -> tensor<16x16xf16>
  return
}

// -----

// Missing wait: the single outstanding copy is inferred as the mma's dependency
// and is unretired.
func.func @missing_wait(%x: tensor<16x16xf16>) {
  "tile.async_copy"() {tile.barrier_id = "b0"} : () -> ()
  // expected-error @+1 {{ROCM_WAVE_LDS_MISSING_WAITCNT}}
  %c = "tile.mma"(%x, %x) : (tensor<16x16xf16>, tensor<16x16xf16>) -> tensor<16x16xf16>
  return
}

// -----

// Multi-stage with no explicit tile.depends_on: the consumed stage is ambiguous.
func.func @ambiguous(%x: tensor<16x16xf16>) {
  "tile.async_copy"() {tile.barrier_id = "b0"} : () -> ()
  "tile.async_copy"() {tile.barrier_id = "b1"} : () -> ()
  // expected-error @+1 {{ROCM_WAVE_LDS_AMBIGUOUS_DEPENDENCY}}
  %c = "tile.mma"(%x, %x) : (tensor<16x16xf16>, tensor<16x16xf16>) -> tensor<16x16xf16>
  return
}

// -----

// s_barrier drains ALL outstanding ids, so the mma after it has no live deps.
func.func @s_barrier_drains(%x: tensor<16x16xf16>) {
  "tile.async_copy"() {tile.barrier_id = "b0"} : () -> ()
  "tile.async_copy"() {tile.barrier_id = "b1"} : () -> ()
  "tile.s_barrier"() {tile.barrier = #tile.barrier<kind = "s_barrier", expect = 0>} : () -> ()
  %c = "tile.mma"(%x, %x) : (tensor<16x16xf16>, tensor<16x16xf16>) -> tensor<16x16xf16>
  return
}

// -----

// NVIDIA-only name-based ops are rejected on the ROCm path (no #tile.barrier to
// discriminate on) — they no longer slip through a name.contains("barrier") sniff.
func.func @nv_only(%d: memref<64xf16>, %s: memref<64xf16>, %b: index) {
  "tile.async_copy"(%d, %s, %b) {tile.barrier_id = "b0"} : (memref<64xf16>, memref<64xf16>, index) -> i32
  // expected-error @+1 {{ROCM_WAVE_LDS_UNSUPPORTED_NV_CONSTRUCT}}
  "tile.mbarrier.try_wait"() : () -> ()
  return
}
