// RUN: tessera-opt --allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s
//
// C1 (2026-06-23, TIRx review / COMPILER_AUDIT item C1): the structured
// TileLayout algebra — `S[(extents):(strides on axes)] (+ R[..]) + offset`
// with a SEPARATE `#tile.swizzle` composition — replacing the flat
// `tessera.layout` string enum. Round-trip + verifier (parallel-array lengths,
// positive extents, known hardware axes).

// CHECK-LABEL: func.func @frag_lane_reg
func.func @frag_lane_reg() {
  // A tensor-core register fragment: logical rows/cols distributed across lane
  // and register axes, no replica, no swizzle.
  // CHECK: #tile.layout<shard = [8, 4, 2] : [4, 1, 1] on ["laneid", "laneid", "reg"], replica = [] : [] on [], offset = 0>
  "test.buf"() {frag = #tile.layout<shard = [8, 4, 2] : [4, 1, 1] on ["laneid", "laneid", "reg"], replica = [] : [] on [], offset = 0>} : () -> ()
  return
}

// CHECK-LABEL: func.func @smem_swizzled
func.func @smem_swizzled() {
  // A shared-memory tile placed on the linear `m` axis with an XOR swizzle
  // composed on top (held as a separate attribute, not folded into strides).
  // CHECK: #tile.layout<shard = [64, 64] : [64, 1] on ["m", "m"], replica = [] : [] on [], offset = 0, swizzle = #tile.swizzle<per_element = 4, len = 3, atom = 8>>
  "test.buf"() {smem = #tile.layout<shard = [64, 64] : [64, 1] on ["m", "m"], replica = [] : [] on [], offset = 0, swizzle = #tile.swizzle<per_element = 4, len = 3, atom = 8>>} : () -> ()
  return
}

// CHECK-LABEL: func.func @tmem_replicated_scale
func.func @tmem_replicated_scale() {
  // A scale factor on TMEM lanes, broadcast (replicated) across warpgroups —
  // the one-to-many `R[..]` term the flat string enum cannot express.
  // CHECK: replica = [4] : [32] on ["tlane"]
  "test.buf"() {scale = #tile.layout<shard = [32] : [1] on ["tlane"], replica = [4] : [32] on ["tlane"], offset = 0>} : () -> ()
  return
}

// -----

func.func @bad_unknown_axis() {
  // expected-error @+1 {{TILE_LAYOUT_UNKNOWN_AXIS}}
  "test.buf"() {l = #tile.layout<shard = [8] : [1] on ["bogus"], replica = [] : [] on [], offset = 0>} : () -> ()
  return
}

// -----

func.func @bad_rank_mismatch() {
  // expected-error @+1 {{TILE_LAYOUT_RANK_MISMATCH}}
  "test.buf"() {l = #tile.layout<shard = [8, 4] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : () -> ()
  return
}

// -----

func.func @bad_nonpositive_extent() {
  // expected-error @+1 {{TILE_LAYOUT_NONPOSITIVE_EXTENT}}
  "test.buf"() {l = #tile.layout<shard = [0] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : () -> ()
  return
}
