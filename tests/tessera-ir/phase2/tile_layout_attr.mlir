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

// The typed #tile.buffer_ref contract (name + space + access) replaces the old
// tile.buffer/tile.access string markers.
// CHECK-LABEL: func.func @buffer_ref
func.func @buffer_ref() {
  // CHECK: #tile.buffer_ref<name = "warpspec.0.smem.0", space = "smem", access = "write">
  "test.buf"() {b = #tile.buffer_ref<name = "warpspec.0.smem.0", space = "smem", access = "write">} : () -> ()
  // CHECK: #tile.buffer_ref<name = "acc", space = "tmem", access = "free">
  "test.buf"() {b = #tile.buffer_ref<name = "acc", space = "tmem", access = "free">} : () -> ()
  // AMD is first-class: LDS (Local Data Share) is a named memory space.
  // CHECK: #tile.buffer_ref<name = "kv", space = "lds", access = "write">
  "test.buf"() {b = #tile.buffer_ref<name = "kv", space = "lds", access = "write">} : () -> ()
  return
}

// Backend-neutral layout vocabulary — an AMD tile placed on LDS + wave axes,
// exactly as the NVIDIA fragment above uses smem/warp (neither is privileged).
// CHECK-LABEL: func.func @amd_lds_layout
func.func @amd_lds_layout() {
  // CHECK: #tile.layout<shard = [64, 32] : [32, 1] on ["lds", "waveid"], replica = [] : [] on [], offset = 0>
  "test.buf"() {frag = #tile.layout<shard = [64, 32] : [32, 1] on ["lds", "waveid"], replica = [] : [] on [], offset = 0>} : () -> ()
  return
}

// Tuple-valued composition stays structured: the outer coordinate has two
// modes; each basis component is a `[shape_tree, stride_tree]` pair.  The
// second outer shape mode is dynamic and therefore must remain `-1` here.
// CHECK-LABEL: func.func @composed_layout_dynamic_tuple_basis
func.func @composed_layout_dynamic_tuple_basis() {
  // CHECK: #tile.composed_layout<{{\[\[6, -1\], 2\], \[\[8, 2\], 1\], \[\[\[3, 4\], \[1, 3\]\], \[\[4\], \[12\]\], \[\[2\], \[1\]\]\], \[2, 0, 1\]}}>
  "test.buf"() {l = #tile.composed_layout<[[6, -1], 2], [[8, 2], 1], [[[3, 4], [1, 3]], [[4], [12]], [[2], [1]]], [2, 0, 1]>} : () -> ()
  return
}

// CHECK-LABEL: func.func @materialize_composed_layout
func.func @materialize_composed_layout(%r: i64, %c: i64) {
  %0 = "tile.materialize_composed_layout"(%r, %c) {layout = #tile.composed_layout<[16, 16], [16, 1], [[[16], [1]], [[16], [1]]], [0, 0]>} : (i64, i64) -> i64
  return
}

// CHECK-LABEL: func.func @materialize_composed_layout_tuple_basis
func.func @materialize_composed_layout_tuple_basis(%c: i64) {
  %0 = "tile.materialize_composed_layout"(%c) {layout = #tile.composed_layout<[8], [1], [[[2, 4], [1, 2]]], [0]>} : (i64) -> i64
  return
}

// CHECK-LABEL: func.func @materialize_composed_layout_tuple_codomain
func.func @materialize_composed_layout_tuple_codomain(%c: i64) -> (i64, i64) {
  %pair:2 = "tile.materialize_composed_layout_tuple"(%c) {layouts = [#tile.composed_layout<[8], [1], [[[2, 4], [1, 2]]], [0]>, #tile.composed_layout<[8], [2], [[[4, 2], [1, 4]]], [3]>]} : (i64) -> (i64, i64)
  return %pair#0, %pair#1 : i64, i64
}

// -----

// Runtime leaves follow coordinates in canonical preorder: outer shape, then
// outer stride, then basis shape/stride.  Nested outer tuples preserve their
// tree in the carrier while the scalar affine map remains materializable.
func.func @materialize_composed_layout_dynamic_nested(%r: i64, %c: i64,
                                                       %m: i64, %lda: i64) {
  %0 = "tile.materialize_composed_layout"(%r, %c, %m, %lda) {layout = #tile.composed_layout<[[-1], [16]], [[-1], [1]], [[[16], [1]], [[16], [1]]], [0, 0]>} : (i64, i64, i64, i64) -> i64
  return
}

// -----

func.func @materialize_composed_layout_dynamic_missing_leaf(%r: i64, %c: i64,
                                                             %m: i64) {
  // expected-error @+1 {{TILE_COMPOSED_LAYOUT_NOT_MATERIALIZABLE}}
  %0 = "tile.materialize_composed_layout"(%r, %c, %m) {layout = #tile.composed_layout<[[-1], [16]], [[-1], [1]], [[[16], [1]], [[16], [1]]], [0, 0]>} : (i64, i64, i64) -> i64
  return
}

// -----

func.func @materialize_composed_layout_dynamic_rejected(%a: i64, %b: i64,
                                                        %c: i64) {
  // expected-error @+1 {{TILE_COMPOSED_LAYOUT_NOT_MATERIALIZABLE}}
  %0 = "tile.materialize_composed_layout"(%a, %b, %c) {layout = #tile.composed_layout<[[6, -1], 2], [[8, 2], 1], [[[3, 4], [1, 3]], [[4], [12]], [[2], [1]]], [2, 0, 1]>} : (i64, i64, i64) -> i64
  return
}

// -----

func.func @bad_composed_basis_rank() {
  // expected-error @+1 {{TILE_COMPOSED_LAYOUT_BASIS_RANK}}
  "test.buf"() {l = #tile.composed_layout<[[6, -1], 2], [[8, 2], 1], [[[3, 4], [1, 3]]], [2, 0, 1]>} : () -> ()
  return
}

// -----

func.func @bad_composed_profile() {
  // expected-error @+1 {{TILE_COMPOSED_LAYOUT_PROFILE_MISMATCH}}
  "test.buf"() {l = #tile.composed_layout<[[6, 2], 2], [[8], 1], [[[3, 4], [1, 3]], [[4], [12]], [[2], [1]]], [2, 0, 1]>} : () -> ()
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

// -----

func.func @bad_buffer_space() {
  // expected-error @+1 {{TILE_BUFFER_REF_BAD_SPACE}}
  "test.buf"() {b = #tile.buffer_ref<name = "x", space = "bogus", access = "write">} : () -> ()
  return
}

// -----

func.func @bad_buffer_access() {
  // expected-error @+1 {{TILE_BUFFER_REF_BAD_ACCESS}}
  "test.buf"() {b = #tile.buffer_ref<name = "x", space = "smem", access = "bogus">} : () -> ()
  return
}

// -----

func.func @bad_buffer_empty_name() {
  // expected-error @+1 {{TILE_BUFFER_REF_EMPTY_NAME}}
  "test.buf"() {b = #tile.buffer_ref<name = "", space = "smem", access = "write">} : () -> ()
  return
}
