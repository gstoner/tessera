// RUN: tessera-opt --allow-unregistered-dialect --tessera-tile-barrier-reuse-legality -split-input-file -verify-diagnostics %s | FileCheck %s
//
// C2 (2026-06-23, TIRx review / COMPILER_AUDIT item C2): "barriers are a
// layout-reuse correctness property." Two writes to overlapping storage-axis
// (m/tlane/tcol) footprints of one buffer's #tile.layout with no intervening
// barrier are a race. The motivating case is FA-4's TMEM allocation aliased as
// an fp32 view (S/O) and an fp16 view (P) over the same bytes.

// The canonical race: a TMEM buffer written as an fp32 view then re-written as
// an fp16 view (2x column density, same bytes) with no barrier between.
// (No CHECK-LABEL — this chunk fails legality, so no IR is printed for it; the
// expected-error/note below are what verify this case.)
func.func @tmem_alias_race() {
  // expected-note @+1 {{previous write to buffer "tmem0" here}}
  "tile.tmem_write"() {tile.buf = #tile.buffer_ref<name = "tmem0", space = "tmem", access = "write">,
    tile.layout = #tile.layout<shard = [128] : [1] on ["tlane"], replica = [] : [] on [], offset = 0>} : () -> ()
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER: buffer "tmem0"}}
  "tile.tmem_write"() {tile.buf = #tile.buffer_ref<name = "tmem0", space = "tmem", access = "write">,
    tile.layout = #tile.layout<shard = [256] : [1] on ["tlane"], replica = [] : [] on [], offset = 0>} : () -> ()
  return
}

// -----

// Same two writes, but an mbarrier wait separates them — the barrier releases
// the reuse hazard, so the layout reuse is legal.
// CHECK-LABEL: func.func @tmem_alias_barriered
func.func @tmem_alias_barriered() {
  "tile.tmem_write"() {tile.buf = #tile.buffer_ref<name = "tmem0", space = "tmem", access = "write">,
    tile.layout = #tile.layout<shard = [128] : [1] on ["tlane"], replica = [] : [] on [], offset = 0>} : () -> ()
  // CHECK: tile.mbarrier_wait
  "tile.mbarrier_wait"() : () -> ()
  "tile.tmem_write"() {tile.buf = #tile.buffer_ref<name = "tmem0", space = "tmem", access = "write">,
    tile.layout = #tile.layout<shard = [256] : [1] on ["tlane"], replica = [] : [] on [], offset = 0>} : () -> ()
  return
}

// -----

// Double-buffering: two writes to the same buffer but at disjoint offsets
// (stages 0 and 1) — footprints do not overlap, so no barrier is required.
// CHECK-LABEL: func.func @double_buffer_disjoint
func.func @double_buffer_disjoint() {
  "tile.smem_write"() {tile.buf = #tile.buffer_ref<name = "smem0", space = "smem", access = "write">,
    tile.layout = #tile.layout<shard = [128] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : () -> ()
  "tile.smem_write"() {tile.buf = #tile.buffer_ref<name = "smem0", space = "smem", access = "write">,
    tile.layout = #tile.layout<shard = [128] : [1] on ["m"], replica = [] : [] on [], offset = 128>} : () -> ()
  return
}

// -----

// A pure register/lane fragment (no storage axis) touches no shared storage, so
// two writes to the "same" buffer name carry no aliasing hazard.
// CHECK-LABEL: func.func @register_fragment_no_hazard
func.func @register_fragment_no_hazard() {
  "tile.reg_write"() {tile.buf = #tile.buffer_ref<name = "frag", space = "reg", access = "write">,
    tile.layout = #tile.layout<shard = [8, 4] : [4, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>} : () -> ()
  "tile.reg_write"() {tile.buf = #tile.buffer_ref<name = "frag", space = "reg", access = "write">,
    tile.layout = #tile.layout<shard = [8, 4] : [4, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>} : () -> ()
  return
}

// -----

// ROCm is first-class: reuse of an AMD LDS buffer (the `lds` storage axis)
// without a barrier is the same race as the NVIDIA SMEM/TMEM cases above.
func.func @lds_alias_race() {
  // expected-note @+1 {{previous write to buffer "lds0" here}}
  "tile.lds_write"() {tile.buf = #tile.buffer_ref<name = "lds0", space = "lds", access = "write">,
    tile.layout = #tile.layout<shard = [128] : [1] on ["lds"], replica = [] : [] on [], offset = 0>} : () -> ()
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER: buffer "lds0"}}
  "tile.lds_write"() {tile.buf = #tile.buffer_ref<name = "lds0", space = "lds", access = "write">,
    tile.layout = #tile.layout<shard = [256] : [1] on ["lds"], replica = [] : [] on [], offset = 0>} : () -> ()
  return
}
