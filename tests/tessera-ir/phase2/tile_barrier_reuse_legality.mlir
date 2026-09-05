// RUN: %tessera_strict_opt --tessera-tile-dataflow-legality -split-input-file -verify-diagnostics %s
// RUN: %tessera_strict_opt --tessera-tile-barrier-reuse-legality -split-input-file -verify-diagnostics %s | FileCheck %s
//
// C2 (2026-06-23, TIRx review / COMPILER_AUDIT item C2): "barriers are a
// layout-reuse correctness property." Two writes to overlapping storage-axis
// (m/lds/tlane/tcol) footprints of one buffer's #tile.layout with no
// intervening barrier are a race. The motivating case is FA-4's TMEM
// allocation aliased as an fp32 view (S/O) and an fp16 view (P) over the same
// bytes.
//
// Ported to the REGISTERED vocabulary (P1a second increment, CAKE §5.4,
// 2026-08-15): no `--allow-unregistered-dialect`. The old
// smem/tmem/lds/reg_write husk spellings are one registered op —
// `tile.buffer_write` (the space comes from the alloc, not the op name) — and
// the release marker is the registered `tile.wait_async`, recognized by typed
// identity rather than name matching.

// The canonical race: a TMEM buffer written as an fp32 view then re-written as
// an fp16 view (2x column density, same bytes) with no barrier between.
// (No CHECK-LABEL — this chunk fails legality, so no IR is printed for it; the
// expected-error/note below are what verify this case.)
func.func @tmem_alias_race() {
  %buffer = tile.alloc {bytes = 1024 : i64, space = "tmem",
    layout = #tile.layout<shard = [256] : [1] on ["tlane"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  // expected-note @+1 {{previous write to the same allocation root here}}
  tile.buffer_write %buffer {
    tile.layout = #tile.layout<shard = [128] : [1] on ["tlane"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER: buffer SSA allocation}}
  tile.buffer_write %buffer {
    tile.layout = #tile.layout<shard = [256] : [1] on ["tlane"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.cta_sync
  tile.dealloc %buffer : !tile.buffer
  return
}

// -----

// Same two writes, but a wait separates them — the barrier releases the reuse
// hazard, so the layout reuse is legal.
// CHECK-LABEL: func.func @tmem_alias_barriered
func.func @tmem_alias_barriered() {
  %buffer = tile.alloc {bytes = 1024 : i64, space = "tmem",
    layout = #tile.layout<shard = [256] : [1] on ["tlane"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.buffer_write %buffer {
    tile.layout = #tile.layout<shard = [128] : [1] on ["tlane"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  // CHECK: tile.wait_async
  tile.wait_async : () -> ()
  tile.buffer_write %buffer {
    tile.layout = #tile.layout<shard = [256] : [1] on ["tlane"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.cta_sync
  tile.dealloc %buffer : !tile.buffer
  return
}

// -----

// Double-buffering: two writes to the same buffer but at disjoint offsets
// (stages 0 and 1) — footprints do not overlap, so no barrier is required.
// CHECK-LABEL: func.func @double_buffer_disjoint
func.func @double_buffer_disjoint() {
  %buffer = tile.alloc {bytes = 1024 : i64, space = "smem",
    layout = #tile.layout<shard = [256] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.buffer_write %buffer {
    tile.layout = #tile.layout<shard = [128] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.buffer_write %buffer {
    tile.layout = #tile.layout<shard = [128] : [1] on ["m"], replica = [] : [] on [], offset = 128>} : !tile.buffer
  tile.cta_sync
  tile.dealloc %buffer : !tile.buffer
  return
}

// -----

// A pure register/lane fragment (no storage axis) touches no shared storage, so
// two writes to the same allocation carry no aliasing hazard.
// CHECK-LABEL: func.func @register_fragment_no_hazard
func.func @register_fragment_no_hazard() {
  %buffer = tile.alloc {bytes = 128 : i64, space = "gmem",
    layout = #tile.layout<shard = [8, 4] : [4, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.buffer_write %buffer {
    tile.layout = #tile.layout<shard = [8, 4] : [4, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.buffer_write %buffer {
    tile.layout = #tile.layout<shard = [8, 4] : [4, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.cta_sync
  tile.dealloc %buffer : !tile.buffer
  return
}

// -----

// ROCm is first-class: reuse of an AMD LDS buffer (the `lds` storage axis)
// without a barrier is the same race as the NVIDIA SMEM/TMEM cases above.
func.func @lds_alias_race() {
  %buffer = tile.alloc {bytes = 1024 : i64, space = "smem",
    layout = #tile.layout<shard = [256] : [1] on ["lds"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  // expected-note @+1 {{previous write to the same allocation root here}}
  tile.buffer_write %buffer {
    tile.layout = #tile.layout<shard = [128] : [1] on ["lds"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER: buffer SSA allocation}}
  tile.buffer_write %buffer {
    tile.layout = #tile.layout<shard = [256] : [1] on ["lds"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.cta_sync
  tile.dealloc %buffer : !tile.buffer
  return
}

// -----

// Physical identity comes from the tile.alloc SSA result, not a buffer name;
// the legality pass follows the common allocation root through the registered
// write op.
func.func @ssa_allocation_alias_race() {
  %buffer = tile.alloc {
    bytes = 256 : i64, space = "smem",
    layout = #tile.layout<shard = [256] : [1] on ["m"],
                          replica = [] : [] on [], offset = 0>
  } : !tile.buffer
  // expected-note @+1 {{previous write to the same allocation root here}}
  tile.buffer_write %buffer {
    tile.layout = #tile.layout<shard = [128] : [1] on ["m"],
                               replica = [] : [] on [], offset = 0>
  } : !tile.buffer
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER: buffer SSA allocation}}
  tile.buffer_write %buffer {
    tile.layout = #tile.layout<shard = [128] : [1] on ["m"],
                               replica = [] : [] on [], offset = 0>
  } : !tile.buffer
  tile.cta_sync
  tile.dealloc %buffer : !tile.buffer
  return
}

// -----

// A typed barrier still clears hazards for the SSA allocation form.
// CHECK-LABEL: func.func @ssa_allocation_barriered
func.func @ssa_allocation_barriered() {
  %buffer = tile.alloc {
    bytes = 256 : i64, space = "smem",
    layout = #tile.layout<shard = [256] : [1] on ["m"],
                          replica = [] : [] on [], offset = 0>
  } : !tile.buffer
  tile.buffer_write %buffer {
    tile.layout = #tile.layout<shard = [128] : [1] on ["m"],
                               replica = [] : [] on [], offset = 0>
  } : !tile.buffer
  tile.wait_async : () -> ()
  tile.buffer_write %buffer {
    tile.layout = #tile.layout<shard = [128] : [1] on ["m"],
                               replica = [] : [] on [], offset = 0>
  } : !tile.buffer
  tile.cta_sync
  tile.dealloc %buffer : !tile.buffer
  return
}

// -----

// A barrier attribute describes policy; it cannot turn a non-wait into a
// completing operation. In particular a decorated poll must not release reuse.
func.func @barrier_attribute_is_not_completion() {
  %buffer = tile.alloc {bytes = 1024 : i64, space = "smem",
    layout = #tile.layout<shard = [256] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %bar = tile.mbarrier.init {slots = 1 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  %token = tile.mbarrier.arrive_expect_tx %bar {slot = 0 : i64, bytes = 128 : i64} : !tile.mbarrier -> !tile.mbarrier_token
  // expected-note @+1 {{previous write to the same allocation root here}}
  tile.buffer_write %buffer {
    tile.layout = #tile.layout<shard = [128] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %ready = tile.mbarrier.try_wait %bar, %token {slot = 0 : i64, tile.barrier = #tile.barrier<kind = "mbarrier", expect = 1>} : !tile.mbarrier, !tile.mbarrier_token
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
  tile.buffer_write %buffer {
    tile.layout = #tile.layout<shard = [128] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  tile.cta_sync
  tile.dealloc %buffer : !tile.buffer
  return
}
