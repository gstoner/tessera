// RUN: %tessera_strict_opt --tessera-tile-barrier-reuse-legality -split-input-file -verify-diagnostics %s

func.func @loop_descriptor_pending() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %d = tile.tma.descriptor %a {tile_rows = 8 : i64, tile_cols = 8 : i64, slot = 0 : i64, expect_tx = 64 : i64} : !tile.buffer -> !tile.tma_descriptor
  %bar = tile.mbarrier.init {slots = 1 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %forwarded = scf.for %i = %c0 to %c1 step %c1 iter_args(%carried = %d) -> (!tile.tma_descriptor) {
    scf.yield %carried : !tile.tma_descriptor
  }
  // expected-note @+1 {{previous write to the same allocation root}}
  %t = tile.tma.copy_async %forwarded, %bar {operandSegmentSizes = array<i32: 1, 1, 0>, mbarrier_slot = 0 : i64, expect_tx = 64 : i64} : (!tile.tma_descriptor, !tile.mbarrier) -> !tile.async_token
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
  tile.dealloc %a : !tile.buffer
  return
}

// -----

func.func @loop_descriptor_waited() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %d = tile.tma.descriptor %a {tile_rows = 8 : i64, tile_cols = 8 : i64, slot = 0 : i64, expect_tx = 64 : i64} : !tile.buffer -> !tile.tma_descriptor
  %bar = tile.mbarrier.init {slots = 1 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %forwarded = scf.for %i = %c0 to %c1 step %c1 iter_args(%carried = %d) -> (!tile.tma_descriptor) {
    scf.yield %carried : !tile.tma_descriptor
  }
  %t = tile.tma.copy_async %forwarded, %bar {operandSegmentSizes = array<i32: 1, 1, 0>, mbarrier_slot = 0 : i64, expect_tx = 64 : i64} : (!tile.tma_descriptor, !tile.mbarrier) -> !tile.async_token
  tile.mbarrier.wait %bar, %t {operandSegmentSizes = array<i32: 1, 0, 1>, slot = 0 : i64} : !tile.mbarrier, !tile.async_token
  tile.dealloc %a : !tile.buffer
  return
}

// -----

func.func @opaque_descriptor(%forwarded: !tile.tma_descriptor, %bar: !tile.mbarrier) {
  // expected-error @+1 {{unresolved TMA descriptor lifetime}}
  %t = tile.tma.copy_async %forwarded, %bar {operandSegmentSizes = array<i32: 1, 1, 0>, mbarrier_slot = 0 : i64, expect_tx = 64 : i64} : (!tile.tma_descriptor, !tile.mbarrier) -> !tile.async_token
  return
}

// -----

func.func @iter_arg_pending() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %d = tile.tma.descriptor %a {tile_rows = 8 : i64, tile_cols = 8 : i64, slot = 0 : i64, expect_tx = 64 : i64} : !tile.buffer -> !tile.tma_descriptor
  %bar = tile.mbarrier.init {slots = 1 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %out = scf.for %i = %c0 to %c1 step %c1 iter_args(%carried = %d) -> (!tile.tma_descriptor) {
    // expected-note @+1 {{previous write to the same allocation root}}
    %t = tile.tma.copy_async %carried, %bar {operandSegmentSizes = array<i32: 1, 1, 0>, mbarrier_slot = 0 : i64, expect_tx = 64 : i64} : (!tile.tma_descriptor, !tile.mbarrier) -> !tile.async_token
    // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER}}
    tile.dealloc %a : !tile.buffer
    scf.yield %carried : !tile.tma_descriptor
  }
  return
}

// -----

func.func @iter_arg_waited() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %d = tile.tma.descriptor %a {tile_rows = 8 : i64, tile_cols = 8 : i64, slot = 0 : i64, expect_tx = 64 : i64} : !tile.buffer -> !tile.tma_descriptor
  %bar = tile.mbarrier.init {slots = 1 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %out = scf.for %i = %c0 to %c1 step %c1 iter_args(%carried = %d) -> (!tile.tma_descriptor) {
    %t = tile.tma.copy_async %carried, %bar {operandSegmentSizes = array<i32: 1, 1, 0>, mbarrier_slot = 0 : i64, expect_tx = 64 : i64} : (!tile.tma_descriptor, !tile.mbarrier) -> !tile.async_token
    tile.mbarrier.wait %bar, %t {operandSegmentSizes = array<i32: 1, 0, 1>, slot = 0 : i64} : !tile.mbarrier, !tile.async_token
    tile.dealloc %a : !tile.buffer
    scf.yield %carried : !tile.tma_descriptor
  }
  return
}
