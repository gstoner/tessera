// RUN: tessera-opt %s | FileCheck %s

module {
  func.func @typed_async_and_sm100_contract(
      %src: tensor<64x64xbf16>,
      %lhs: !tile.fragment<m = 64, n = 64, k = 32, elem = "bf16", acc = "f32", role = "a", layout = "row_major", family = "tcgen05">,
      %rhs: !tile.fragment<m = 64, n = 64, k = 32, elem = "bf16", acc = "f32", role = "b", layout = "col_major", family = "tcgen05">) {
    %barrier = tile.mbarrier.init {
      slots = 1 : i64, phase_bits = 2 : i64
    } : !tile.mbarrier
    %descriptor = tile.tma.descriptor %src {
      tile_rows = 64 : i64, tile_cols = 64 : i64,
      slot = 0 : i64, expect_tx = 8192 : i64
    } : tensor<64x64xbf16> -> !tile.tma_descriptor
    %copy = tile.tma.copy_async %descriptor, %barrier {
      operandSegmentSizes = array<i32: 1, 1, 0>,
      mbarrier_slot = 0 : i64, expect_tx = 8192 : i64
    } : (!tile.tma_descriptor, !tile.mbarrier) -> !tile.async_token
    tile.mbarrier.wait %barrier, %copy {
      operandSegmentSizes = array<i32: 1, 0, 1>, slot = 0 : i64
    } : !tile.mbarrier, !tile.async_token
    %arrival = tile.mbarrier.arrive_expect_tx %barrier {
      slot = 0 : i64, bytes = 8192 : i64
    } : !tile.mbarrier -> !tile.mbarrier_token
    // The arrive→wait edge is expressible in the type system (CAKE Phase 1
    // §5.2 row 1): the wait consumes the typed mbarrier token.
    tile.mbarrier.wait %barrier, %arrival {
      operandSegmentSizes = array<i32: 1, 1, 0>, slot = 0 : i64
    } : !tile.mbarrier, !tile.mbarrier_token
    %ready = tile.mbarrier.try_wait %barrier, %arrival {
      slot = 0 : i64
    } : !tile.mbarrier, !tile.mbarrier_token

    %acc = tile.tmem.allocate {
      bytes = 16384 : i64, alignment = 128 : i64
    } : !tile.tmem
    %next = tile.tcgen05.mma %lhs, %rhs, %acc {
      cta_group = 2 : i64,
      mma = #tile.mma_desc<family = "tcgen05", m = 64, n = 64, k = 32,
        a = "bf16", b = "bf16", acc = "f32",
        a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : !tile.fragment<m = 64, n = 64, k = 32, elem = "bf16", acc = "f32", role = "a", layout = "row_major", family = "tcgen05">,
        !tile.fragment<m = 64, n = 64, k = 32, elem = "bf16", acc = "f32", role = "b", layout = "col_major", family = "tcgen05">,
        !tile.tmem -> !tile.tmem
    %value = tile.tmem.load %next : (!tile.tmem) -> f32
    tile.tmem.store %value, %next : f32, !tile.tmem
    return
  }
}

// CHECK: %[[BARRIER:.*]] = tile.mbarrier.init
// CHECK-SAME: !tile.mbarrier
// CHECK: %[[DESC:.*]] = tile.tma.descriptor
// CHECK-SAME: !tile.tma_descriptor
// CHECK: %[[COPY:.*]] = tile.tma.copy_async %[[DESC]], %[[BARRIER]]
// CHECK-SAME: !tile.async_token
// CHECK: tile.mbarrier.wait %[[BARRIER]], %[[COPY]]
// CHECK: %[[ARRIVAL:.*]] = tile.mbarrier.arrive_expect_tx %[[BARRIER]]
// CHECK-SAME: !tile.mbarrier_token
// CHECK: tile.mbarrier.wait %[[BARRIER]], %[[ARRIVAL]]
// CHECK-SAME: !tile.mbarrier_token
// CHECK: tile.mbarrier.try_wait %[[BARRIER]], %[[ARRIVAL]]
// CHECK: %[[ACC:.*]] = tile.tmem.allocate
// CHECK-SAME: !tile.tmem
// CHECK: tile.tcgen05.mma
// CHECK-SAME: family = "tcgen05"
