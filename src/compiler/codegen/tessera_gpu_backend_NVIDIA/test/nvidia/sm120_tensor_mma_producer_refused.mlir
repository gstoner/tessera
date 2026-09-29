// RUN: not %tnv --lower-tile-to-nvidia=sm=120 %s 2>&1 | FileCheck %s
// CHECK: sm_120 tile.mma requires typed fragment registers and an accumulator

module attributes {tessera.metadata_snapshot = {entry = {layout = ["\22row_major\22", 1]}}} {
  func.func @entry(%arg0: tensor<64x16xbf16>, %arg1: tensor<16x256xbf16>) -> tensor<64x256xf32> attributes {tessera.effect = "pure", tessera.lowering.dropped = {layout = "re_expressed"}} {
    %0 = tile.tma.descriptor %arg0 {expect_tx = 8192 : i64, slot = 0 : i64, source_shape = array<i64: 64, 16>, tile.barrier = #tile.barrier<kind = "tma", expect = 8192>, tile.barrier_id = "mbar.0", tile_cols = 64 : i64, tile_rows = 64 : i64} : tensor<64x16xbf16> -> !tile.tma_descriptor
    %1 = tile.tma.descriptor %arg1 {expect_tx = 8192 : i64, slot = 1 : i64, source_shape = array<i64: 16, 256>, tile.barrier = #tile.barrier<kind = "tma", expect = 8192>, tile.barrier_id = "mbar.1", tile_cols = 64 : i64, tile_rows = 64 : i64} : tensor<16x256xbf16> -> !tile.tma_descriptor
    %2 = tile.role {kind = "producer", members = ["async_copy"], name = "warpspec.0.producer"} : !tile.role
    %3 = tile.role {kind = "consumer", members = ["mma"], name = "warpspec.0.consumer"} : !tile.role
    %4 = tile.mbarrier.init with %2, %3 {phase_bits = 2 : i64, slots = 2 : i64, tile.pipeline = "warpspec.0"} : !tile.mbarrier
    %5:2 = tile.tma.copy_async %0, %4 {coordinate_count = 0 : i64, expect_tx = 8192 : i64, mbarrier_slot = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 0>, tile.barrier = #tile.barrier<kind = "tma", expect = 8192>, tile.barrier_id = "mbar.0"} : (!tile.tma_descriptor, !tile.mbarrier) -> (tensor<64x16xbf16>, !tile.async_token)
    %6:2 = tile.tma.copy_async %1, %4 {coordinate_count = 0 : i64, expect_tx = 8192 : i64, mbarrier_slot = 1 : i64, operandSegmentSizes = array<i32: 1, 1, 0>, tile.barrier = #tile.barrier<kind = "tma", expect = 8192>, tile.barrier_id = "mbar.1"} : (!tile.tma_descriptor, !tile.mbarrier) -> (tensor<16x256xbf16>, !tile.async_token)
    tile.mbarrier.wait %4, %5#1, %6#1 {operandSegmentSizes = array<i32: 1, 0, 2>, slot = 0 : i64} : !tile.mbarrier, !tile.async_token, !tile.async_token
    %7 = tile.mma %5#0, %6#0, %5#1, %6#1 {sm = 120 : i32} : (tensor<64x16xbf16>, tensor<16x256xbf16>, !tile.async_token, !tile.async_token) -> tensor<64x256xf32>
    return %7 : tensor<64x256xf32>
  }
}
