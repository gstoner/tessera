// The WarpSpec-emits-the-markers join (2026-06-23): WarpSpecializationPass now
// stamps the C1/C3 markers (tile.warp_role + typed #tile.pipeline_state, with
// producer phase=1 / consumer phase=0) on the schedule.warp ops it creates, so
// the C3/C6 legality passes verify *real lowering output* instead of a
// convention. The second RUN proves WarpSpec output flows clean through the
// TilePipelineLegality (C3) + WarpSpecLegality (C6) gates.
//
// RUN: tessera-opt --tessera-warp-specialization --allow-unregistered-dialect %s | FileCheck %s
// RUN: tessera-opt --tessera-warp-specialization --tessera-tile-pipeline-legality --tessera-warpspec-legality --allow-unregistered-dialect %s | FileCheck %s --check-prefix=GATED

// The markers print on the schedule.warp region's closing-brace attr line.
// CHECK: role = "producer"
// CHECK-SAME: tile.pipeline = "warpspec.0"
// CHECK-SAME: tile.pipeline_state = #tile.pipeline_state<depth = 2, stage = 0, phase = 1, role = "producer">
// CHECK-SAME: tile.warp_role = "producer"
// CHECK: role = "consumer"
// CHECK-SAME: tile.pipeline = "warpspec.0"
// CHECK-SAME: tile.pipeline_state = #tile.pipeline_state<depth = 2, stage = 0, phase = 0, role = "consumer">
// CHECK-SAME: tile.warp_role = "consumer"

// The legality gates pass → the IR is still emitted (func survives).
// GATED: @gemm_kernel

module attributes {tessera.ir.version = "1.0"} {
  func.func @gemm_kernel(
      %A: tensor<64x64xbf16> {tessera.effect = "read"},
      %B: tensor<64x64xbf16> {tessera.effect = "read"}
  ) -> tensor<64x64xf32> {

    "schedule.mesh.region"() ({
      %tA = "tile.async_copy"(%A) {tile_rows = 64 : i64, tile_cols = 64 : i64}
              : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
      %tB = "tile.async_copy"(%B) {tile_rows = 64 : i64, tile_cols = 64 : i64}
              : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
      "tile.wait_async"() : () -> ()
      %C  = "tile.mma"(%tA, %tB) {sm = 90 : i32}
              : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xf32>
      "schedule.yield"(%C) : (tensor<64x64xf32>) -> ()
    }) {mesh = @mesh0, axis = "tp"} : () -> ()

    %zero = arith.constant 0.0 : f32
    %out  = tensor.empty() : tensor<64x64xf32>
    return %out : tensor<64x64xf32>
  }
}
