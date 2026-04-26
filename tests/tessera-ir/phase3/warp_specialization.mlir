// Phase 3 WarpSpecializationPass lit tests.
// Verifies that tile IR ops inside a schedule.mesh.region are split into
// producer (tile.async_copy) and consumer (tile.mma / tessera.attn.*) regions.
//
// RUN: tessera-opt --tessera-warp-specialization %s | FileCheck %s

// CHECK-LABEL: func.func @gemm_kernel
// CHECK:       tessera.schedule.warp
// CHECK-SAME:  role = "producer"
// CHECK:       tile.async_copy
// CHECK:       tessera.queue.push
// CHECK:       tessera.schedule.warp
// CHECK-SAME:  role = "consumer"
// CHECK:       tessera.queue.pop
// CHECK:       tile.mma

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
