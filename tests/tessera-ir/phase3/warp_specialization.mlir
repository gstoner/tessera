// Phase 3 WarpSpecializationPass lit tests.
// Verifies that tile IR ops inside a schedule.mesh.region are split into
// producer (tile.async_copy) and consumer (tile.mma) warp regions, with the
// cross-boundary values threaded as warp-region results (valid SSA).
//
// 2026-06: un-XFAIL'd.  The pass now produces *valid* IR — each schedule.warp
// region yields the values consumed across the boundary so they dominate the
// sibling region (the old version left dangling cross-region SSA refs, which
// segfaulted downstream).  Output uses the unregistered schedule.*/tile.*
// dialects (--allow-unregistered-dialect → generic printing → sym_name match);
// the warp `role` attr prints on the region's closing brace (plain CHECK).
//
// RUN: tessera-opt --tessera-warp-specialization --allow-unregistered-dialect \
// RUN:   --verify-each=false %s | FileCheck %s

// CHECK: @gemm_kernel
// CHECK: schedule.warp
// CHECK: tile.async_copy
// CHECK: schedule.yield
// CHECK: role = "producer"
// CHECK: schedule.warp
// CHECK: tile.mma
// CHECK: role = "consumer"

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
