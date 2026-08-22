// RUN: tessera-opt --tessera-warp-specialization --tessera-async-copy-lowering --tessera-nvtma-descriptor --allow-unregistered-dialect %s | FileCheck %s

// Each schedule pipeline owns a distinct role-bearing barrier. Descriptor slots
// are local to that barrier, so both one-copy pipelines correctly use slot 0.
// CHECK: %[[B0:[^ ]+]] = tile.mbarrier.init with %{{[^,]+}}, %{{[^ ]+}}
// CHECK-SAME: slots = 1
// CHECK-SAME: tile.pipeline = "warpspec.0"
// CHECK: tile.tma.copy_async %{{.*}}, %[[B0]]
// CHECK-SAME: mbarrier_slot = 0
// CHECK: tile.mbarrier.wait %[[B0]]
// CHECK: %[[B1:[^ ]+]] = tile.mbarrier.init with %{{[^,]+}}, %{{[^ ]+}}
// CHECK-SAME: slots = 1
// CHECK-SAME: tile.pipeline = "warpspec.1"
// CHECK: tile.tma.copy_async %{{.*}}, %[[B1]]
// CHECK-SAME: mbarrier_slot = 0
// CHECK: tile.mbarrier.wait %[[B1]]

module attributes {tessera.ir.version = "1.0"} {
  func.func @two_pipelines(%a: tensor<32x32xbf16>,
                           %b: tensor<32x32xbf16>) {
    "schedule.mesh.region"() ({
      %ta = "tile.async_copy"(%a) {tile_rows = 32 : i64, tile_cols = 32 : i64}
          : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
      "tile.wait_async"() : () -> ()
      %ca = "tile.mma"(%ta, %ta) {sm = 90 : i32}
          : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xf32>
      "schedule.yield"() : () -> ()
    }) {mesh = @mesh0, axis = "tp"} : () -> ()
    "schedule.mesh.region"() ({
      %tb = "tile.async_copy"(%b) {tile_rows = 32 : i64, tile_cols = 32 : i64}
          : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
      "tile.wait_async"() : () -> ()
      %cb = "tile.mma"(%tb, %tb) {sm = 90 : i32}
          : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xf32>
      "schedule.yield"() : () -> ()
    }) {mesh = @mesh0, axis = "tp"} : () -> ()
    return
  }
}
