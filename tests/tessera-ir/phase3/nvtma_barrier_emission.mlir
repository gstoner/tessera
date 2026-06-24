// #tile.barrier emission (2026-06-23): NVTMADescriptorPass stamps a typed
// #tile.barrier (kind = tma, transaction byte-count `expect`) + a per-slot
// tile.barrier_id on both the setup_descriptor (init) and the copy_async
// (arrive) for each mbarrier slot. The init and arrive for one slot carry the
// SAME (kind, expect, id), so TilePipelineLegality (C3, kind consistency) and
// WarpSpecLegality (C6, arrival-count == init-count) verify them live on real
// lowering output (second RUN).
//
// RUN: tessera-opt --tessera-warp-specialization --tessera-async-copy-lowering --tessera-nvtma-descriptor --allow-unregistered-dialect %s | FileCheck %s
// RUN: tessera-opt --tessera-warp-specialization --tessera-async-copy-lowering --tessera-nvtma-descriptor --tessera-tile-pipeline-legality --tessera-warpspec-legality --allow-unregistered-dialect %s | FileCheck %s --check-prefix=GATED

// The per-slot setup (init) carries a tma barrier with the transaction byte
// count (64*64*2 = 8192) and a slot id.
// CHECK: tile.tma.setup_descriptor
// CHECK-SAME: tile.barrier = #tile.barrier<kind = "tma", expect = 8192>
// CHECK-SAME: tile.barrier_id = "mbar.0"
// The copy_async (arrive) carries the matching barrier on the same id.
// CHECK: tile.tma.copy_async
// CHECK-SAME: tile.barrier = #tile.barrier<kind = "tma", expect = 8192>

// The barriers pass the C3 + C6 gates → IR survives.
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

    %out = tensor.empty() : tensor<64x64xf32>
    return %out : tensor<64x64xf32>
  }
}
