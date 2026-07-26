// Buffer-marker emission (2026-06-23): WarpSpecializationPass stamps C1
// #tile.layout on SSA allocations and staged-buffer writes; physical identity
// flows only through !tile.buffer operands
// — each tile.async_copy
// stages into its own shared-memory tile (row-major, `m` axis, space=smem) and
// each tile.mma writes its accumulator to a TMEM tile (tlane/tcol, space=tmem).
// Distinct buffers ⇒ TileBarrierReuseLegality (C2) runs live and clean; the
// dealloc epilogue (cta_sync + per-buffer SSA deallocs) satisfies the C6
// use-after-free invariant (second RUN).
//
// RUN: tessera-opt --tessera-warp-specialization --allow-unregistered-dialect %s | FileCheck %s
// RUN: tessera-opt --tessera-warp-specialization --tessera-tile-barrier-reuse-legality --tessera-warpspec-legality --allow-unregistered-dialect %s | FileCheck %s --check-prefix=C2

// The allocations are the authoritative buffer identities; no name-based
// #tile.buffer_ref compatibility metadata is emitted.
// CHECK-DAG: %[[SMEMA:.*]] = tile.alloc{{.*}}space = "smem"
// CHECK-DAG: %[[SMEMB:.*]] = tile.alloc{{.*}}space = "smem"
// CHECK-DAG: %[[TMEM:.*]] = tile.alloc{{.*}}space = "tmem"

// Producer staging writes — distinct shared-memory buffers.
// CHECK: tile.async_copy
// CHECK-SAME: tile.layout = #tile.layout<shard = [64, 64] : [64, 1] on ["m", "m"], replica = [] : [] on [], offset = 0>
// CHECK: tile.async_copy

// Consumer accumulator write — a TMEM tile on the tlane/tcol axes.
// CHECK: tile.mma
// CHECK-SAME: tile.layout = #tile.layout<shard = [64, 64] : [64, 1] on ["tlane", "tcol"], replica = [] : [] on [], offset = 0>
// CHECK-NOT: tile.buf = #tile.buffer_ref

// Writeback-dealloc epilogue: a cta_sync precedes the per-buffer SSA deallocs,
// so the C6 use-after-free invariant is satisfied.
// CHECK: tile.cta_sync
// CHECK: tile.dealloc %[[SMEMA]]
// CHECK: tile.dealloc %[[SMEMB]]
// CHECK: tile.dealloc %[[TMEM]]

// C2 (reuse) + C6 (use-after-free) run clean on the real output; IR survives.
// C2: @gemm_kernel

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
