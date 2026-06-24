// Buffer-marker emission (2026-06-23): WarpSpecializationPass stamps C1
// #tile.layout + the typed #tile.buffer_ref (name + space + access) on the
// staged-buffer writes it moves into the warp regions — each tile.async_copy
// stages into its own shared-memory tile (row-major, `m` axis, space=smem) and
// each tile.mma writes its accumulator to a TMEM tile (tlane/tcol, space=tmem).
// Distinct buffers ⇒ TileBarrierReuseLegality (C2) runs live and clean; the
// dealloc epilogue (cta_sync + per-buffer frees, access=free) satisfies the C6
// use-after-free invariant (second RUN).
//
// RUN: tessera-opt --tessera-warp-specialization --allow-unregistered-dialect %s | FileCheck %s
// RUN: tessera-opt --tessera-warp-specialization --tessera-tile-barrier-reuse-legality --tessera-warpspec-legality --allow-unregistered-dialect %s | FileCheck %s --check-prefix=C2

// Producer staging writes — distinct shared-memory buffers (typed buffer_ref).
// CHECK: tile.async_copy
// CHECK-SAME: tile.buf = #tile.buffer_ref<name = "warpspec.0.smem.0", space = "smem", access = "write">
// CHECK-SAME: tile.layout = #tile.layout<shard = [64, 64] : [64, 1] on ["m", "m"], replica = [] : [] on [], offset = 0>
// CHECK: tile.async_copy
// CHECK-SAME: tile.buf = #tile.buffer_ref<name = "warpspec.0.smem.1", space = "smem", access = "write">

// Consumer accumulator write — a TMEM tile on the tlane/tcol axes.
// CHECK: tile.mma
// CHECK-SAME: tile.buf = #tile.buffer_ref<name = "warpspec.0.tmem.acc.0", space = "tmem", access = "write">
// CHECK-SAME: tile.layout = #tile.layout<shard = [64, 64] : [64, 1] on ["tlane", "tcol"], replica = [] : [] on [], offset = 0>

// Writeback-dealloc epilogue: a cta_sync precedes the per-buffer frees
// (access=free), so the C6 use-after-free invariant is satisfied.
// CHECK: tile.cta_sync
// CHECK: tile.buffer_free
// CHECK-SAME: name = "warpspec.0.smem.0", space = "smem", access = "free"
// CHECK: tile.buffer_free
// CHECK-SAME: name = "warpspec.0.smem.1", space = "smem", access = "free"
// CHECK: tile.buffer_free
// CHECK-SAME: name = "warpspec.0.tmem.acc.0", space = "tmem", access = "free"

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
