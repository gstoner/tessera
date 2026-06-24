// Phase C-NV — the !tile.async_token completion edge converges on the NV path.
//
// 1. WarpSpecialization threads the token across the producer/consumer boundary:
//    a producer tile.async_copy mints it, and a consumer tile.mma that consumes
//    it has its operand rewired to the producer warp's token *result* (the same
//    generic cross-region SSA threading the data tiles use). schedule.warp /
//    schedule.yield tolerate the !tile.async_token type.
// 2. AsyncCopyLowering carries the token through TMA lowering (SM>=90), so the
//    consuming wait/mma/yield operands stay valid SSA after the copy is lowered.
//
// This is the NV analogue of the ROCm token edge (the dependency is an SSA
// def-use, not a tile.barrier_id string + program order).
//
// RUN: tessera-opt --tessera-warp-specialization --allow-unregistered-dialect \
// RUN:   %s | FileCheck %s --check-prefix=WARP
//
// RUN: tessera-opt --tessera-warp-specialization \
// RUN:   --tessera-async-copy-lowering='sm=90' --allow-unregistered-dialect \
// RUN:   %s | FileCheck %s --check-prefix=LOWER

// The producer warp region yields the token (a !tile.async_token warp result);
// the consumer mma reads that producer-warp result — the cross-region edge.
// WARP: %[[PROD:.*]]:3 = "schedule.warp"
// WARP:   tile.async_copy
// WARP-SAME: -> (tensor<64x64xbf16>, !tile.async_token)
// WARP:   schedule.yield
// WARP-SAME: !tile.async_token
// WARP: role = "producer"
// WARP-SAME: -> (tensor<64x64xbf16>, tensor<64x64xbf16>, !tile.async_token)
// WARP: "schedule.warp"
// WARP:   tile.mma %[[PROD]]#0, %[[PROD]]#1, %[[PROD]]#2
// WARP-SAME: !tile.async_token
// WARP: role = "consumer"

// After TMA lowering the token rides tile.tma.copy_async (the mbarrier still
// carries the byte count); the consumer mma still consumes it.
// LOWER: tile.tma.copy_async
// LOWER-SAME: -> (tensor<64x64xbf16>, !tile.async_token)
// LOWER: tile.mma
// LOWER-SAME: !tile.async_token

module attributes {tessera.ir.version = "1.0"} {
  func.func @gemm(%A: tensor<64x64xbf16>, %B: tensor<64x64xbf16>)
      -> tensor<64x64xf32> {
    "schedule.mesh.region"() ({
      %tA, %tokA = "tile.async_copy"(%A) {tile_rows = 64 : i64, tile_cols = 64 : i64}
              : (tensor<64x64xbf16>) -> (tensor<64x64xbf16>, !tile.async_token)
      %tB = "tile.async_copy"(%B) {tile_rows = 64 : i64, tile_cols = 64 : i64}
              : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
      "tile.wait_async"(%tokA) : (!tile.async_token) -> ()
      %C  = "tile.mma"(%tA, %tB, %tokA) {sm = 90 : i32}
              : (tensor<64x64xbf16>, tensor<64x64xbf16>, !tile.async_token)
                -> tensor<64x64xf32>
      "schedule.yield"(%C) : (tensor<64x64xf32>) -> ()
    }) {mesh = @mesh0, axis = "tp"} : () -> ()
    %out = tensor.empty() : tensor<64x64xf32>
    return %out : tensor<64x64xf32>
  }
}
