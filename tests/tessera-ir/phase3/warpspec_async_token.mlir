// Phase C-NV — the !tile.async_token completion edge converges on the NV path.
//
// 1. WarpSpecialization AUTO-MINTS the token: from a token-less input it reads
//    each consumer tile.mma's data operands, mints a !tile.async_token on every
//    producer tile.async_copy the mma consumes, and threads it as an explicit mma
//    operand — a first-class SSA copy→mma synchronization edge derived straight
//    from the dataflow (no program-order guess). The generic producer→consumer
//    threading then carries each token across the schedule.warp boundary (the
//    consumer mma reads the producer warp's token *result*).
// 2. AsyncCopyLowering carries the token through TMA lowering (SM>=90), so the
//    consuming mma/yield operands stay valid SSA after the copy is lowered.
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

// Both copies are auto-minted a token; the producer warp yields the two tiles
// plus the two tokens, and the consumer mma reads all four producer-warp results
// — the data edges (#0,#1) and the synthesized token edges (#2,#3).
// WARP: %[[PROD:.*]]:4 = "schedule.warp"
// WARP:   tile.async_copy
// WARP-SAME: -> (tensor<64x64xbf16>, !tile.async_token)
// WARP: role = "producer"
// WARP-SAME: -> (tensor<64x64xbf16>, tensor<64x64xbf16>, !tile.async_token, !tile.async_token)
// WARP: tile.mma %[[PROD]]#0, %[[PROD]]#1, %[[PROD]]#2, %[[PROD]]#3
// WARP-SAME: (tensor<64x64xbf16>, tensor<64x64xbf16>, !tile.async_token, !tile.async_token)
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
