// REQUIRES: tessera-apple-backend
//
// RUN: not tessera-opt %s --allow-unregistered-dialect \
// RUN:   --tessera-tile-ir-lowering='tile-q=64 tile-kv=64 sm=90' \
// RUN:   --tessera-apple-streaming-attention 2>&1 | FileCheck %s

// APPLE-ATTN-STREAM-1 (2026-07-26) — the blocking finding, pinned.
//
// `tessera-apple-streaming-attention` recognizes the shared KV-block
// recurrence (`CORE-STREAMING-ATTN-2026-07-26`) and can re-form it as one Apple
// flash-attention dispatch, carrying causal / sliding-window / logical-length
// semantics read off `tessera_attn.boundary_mask` rather than re-deriving them.
//
// It cannot yet do so for any real program, and this fixture records exactly
// why. The shared lowering always terminates the recurrence with
// `tessera_attn.lse_accumulate` -> `tessera_attn.lse.save`, the forward-side
// half of the LSE checkpoint that `tessera_attn.lse.load` reads during
// backward. `LseSaveOp` carries no `Pure` trait, so it is a real side effect,
// not dead code. Apple's fused flash-attention ABI
// (`tessera_apple_gpu_flash_attn_*`) returns the attention output only.
//
// Re-forming anyway would leave the entire recurrence alive to recompute the
// LSE — the program would do attention twice and the "fused" dispatch would be
// pure overhead — or would silently drop a checkpoint backward depends on. The
// pass refuses instead, and the refusal is the honest current state.
//
// Unblocking needs one of: an Apple fused ABI that also returns per-row LSE, or
// a shared lowering that elides `lse.save` when the LSE is provably dead.

// CHECK: APPLE_STREAMING_ATTN_LSE_UNSUPPORTED
// CHECK-SAME: returns the attention output only

module attributes {tessera.ir.version = "1.0",
                   tessera.target = {sm = 90 : i32, warps = 4 : i32,
                                     smem = 233472 : i64,
                                     pipeline_stages = 2 : i32}} {
  func.func @attn(%Q: tensor<64x64xbf16>, %K: tensor<130x64xbf16>,
                  %V: tensor<130x64xbf16>) -> tensor<64x64xf32> {
    %out = "tessera.flash_attn"(%Q, %K, %V)
        <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {
      causal = true, head_dim = 64 : i64,
      tessera.tile_q = 64 : i32, tessera.tile_kv = 64 : i32
    } : (tensor<64x64xbf16>, tensor<130x64xbf16>, tensor<130x64xbf16>)
          -> tensor<64x64xf32>
    return %out : tensor<64x64xf32>
  }
}
