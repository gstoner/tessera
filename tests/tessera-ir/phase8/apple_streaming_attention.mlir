// REQUIRES: tessera-apple-backend
//
// The input is a plain Graph-IR flash_attn: the *shared* TileIRLowering builds
// the streaming recurrence, and only then does Apple claim it. Driving both in
// one run proves Apple consumes the real shared form.
//
// RUN: tessera-opt %s --allow-unregistered-dialect \
// RUN:   --tessera-tile-ir-lowering='tile-q=64 tile-kv=64 sm=90' \
// RUN:   --tessera-apple-streaming-attention | FileCheck %s
//
// The same claim must hold through the real target pipeline, where this pass
// runs ahead of APPLE-PIPE-1 (see the ordering note in Passes.cpp). Piped as
// two invocations because a named pipeline cannot be combined with standalone
// pass flags in one command.
// RUN: tessera-opt %s --allow-unregistered-dialect \
// RUN:   --tessera-tile-ir-lowering='tile-q=64 tile-kv=64 sm=90' \
// RUN:   | tessera-opt --allow-unregistered-dialect \
// RUN:       --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu)' \
// RUN:   | FileCheck %s

// APPLE-ATTN-STREAM-1 (2026-07-26). `CORE-STREAMING-ATTN-2026-07-26` states
// rank-2 FlashAttention as a KV-block `scf.for` carrying the FP32 accumulator,
// running max/sum, producer/consumer `!tile.pipeline_state`, and an absolute
// boundary offset. Apple re-forms that recurrence as one fused dispatch: the
// loop is a semantic contract, and Metal — which stages through a ping-pong
// pair and has no async copy engine — cannot realize the depth-3 ring as
// written.
//
// The decisive property is that causal / sliding-window / logical-length
// semantics are READ OFF `tessera_attn.boundary_mask` rather than re-derived.
// Before this pass, `FlashAttnToAppleGPU` rewrote `tessera.flash_attn` straight
// from Graph IR and re-implemented those boundary rules independently.

// The recurrence and all of its staging are consumed, not left beside the
// dispatch: a leftover depth-3 ring would fail APPLE-PIPE-1 for a schedule the
// program no longer has.
// CHECK-LABEL: func.func @attn
// CHECK-NOT: scf.for
// CHECK-NOT: tile.pipeline_init
// CHECK-NOT: tile.async_copy
// CHECK-NOT: tessera_attn.
// CHECK: tessera_apple.gpu.kernel_call
// CHECK-SAME: op_kind = "flash_attn"
// CHECK-SAME: symbol = "tessera_apple_gpu_flash_attn_bf16"
// Boundary semantics carried from the shared ops, not re-derived.
// CHECK-SAME: tessera_apple.causal = true
// CHECK-SAME: tessera_apple.head_dim = 64
// CHECK-SAME: tessera_apple.kv_block = 64
// CHECK-SAME: tessera_apple.logical_sk = 130
// CHECK-SAME: tessera_apple.streaming_recurrence = true
// CHECK-SAME: tessera_apple.window_left = -1
// CHECK-SAME: tessera_apple.window_right = -1
// CHECK-NOT: scf.for

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
