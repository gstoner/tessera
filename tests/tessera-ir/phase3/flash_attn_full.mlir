// Phase 3 end-to-end FlashAttention pipeline lit test.
// Runs the full Phase 3 GPU lowering chain on a BF16 causal flash_attn.
//
// 2026-06: un-XFAIL'd after fixing the two real IR-correctness bugs that made
// the chain crash end-to-end (it had never run since the MLIR-22 bump):
//   1. WarpSpecializationPass produced SSA-invalid IR — it split ops into
//      producer/consumer schedule.warp regions without rewiring the cross-region
//      value flow.  It now yields the cross-boundary values as warp-region
//      results (so they dominate the sibling region) and rewires the uses.
//   2. AsyncCopyLoweringPass's tma.copy_async carried no result, so replaceOp
//      on the 1-result tile.async_copy was a result-count mismatch that left a
//      dangling Value (→ segfault when later folded/printed).  It now carries
//      the tile result type for a 1:1 replacement.
// The flash-attn path lowers through the attn.* online-softmax pipeline (not
// tile.mma), so there is no wgmma in the output.  --verify-each=false is still
// needed for the unregistered schedule.*/tile.* ops + a separate pre-existing
// tessera.attn.lse.save scalar-vs-tensor verifier mismatch.
//
// RUN: tessera-opt \
// RUN:   --tessera-distribution-lowering='mesh-axes=tp mesh-sizes=1' \
// RUN:   --tessera-effect-annotation \
// RUN:   --tessera-tile-ir-lowering='tile-q=64 tile-kv=64 sm=90' \
// RUN:   --tessera-warp-specialization \
// RUN:   --tessera-async-copy-lowering='sm=90' \
// RUN:   --tessera-nvwgmma-lowering='sm=90' \
// RUN:   --tessera-nvtma-descriptor \
// RUN:   --tessera-nvflash-attn-emitter='sm=90 tile-q=64 tile-kv=64 warps=4' \
// RUN:   --verify-each=false %s | FileCheck %s
//
// Alternatively, use the named pipeline:
// RUN: tessera-opt -tessera-lower-to-gpu --verify-each=false %s \
// RUN:   | FileCheck %s --check-prefix=PIPE

// The unregistered schedule.*/tile.* ops force generic printing, so match the
// generic sym_name + the lowered op sequence in emitted order.
// CHECK:          sym_name = "flash_attn_fwd"
// CHECK:          tessera.tma.setup_descriptor
// CHECK:          tessera.mbarrier.arrive.expect_tx
// CHECK:          tessera.mbarrier.try_wait.parity
// CHECK:          tessera.mbarrier.init
// CHECK:          schedule.warp
// CHECK:          tessera.tma.copy_async
// CHECK:          role = "producer"
// CHECK:          schedule.warp
// CHECK:          tessera.attn.scaled_dot_product
// CHECK:          tessera.attn.causal_mask
// CHECK:          tessera.attn.online_softmax
// CHECK:          tessera.attn.lse_accumulate
// CHECK:          tessera.attn.lse.save
// CHECK:          role = "consumer"
// CHECK:          nvvm.kernel
// CHECK-NOT:      tessera.flash_attn
// CHECK-NOT:      tile.mma

// PIPE:           sym_name = "flash_attn_fwd"
// PIPE:           tessera.tma.copy_async
// PIPE:           tessera.attn.scaled_dot_product
// PIPE:           nvvm.kernel

module attributes {tessera.ir.version = "1.0",
                   tessera.target = {sm = 90 : i32, warps = 4 : i32,
                                     smem = 233472 : i64,
                                     pipeline_stages = 2 : i32}} {
  func.func @flash_attn_fwd(
      %Q: tensor<64x64xbf16>
            {tessera.effect = "read",
             tessera.shard  = {axes = ["tp"], dims = [0], sizes = [1]}},
      %K: tensor<64x64xbf16>
            {tessera.effect = "read"},
      %V: tensor<64x64xbf16>
            {tessera.effect = "read"}
  ) -> tensor<64x64xf32> {
    %out = "tessera.flash_attn"(%Q, %K, %V) {
      causal           = true,
      head_dim         = 64 : i64,
      tessera.tile_q   = 64 : i32,
      tessera.tile_kv  = 64 : i32
    } : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>)
          -> tensor<64x64xf32>
    return %out : tensor<64x64xf32>
  }
}
