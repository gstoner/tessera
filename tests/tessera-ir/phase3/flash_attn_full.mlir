// Phase 3 end-to-end FlashAttention pipeline lit test.
// Runs the full Phase 3 GPU lowering chain on a BF16 causal flash_attn.
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
// RUN:   %s | FileCheck %s
//
// Alternatively, use the named pipeline:
// RUN: tessera-opt -tessera-lower-to-gpu %s | FileCheck %s --check-prefix=PIPE

// CHECK-LABEL:    func.func @flash_attn_fwd
// CHECK-SAME:     nvvm.kernel
// CHECK-SAME:     tessera.tile_q = 64
// CHECK:          tessera.mbarrier.init
// CHECK:          tessera.tma.setup_descriptor
// CHECK:          tessera.mbarrier.arrive.expect_tx
// CHECK:          tessera.mbarrier.try_wait.parity
// CHECK:          schedule.warp
// CHECK-SAME:     role = "producer"
// CHECK:          tessera.tma.copy_async
// CHECK:          schedule.warp
// CHECK-SAME:     role = "consumer"
// CHECK:          tessera.attn.scaled_dot_product
// CHECK:          tessera.attn.causal_mask
// CHECK:          tessera.attn.online_softmax
// CHECK:          tessera.attn.lse_accumulate
// CHECK:          tessera.attn.lse.save
// CHECK:          tessera.nvgpu.wgmma.mma_async
// CHECK-SAME:     shape = "m64n64k16"
// CHECK:          tessera.nvgpu.wgmma.commit_group
// CHECK:          tessera.nvgpu.wgmma.wait_group
// CHECK-NOT:      tessera.flash_attn
// CHECK-NOT:      tile.mma

// PIPE-LABEL:     func.func @flash_attn_fwd

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
      tessera.tile_q   = 64 : i32,
      tessera.tile_kv  = 64 : i32
    } : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>)
          -> tensor<64x64xf32>
    return %out : tensor<64x64xf32>
  }
}
