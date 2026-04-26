// Phase 3 TileIRLoweringPass lit tests.
// Verifies that tessera.flash_attn + tessera.matmul inside schedule.mesh.region
// are lowered to FA-4 Tile IR ops.
//
// RUN: tessera-opt --tessera-tile-ir-lowering='tile-q=64 tile-kv=64 sm=90' \
// RUN:   %s | FileCheck %s
//
// RUN: tessera-opt --tessera-tile-ir-lowering='sm=80' %s \
// RUN:   | FileCheck %s --check-prefix=SM80

// CHECK-LABEL: func.func @flash_attn_step
// CHECK:       tile.async_copy
// CHECK:       tile.wait_async
// CHECK:       tessera.attn.scaled_dot_product
// CHECK:       tessera.attn.causal_mask
// CHECK:       tessera.attn.online_softmax
// CHECK:       tessera.attn.lse_accumulate
// CHECK-NOT:   tessera.flash_attn

// SM80-LABEL:  func.func @flash_attn_step
// SM80:        tile.async_copy
// SM80:        tile.mma

module attributes {tessera.ir.version = "1.0",
                   tessera.target = {sm = 90 : i32, warps = 4 : i32,
                                     smem = 233472 : i64,
                                     pipeline_stages = 2 : i32}} {
  func.func @flash_attn_step(
      %Q: tensor<64x64xbf16>
            {tessera.effect = "read"},
      %K: tensor<64x64xbf16>
            {tessera.effect = "read"},
      %V: tensor<64x64xbf16>
            {tessera.effect = "read"}
  ) -> tensor<64x64xf32> {
    %out = "tessera.flash_attn"(%Q, %K, %V) {
      causal = true,
      tessera.tile_q  = 64 : i32,
      tessera.tile_kv = 64 : i32
    } : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>)
          -> tensor<64x64xf32>
    return %out : tensor<64x64xf32>
  }
}
