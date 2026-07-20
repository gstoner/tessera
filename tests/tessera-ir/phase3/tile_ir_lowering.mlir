// Phase 3 TileIRLoweringPass lit tests.
// Verifies that tessera.flash_attn + tessera.matmul inside schedule.mesh.region
// are lowered to FA-4 Tile IR ops.
//
// 2026-06: un-XFAIL'd.  flash_attn now carries the required head_dim attr, and
// the pass lowers to FA-4 Tile IR ops in the unregistered `tile.*` dialect
// (tile.async_copy / tile.mma) — so --allow-unregistered-dialect is required,
// but the result VERIFIES (the registered tessera_attn.* ops are valid after the
// LseSaveOp per-row-LSE verifier fix), so no --verify-each=false is needed.  The
// func.func prints pretty; the unregistered tile.* ops print generically inline.
//
// RUN: tessera-opt --tessera-tile-ir-lowering='tile-q=64 tile-kv=64 sm=90' \
// RUN:   --allow-unregistered-dialect %s | FileCheck %s
//
// RUN: tessera-opt --tessera-tile-ir-lowering='sm=80' \
// RUN:   --allow-unregistered-dialect %s | FileCheck %s --check-prefix=SM80

// CHECK:       func.func @flash_attn_step
// CHECK:       tile.async_copy
// CHECK:       tile.wait_async
// CHECK:       tessera_attn.scaled_dot_product
// CHECK:       tessera_attn.causal_mask
// CHECK:       tessera_attn.online_softmax
// CHECK:       tessera_attn.lse_accumulate
// CHECK-NOT:   tessera.flash_attn

// The flash_attn lowering is FA-4-shaped on every SM target (the online-softmax
// attn pipeline), so sm=80 produces the same async-copy + attn op sequence.
// SM80:        func.func @flash_attn_step
// SM80:        tile.async_copy
// SM80:        tessera_attn.scaled_dot_product

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
    %out = "tessera.flash_attn"(%Q, %K, %V) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {
      causal = true,
      head_dim = 64 : i64,
      tessera.tile_q  = 64 : i32,
      tessera.tile_kv = 64 : i32
    } : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>)
          -> tensor<64x64xf32>
    return %out : tensor<64x64xf32>
  }
}
