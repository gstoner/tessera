// RUN: tessera-opt %s --allow-unregistered-dialect | FileCheck %s
// RUN: tessera-opt %s -tessera-lower-to-apple_gpu --allow-unregistered-dialect | FileCheck %s --check-prefix=GPU

// DiffusionGemma block-diffusion canvas-denoise step — a structured Graph IR
// region op the compiler sees as one op (vs. the unrolled per-layer attention/
// MoE graph). It carries the canvas + committed encoder KV + the step's
// structural attributes, verifies the bidirectional / GQA / head_dim contract,
// and lowers as an Apple GPU metal_runtime region.

// CHECK-LABEL: func.func @step
// CHECK: tessera.diffusion_block_step
// CHECK-SAME: head_dim = 256
// CHECK-SAME: num_attention_heads = 10
// CHECK-SAME: num_kv_heads = 2
func.func @step(%c: tensor<256x2560xf32>, %k: tensor<2x8x256xf32>,
                %v: tensor<2x8x256xf32>) -> tensor<256x2560xf32> {
  %o = "tessera.diffusion_block_step"(%c, %k, %v)
        {num_denoise_layers = 30 : i64, num_attention_heads = 10 : i64,
         num_kv_heads = 2 : i64, head_dim = 256 : i64, causal = false}
        : (tensor<256x2560xf32>, tensor<2x8x256xf32>, tensor<2x8x256xf32>) -> tensor<256x2560xf32>
  return %o : tensor<256x2560xf32>
}

// The Tile→Apple pass recognizes the region as an Apple GPU runtime op and tags
// it metal_runtime (the per-head attention rides the attn_bias/flash_attn lane,
// the MoE the moe_swiglu_block lane).

// GPU-LABEL: func.func @tile_step
// GPU: source = "tessera.diffusion_block_step"
// GPU-SAME: status = "metal_runtime"
func.func @tile_step() {
  "tile.mock"() {source = "tessera.diffusion_block_step", result = "O", ordinal = 0 : i64} : () -> ()
  return
}
