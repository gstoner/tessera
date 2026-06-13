// RUN: tessera-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect

// Canvas denoising must be bidirectional — a causal step is the encoder prefill.
func.func @causal_canvas(%c: tensor<256x2560xf32>, %k: tensor<2x8x256xf32>,
                         %v: tensor<2x8x256xf32>) -> tensor<256x2560xf32> {
  // expected-error @+1 {{canvas denoising must be bidirectional (causal = false)}}
  %o = "tessera.diffusion_block_step"(%c, %k, %v)
        {num_denoise_layers = 30 : i64, num_attention_heads = 10 : i64,
         num_kv_heads = 2 : i64, head_dim = 256 : i64, causal = true}
        : (tensor<256x2560xf32>, tensor<2x8x256xf32>, tensor<2x8x256xf32>) -> tensor<256x2560xf32>
  return %o : tensor<256x2560xf32>
}

// -----

// GQA: query heads must be a multiple of KV heads.
func.func @bad_gqa(%c: tensor<256x2560xf32>, %k: tensor<2x8x256xf32>,
                   %v: tensor<2x8x256xf32>) -> tensor<256x2560xf32> {
  // expected-error @+1 {{GQA requires num_attention_heads (10) to be a multiple of num_kv_heads (3)}}
  %o = "tessera.diffusion_block_step"(%c, %k, %v)
        {num_denoise_layers = 30 : i64, num_attention_heads = 10 : i64,
         num_kv_heads = 3 : i64, head_dim = 256 : i64, causal = false}
        : (tensor<256x2560xf32>, tensor<2x8x256xf32>, tensor<2x8x256xf32>) -> tensor<256x2560xf32>
  return %o : tensor<256x2560xf32>
}

// -----

// head_dim must fit the GPU attention kernel (<= 256).
func.func @bad_head_dim(%c: tensor<256x2560xf32>, %k: tensor<2x8x512xf32>,
                        %v: tensor<2x8x512xf32>) -> tensor<256x2560xf32> {
  // expected-error @+1 {{head_dim must be in (0, 256] for the GPU attention kernel; got 512}}
  %o = "tessera.diffusion_block_step"(%c, %k, %v)
        {num_denoise_layers = 30 : i64, num_attention_heads = 10 : i64,
         num_kv_heads = 2 : i64, head_dim = 512 : i64, causal = false}
        : (tensor<256x2560xf32>, tensor<2x8x512xf32>, tensor<2x8x512xf32>) -> tensor<256x2560xf32>
  return %o : tensor<256x2560xf32>
}
