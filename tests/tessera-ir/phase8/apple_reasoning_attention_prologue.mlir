// Sprint 10 (2026-06-03) — Apple reasoning-model attention-family prologue.
//
// The `tessera-lower-to-apple_{cpu,gpu}-full` value pipelines now run the Graph
// IR attention-family recognizer passes (SwiGLU / MLA / DeepSeek NSA / Ling-Kimi
// hybrid / Lightning / DeltaNet-Kimi) BEFORE distribution and tiling — exactly
// like the NVIDIA `tessera-nvidia-pipeline`. This makes reasoning models
// compiler-visible on the Apple spine.
//
// HONESTY CONTRACT (Decision #21 / VALUE_TARGET_IR_CONTRACT.md):
//   * The DeepSeek MLA decode chain (latent_kv_compress → expand_k/v →
//     flash_attn) is RECOGNIZED and fused into `tessera.mla_decode_fused` by the
//     prologue on the Apple spine.
//   * It is NOT (yet) lowered to an executable Apple value call — there is no
//     `tessera_apple.gpu.kernel_call` and no MSL/MPS symbol for it. It stays a
//     compiler-visible Graph IR op, so the runtime never claims it executable.
//   * No `ub.poison` artifact husk is emitted — the prologue is value-honest.
//
// This is the "compiler-visible + capability-gated" state the Sprint 10 plan
// asks for: Lightning / Kimi / hybrid / MLA stay visible to the compiler and
// diagnostically gated until a real Apple runtime execution path lands.
//
// RUN: tessera-opt -tessera-lower-to-apple_gpu-full %s | FileCheck %s --check-prefix=GPU
// RUN: tessera-opt -tessera-lower-to-apple_cpu-full %s | FileCheck %s --check-prefix=CPU

// GPU-LABEL: func.func @mla_decode
// GPU: tessera.mla_decode_fused
// GPU-NOT: tessera.latent_kv_compress
// GPU-NOT: tessera.flash_attn
// GPU-NOT: tessera_apple.gpu.kernel_call
// GPU-NOT: ub.poison

// CPU-LABEL: func.func @mla_decode
// CPU: tessera.mla_decode_fused
// CPU-NOT: tessera.latent_kv_compress
// CPU-NOT: ub.poison
func.func @mla_decode(%x: tensor<4x16xf32>, %Wdkv: tensor<16x8xf32>,
                      %Wuk: tensor<8x16xf32>, %Wuv: tensor<8x16xf32>,
                      %Q: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %c = tessera.latent_kv_compress %x, %Wdkv : (tensor<4x16xf32>, tensor<16x8xf32>) -> tensor<4x8xf32>
  %K = tessera.latent_kv_expand_k %c, %Wuk : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %V = tessera.latent_kv_expand_v %c, %Wuv : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %O = tessera.flash_attn %Q, %K, %V {operandSegmentSizes = array<i32: 1, 1, 1, 0>, head_dim = 16 : i64} : (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
  return %O : tensor<4x16xf32>
}
