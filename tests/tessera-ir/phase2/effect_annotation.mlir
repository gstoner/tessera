// RUN: tessera-opt --tessera-effect-annotation %s | FileCheck %s

// ── Test 1: pure matmul → annotated pure ──────────────────────────────────
// CHECK-LABEL: func.func @pure_gemm
// CHECK-SAME:  tessera.effect = "pure"
// CHECK:       tessera.matmul

// ── Test 2: write arg → annotated memory ──────────────────────────────────
// CHECK-LABEL: func.func @write_step
// CHECK-SAME:  tessera.effect = "memory"

// ── Test 3: flash_attn with dropout → annotated random ────────────────────
// CHECK-LABEL: func.func @stochastic_attn
// CHECK-SAME:  tessera.effect = "random"

// ── Test 4: pre-existing "pure" annotation stays when body is clean ────────
// CHECK-LABEL: func.func @declared_pure
// CHECK-SAME:  tessera.effect = "pure"

// ── Test 5: canonical underscore RNG spelling → annotated random ─────────
// CHECK-LABEL: func.func @canonical_rng
// CHECK-SAME:  tessera.effect = "random"

// ── Test 6: canonical normal RNG spelling → annotated random ─────────────
// CHECK-LABEL: func.func @canonical_rng_normal
// CHECK-SAME:  tessera.effect = "random"

// ── Test 7: internal call summaries reach a fixed point ──────────────────
// CHECK-LABEL: func.func @indirect_rng
// CHECK-SAME:  tessera.effect = "random"

module attributes {tessera.ir.version = "1.0"} {

  func.func @pure_gemm(
      %A: tensor<64x128xbf16>,
      %B: tensor<128x256xbf16>
  ) -> tensor<64x256xf32> {
    %C = "tessera.matmul"(%A, %B) : (tensor<64x128xbf16>, tensor<128x256xbf16>)
                                    -> tensor<64x256xf32>
    return %C : tensor<64x256xf32>
  }

  func.func @write_step(
      %A: tensor<64x128xbf16>  {tessera.effect = "read"},
      %Y: tensor<64x64xf32>    {tessera.effect = "write"}
  ) {
    return
  }

  func.func @stochastic_attn(
      %Q: tensor<8x64x64xbf16>,
      %K: tensor<8x64x64xbf16>,
      %V: tensor<8x64x64xbf16>
  ) -> tensor<8x64x64xf32> {
    %O = "tessera.flash_attn"(%Q, %K, %V) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}>
             {head_dim = 64 : i64, dropout_p = 0.1 : f64, causal = false,
              tessera.effect_kind = "random", tessera.stochastic_identity = "seed_counter"}
             : (tensor<8x64x64xbf16>, tensor<8x64x64xbf16>, tensor<8x64x64xbf16>)
             -> tensor<8x64x64xf32>
    return %O : tensor<8x64x64xf32>
  }

  // Pre-annotated as pure; body has no side effects so pass must accept it.
  func.func @declared_pure(
      %A: tensor<32x32xbf16>,
      %B: tensor<32x32xbf16>
  ) -> tensor<32x32xf32>
      attributes {tessera.effect = "pure"} {
    %C = "tessera.matmul"(%A, %B) : (tensor<32x32xbf16>, tensor<32x32xbf16>)
                                    -> tensor<32x32xf32>
    return %C : tensor<32x32xf32>
  }

  func.func @canonical_rng() -> tensor<8xf32> {
    %sample = "tessera.rng_uniform"() {shape = [8], seed = 7 : i64,
        tessera.effect_kind = "random", tessera.stochastic_identity = "implicit_stream"}
        : () -> tensor<8xf32>
    return %sample : tensor<8xf32>
  }

  func.func @canonical_rng_normal() -> tensor<8xf32> {
    %sample = "tessera.rng_normal"() {shape = [8], seed = 11 : i64,
        tessera.effect_kind = "random", tessera.stochastic_identity = "implicit_stream"}
        : () -> tensor<8xf32>
    return %sample : tensor<8xf32>
  }

  func.func @random_helper() -> tensor<8xf32> {
    %sample = "tessera.rng_uniform"() {shape = [8], seed = 13 : i64,
        tessera.effect_kind = "random", tessera.stochastic_identity = "implicit_stream"}
        : () -> tensor<8xf32>
    return %sample : tensor<8xf32>
  }

  func.func @indirect_rng() -> tensor<8xf32> {
    %sample = func.call @random_helper() : () -> tensor<8xf32>
    return %sample : tensor<8xf32>
  }
}
