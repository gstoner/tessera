// RUN: tessera-opt %s -tessera-decompose-varlen-sdpa -split-input-file | FileCheck %s

// VarlenSdpaDecomposePass (2026-06-15) — lower tessera.varlen_sdpa onto the
// per-block tessera.flash_attn lane. Static cu_seqlens (arith.constant) decompose
// to extract_slice → flash_attn → insert_slice; dynamic cu_seqlens are preserved
// and annotated tessera.varlen_lowering = "runtime_per_block_flash_attn".

// ─────────────────────────────────────────────────────────────────────────
// Static square-causal (Reasoner pathway): cu_seqlens = [0,3,5] → 2 blocks.
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @varlen_static_square_causal
// CHECK-NOT:   tessera.varlen_sdpa
// CHECK:       tensor.empty
// CHECK:       tensor.extract_slice
// CHECK:       tessera.flash_attn
// CHECK-SAME:  causal = true
// CHECK:       tensor.insert_slice
// CHECK:       tensor.extract_slice
// CHECK:       tessera.flash_attn
// CHECK:       tensor.insert_slice
func.func @varlen_static_square_causal(
    %q: tensor<2x5x8xf32>, %k: tensor<2x5x8xf32>, %v: tensor<2x5x8xf32>
) -> tensor<2x5x8xf32> {
  %cu = arith.constant dense<[0, 3, 5]> : tensor<3xi32>
  %o = tessera.varlen_sdpa %q, %k, %v, %cu, %cu {
    head_dim = 8 : i64, causal = true
  } : (tensor<2x5x8xf32>, tensor<2x5x8xf32>, tensor<2x5x8xf32>,
       tensor<3xi32>, tensor<3xi32>) -> tensor<2x5x8xf32>
  return %o : tensor<2x5x8xf32>
}

// -----

// Static rectangular-bidirectional (Generator pathway): cu_q=[0,2,3] (total 3),
// cu_k=[0,5,8] (total 8) — independent cu_seqlens. Bidirectional always
// decomposes; two flash_attn blocks over the longer key streams.

// CHECK-LABEL: func.func @varlen_static_rectangular
// CHECK-NOT:   tessera.varlen_sdpa
// CHECK-COUNT-2: tessera.flash_attn
func.func @varlen_static_rectangular(
    %q: tensor<2x3x8xf32>, %k: tensor<2x8x8xf32>, %v: tensor<2x8x8xf32>
) -> tensor<2x3x8xf32> {
  %cuq = arith.constant dense<[0, 2, 3]> : tensor<3xi32>
  %cuk = arith.constant dense<[0, 5, 8]> : tensor<3xi32>
  %o = tessera.varlen_sdpa %q, %k, %v, %cuq, %cuk {
    head_dim = 8 : i64, causal = false
  } : (tensor<2x3x8xf32>, tensor<2x8x8xf32>, tensor<2x8x8xf32>,
       tensor<3xi32>, tensor<3xi32>) -> tensor<2x3x8xf32>
  return %o : tensor<2x3x8xf32>
}

// -----

// Dynamic cu_seqlens (function arguments, not constant): the op is preserved
// and annotated with the runtime per-block lowering marker — the honest path
// for data-dependent packing (runtime / Phase G-H frontier kernel).

// CHECK-LABEL: func.func @varlen_dynamic_preserved
// CHECK:       tessera.varlen_sdpa
// CHECK-SAME:  tessera.varlen_lowering = "runtime_per_block_flash_attn"
func.func @varlen_dynamic_preserved(
    %q: tensor<2x5x8xf32>, %k: tensor<2x5x8xf32>, %v: tensor<2x5x8xf32>,
    %cuq: tensor<3xi32>, %cuk: tensor<3xi32>
) -> tensor<2x5x8xf32> {
  %o = tessera.varlen_sdpa %q, %k, %v, %cuq, %cuk {
    head_dim = 8 : i64, causal = false
  } : (tensor<2x5x8xf32>, tensor<2x5x8xf32>, tensor<2x5x8xf32>,
       tensor<3xi32>, tensor<3xi32>) -> tensor<2x5x8xf32>
  return %o : tensor<2x5x8xf32>
}

// -----

// Static causal-RECTANGULAR: flash_attn's triangular causal would not match the
// varlen bottom-right rule, so the pass must NOT decompose — it preserves +
// annotates instead.

// CHECK-LABEL: func.func @varlen_causal_rectangular_preserved
// CHECK:       tessera.varlen_sdpa
// CHECK-SAME:  tessera.varlen_lowering = "runtime_per_block_flash_attn"
func.func @varlen_causal_rectangular_preserved(
    %q: tensor<2x3x8xf32>, %k: tensor<2x8x8xf32>, %v: tensor<2x8x8xf32>
) -> tensor<2x3x8xf32> {
  %cuq = arith.constant dense<[0, 2, 3]> : tensor<3xi32>
  %cuk = arith.constant dense<[0, 5, 8]> : tensor<3xi32>
  %o = tessera.varlen_sdpa %q, %k, %v, %cuq, %cuk {
    head_dim = 8 : i64, causal = true
  } : (tensor<2x3x8xf32>, tensor<2x8x8xf32>, tensor<2x8x8xf32>,
       tensor<3xi32>, tensor<3xi32>) -> tensor<2x3x8xf32>
  return %o : tensor<2x3x8xf32>
}
