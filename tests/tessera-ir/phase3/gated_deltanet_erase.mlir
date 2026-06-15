// RUN: tessera-opt %s | FileCheck %s
//
// Track L (L3.1 / item 1) — `erase` is a first-class attribute on the DeltaNet
// family ops (`Tessera_DeltaAttentionOp` base).  `erase=false` (default) is gated
// linear attention; `erase=true` is the genuine DeltaNet `(I − β k kᵀ)` rule,
// which the apple_gpu runtime honors (routes to the genuine kernel).  This proves
// the attribute parses/verifies/round-trips so the choice is representable in IR.

// CHECK-LABEL: func.func @deltanet_erase
func.func @deltanet_erase(%q: tensor<2x3x16x16xf32>, %k: tensor<2x3x16x16xf32>,
                          %v: tensor<2x3x16x16xf32>) -> tensor<2x3x16x16xf32> {
  // CHECK: tessera.gated_deltanet {{.*}} {erase = true
  %o = "tessera.gated_deltanet"(%q, %k, %v) <{
      erase = true, causal = true, return_state = false, state_dtype = "fp32",
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>}>
      : (tensor<2x3x16x16xf32>, tensor<2x3x16x16xf32>, tensor<2x3x16x16xf32>)
      -> tensor<2x3x16x16xf32>
  return %o : tensor<2x3x16x16xf32>
}

// Default (erase omitted → false) still parses — backward-compatible.
// CHECK-LABEL: func.func @deltanet_default
func.func @deltanet_default(%q: tensor<1x2x8x8xf32>, %k: tensor<1x2x8x8xf32>,
                            %v: tensor<1x2x8x8xf32>) -> tensor<1x2x8x8xf32> {
  // CHECK: tessera.kimi_delta_attention
  %o = "tessera.kimi_delta_attention"(%q, %k, %v) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>}>
      : (tensor<1x2x8x8xf32>, tensor<1x2x8x8xf32>, tensor<1x2x8x8xf32>)
      -> tensor<1x2x8x8xf32>
  return %o : tensor<1x2x8x8xf32>
}
