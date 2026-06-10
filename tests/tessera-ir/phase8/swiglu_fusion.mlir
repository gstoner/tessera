// RUN: tessera-opt %s --tessera-swiglu-fusion --allow-unregistered-dialect | FileCheck %s

// Phase 8.4.8 — SwiGLU Performance Plan, Stage 2b.
//
// Verifies that the Schedule IR fusion recognizer collapses the 3-op
// SwiGLU chain into a single tessera.swiglu_fused op, and that it
// declines to match when the chain shape differs (different %x feeding
// gate vs up matmul, intermediate value with multiple users, etc.).

// ----------------------------------------------------------------------------
// Positive: canonical SwiGLU 3-op chain collapses to swiglu_fused.
//
// Lift x/W_gate/W_up/W_down through the public function arguments so the
// SSA chain is exactly what the pattern looks for: gate.matmul(%x, %Wg) /
// up.matmul(%x, %Wu) / silu_mul / down.matmul(_, %Wd).
// ----------------------------------------------------------------------------

func.func @swiglu_collapses(%x: tensor<8x16xf32>,
                            %Wg: tensor<16x32xf32>,
                            %Wu: tensor<16x32xf32>,
                            %Wd: tensor<32x16xf32>) -> tensor<8x16xf32> {
  // CHECK-LABEL: func.func @swiglu_collapses
  // CHECK:       tessera.swiglu_fused {{.*}} : (tensor<8x16xf32>, tensor<16x32xf32>, tensor<16x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NOT:   tessera.silu_mul
  // CHECK-NOT:   tessera.matmul
  %gate = "tessera.matmul"(%x, %Wg) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %up   = "tessera.matmul"(%x, %Wu) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %h    = "tessera.silu_mul"(%gate, %up) : (tensor<8x32xf32>, tensor<8x32xf32>) -> tensor<8x32xf32>
  %out  = "tessera.matmul"(%h, %Wd) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %out : tensor<8x16xf32>
}

// ----------------------------------------------------------------------------
// Negative #1: gate and up matmuls consume different %x — not a SwiGLU
// block. The pattern must decline to match; the chain stays intact.
// ----------------------------------------------------------------------------

func.func @two_inputs_no_fusion(%x1: tensor<8x16xf32>,
                                %x2: tensor<8x16xf32>,
                                %Wg: tensor<16x32xf32>,
                                %Wu: tensor<16x32xf32>,
                                %Wd: tensor<32x16xf32>) -> tensor<8x16xf32> {
  // CHECK-LABEL: func.func @two_inputs_no_fusion
  // CHECK:       tessera.matmul
  // CHECK:       tessera.matmul
  // CHECK:       tessera.silu_mul
  // CHECK:       tessera.matmul
  // CHECK-NOT:   tessera.swiglu_fused
  %gate = "tessera.matmul"(%x1, %Wg) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %up   = "tessera.matmul"(%x2, %Wu) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %h    = "tessera.silu_mul"(%gate, %up) : (tensor<8x32xf32>, tensor<8x32xf32>) -> tensor<8x32xf32>
  %out  = "tessera.matmul"(%h, %Wd) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %out : tensor<8x16xf32>
}

// ----------------------------------------------------------------------------
// Negative #2: %hidden has a second consumer — fusing would lose that
// consumer's input. The pattern must decline to match.
// ----------------------------------------------------------------------------

func.func @hidden_with_extra_user(%x: tensor<8x16xf32>,
                                  %Wg: tensor<16x32xf32>,
                                  %Wu: tensor<16x32xf32>,
                                  %Wd: tensor<32x16xf32>)
    -> (tensor<8x16xf32>, tensor<8x32xf32>) {
  // CHECK-LABEL: func.func @hidden_with_extra_user
  // CHECK:       tessera.silu_mul
  // CHECK-NOT:   tessera.swiglu_fused
  %gate = "tessera.matmul"(%x, %Wg) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %up   = "tessera.matmul"(%x, %Wu) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %h    = "tessera.silu_mul"(%gate, %up) : (tensor<8x32xf32>, tensor<8x32xf32>) -> tensor<8x32xf32>
  %out  = "tessera.matmul"(%h, %Wd) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %out, %h : tensor<8x16xf32>, tensor<8x32xf32>
}

// ----------------------------------------------------------------------------
// Negative #3: down matmul has a transposeB attribute — needs its own
// fused-op variant. Pattern must decline.
// ----------------------------------------------------------------------------

func.func @down_with_transpose_no_fusion(%x: tensor<8x16xf32>,
                                         %Wg: tensor<16x32xf32>,
                                         %Wu: tensor<16x32xf32>,
                                         %Wd: tensor<16x32xf32>) -> tensor<8x16xf32> {
  // CHECK-LABEL: func.func @down_with_transpose_no_fusion
  // CHECK:       tessera.matmul
  // CHECK-NOT:   tessera.swiglu_fused
  %gate = "tessera.matmul"(%x, %Wg) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %up   = "tessera.matmul"(%x, %Wu) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %h    = "tessera.silu_mul"(%gate, %up) : (tensor<8x32xf32>, tensor<8x32xf32>) -> tensor<8x32xf32>
  %out  = "tessera.matmul"(%h, %Wd) {transposeB = true}
              : (tensor<8x32xf32>, tensor<16x32xf32>) -> tensor<8x16xf32>
  return %out : tensor<8x16xf32>
}

// ----------------------------------------------------------------------------
// numeric_policy propagation (audit 2026-06-10, Decision #15a): a chain whose
// matmuls agree on numeric_policy must carry it onto the fused op instead of
// dropping it.
// ----------------------------------------------------------------------------

func.func @swiglu_propagates_numeric_policy(%x: tensor<8x16xf32>,
                                            %Wg: tensor<16x32xf32>,
                                            %Wu: tensor<16x32xf32>,
                                            %Wd: tensor<32x16xf32>) -> tensor<8x16xf32> {
  // CHECK-LABEL: func.func @swiglu_propagates_numeric_policy
  // CHECK:       tessera.swiglu_fused
  // CHECK-SAME:  numeric_policy = {accum = "fp32", storage = "bf16"}
  %gate = "tessera.matmul"(%x, %Wg) {numeric_policy = {storage = "bf16", accum = "fp32"}}
              : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %up   = "tessera.matmul"(%x, %Wu) {numeric_policy = {storage = "bf16", accum = "fp32"}}
              : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %h    = "tessera.silu_mul"(%gate, %up) : (tensor<8x32xf32>, tensor<8x32xf32>) -> tensor<8x32xf32>
  %out  = "tessera.matmul"(%h, %Wd) {numeric_policy = {storage = "bf16", accum = "fp32"}}
              : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %out : tensor<8x16xf32>
}

// ----------------------------------------------------------------------------
// Conflicting numeric_policy across the chain: one fused op cannot express
// per-stage policies, so the pattern must decline to fuse.
// ----------------------------------------------------------------------------

func.func @swiglu_conflicting_policy_no_fusion(%x: tensor<8x16xf32>,
                                               %Wg: tensor<16x32xf32>,
                                               %Wu: tensor<16x32xf32>,
                                               %Wd: tensor<32x16xf32>) -> tensor<8x16xf32> {
  // CHECK-LABEL: func.func @swiglu_conflicting_policy_no_fusion
  // CHECK:       tessera.silu_mul
  // CHECK-NOT:   tessera.swiglu_fused
  %gate = "tessera.matmul"(%x, %Wg) {numeric_policy = {storage = "bf16", accum = "fp32"}}
              : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %up   = "tessera.matmul"(%x, %Wu) {numeric_policy = {storage = "fp16", accum = "fp32"}}
              : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %h    = "tessera.silu_mul"(%gate, %up) : (tensor<8x32xf32>, tensor<8x32xf32>) -> tensor<8x32xf32>
  %out  = "tessera.matmul"(%h, %Wd) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %out : tensor<8x16xf32>
}
