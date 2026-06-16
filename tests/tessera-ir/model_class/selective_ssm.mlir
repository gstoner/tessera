// RUN: tessera-opt %s | FileCheck %s
//
// Track L (L4) — `tessera.selective_ssm` is now a genuine Graph IR op (Mamba-2
// SSD recurrence).  The coverage registry previously *claimed* this op had
// landed (graph_ir_lowering = "registered") while no ODS op existed; this
// fixture is the proof that the claim is now true.  Verifier:
// `SelectiveSsmOp::verify` (rank-3 x / shape-compatible delta / matching b,c /
// a rank-1|2 with leading dim D and rank-2 trailing dim N / optional gate
// shape-compatible with x / state (B,D,N)).  Dim checks are dynamic-compatible
// (a dynamic dim agrees with anything); negatives in `selective_ssm_invalid.mlir`.

// CHECK-LABEL: func.func @ssm_scalar_state
func.func @ssm_scalar_state(%x: tensor<2x16x8xf32>, %a: tensor<8xf32>,
                            %b: tensor<2x16x4xf32>, %c: tensor<2x16x4xf32>,
                            %delta: tensor<2x16x8xf32>) -> tensor<2x16x8xf32> {
  // CHECK: tessera.selective_ssm %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {chunk_size = 64 : i64}
  %y = tessera.selective_ssm %x, %a, %b, %c, %delta {chunk_size = 64 : i64}
      : (tensor<2x16x8xf32>, tensor<8xf32>, tensor<2x16x4xf32>,
         tensor<2x16x4xf32>, tensor<2x16x8xf32>) -> tensor<2x16x8xf32>
  return %y : tensor<2x16x8xf32>
}

// Per-state-dim A (D, N) + optional output gate and initial-state carry.
// CHECK-LABEL: func.func @ssm_full_state_gated
func.func @ssm_full_state_gated(%x: tensor<1x32x16xf32>, %a: tensor<16x8xf32>,
                                %b: tensor<1x32x8xf32>, %c: tensor<1x32x8xf32>,
                                %delta: tensor<1x32x16xf32>, %g: tensor<1x32x16xf32>,
                                %s: tensor<1x16x8xf32>) -> tensor<1x32x16xf32> {
  // CHECK: tessera.selective_ssm %{{.*}} gate(%{{.*}}) init(%{{.*}})
  %y = tessera.selective_ssm %x, %a, %b, %c, %delta gate(%g) init(%s)
      : (tensor<1x32x16xf32>, tensor<16x8xf32>, tensor<1x32x8xf32>,
         tensor<1x32x8xf32>, tensor<1x32x16xf32>, tensor<1x32x16xf32>,
         tensor<1x16x8xf32>) -> tensor<1x32x16xf32>
  return %y : tensor<1x32x16xf32>
}

// Dynamic batch/sequence dims must verify — a dynamic dim is compatible with
// anything, so this must NOT be rejected (the old direct shape-compare did).
// CHECK-LABEL: func.func @ssm_dynamic_bs
func.func @ssm_dynamic_bs(%x: tensor<?x?x8xf32>, %a: tensor<8xf32>,
                          %b: tensor<?x?x4xf32>, %c: tensor<?x?x4xf32>,
                          %delta: tensor<?x?x8xf32>) -> tensor<?x?x8xf32> {
  // CHECK: tessera.selective_ssm
  %y = tessera.selective_ssm %x, %a, %b, %c, %delta {chunk_size = 64 : i64}
      : (tensor<?x?x8xf32>, tensor<8xf32>, tensor<?x?x4xf32>,
         tensor<?x?x4xf32>, tensor<?x?x8xf32>) -> tensor<?x?x8xf32>
  return %y : tensor<?x?x8xf32>
}
