// SD1-2 — GenerateROCMSpecAcceptSampleKernel lowers a tessera.spec_accept_sample
// (Leviathan rejection sampling, explicit uniforms + CDF-inversion categorical) to
// one single-thread gpu.func: the serial accept loop (accept_u·p_draft <=
// p_target) with a residual/bonus CDF-inversion draw. On-device execution on
// gfx1151 is proven by tests/unit/test_rocm_spec_accept_sample_exec.py.
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s -split-input-file --generate-rocm-spec-accept-sample-kernel \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @f
// ABI (DRAFT i32, TARGET_PROBS f32, DRAFT_PROBS f32, ACCEPT_U f32, RESID_U f32,
// OUT i32); single-thread (tid==0), the accept test (mulf + cmpf ole) and the
// CDF-inversion categorical (cumsum + cmpf ogt + select).
// CHECK:       gpu.func @tessera_spec_accept_sample_0(%{{.*}}: memref<?xi32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xi32>) kernel
// CHECK:         scf.for
// CHECK:           arith.mulf
// CHECK:           arith.cmpf ole
// CHECK:           arith.cmpf ogt
// CHECK:           arith.select
// CHECK:         memref.store
// CHECK:         gpu.return
func.func @f(%d: tensor<3xi32>, %tp: tensor<4x4xf32>, %dp: tensor<3x4xf32>,
    %au: tensor<3xf32>, %ru: tensor<1xf32>) -> tensor<5xi32> {
  %r = "tessera.spec_accept_sample"(%d, %tp, %dp, %au, %ru)
       : (tensor<3xi32>, tensor<4x4xf32>, tensor<3x4xf32>, tensor<3xf32>, tensor<1xf32>)
       -> tensor<5xi32>
  return %r : tensor<5xi32>
}
