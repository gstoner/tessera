// SD1 (tree multi-path rejection) — GenerateROCMSpecAcceptTreeSampleKernel lowers
// a tessera.spec_accept_tree_sample (the device form of speculative.batch_verify:
// per-path Leviathan accept au<=exp(target_lp-draft_lp), longest accepted prefix)
// to one cooperative-workgroup gpu.func: thread/path match-length via the
// exp-accept test into LDS, barrier, thread-0 argmax. Also exercises the op
// verifier. On-device execution on gfx1151 is proven by
// tests/unit/test_rocm_spec_accept_tree_sample_exec.py.
//
// RUN: tessera-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// ─── verifier: positive ─────────────────────────────────────────────────────
// CHECK-LABEL: func.func @ok
func.func @ok(%t: tensor<3x4xf32>, %d: tensor<3x4xf32>, %u: tensor<3x4xf32>) -> tensor<2xi32> {
  %r = "tessera.spec_accept_tree_sample"(%t, %d, %u) : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}

// -----
func.func @bad_shapes(%t: tensor<3x4xf32>, %d: tensor<2x4xf32>, %u: tensor<3x4xf32>) -> tensor<2xi32> {
  // expected-error @+1 {{draft_log_probs must match target_log_probs}}
  %r = "tessera.spec_accept_tree_sample"(%t, %d, %u) : (tensor<3x4xf32>, tensor<2x4xf32>, tensor<3x4xf32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}

// -----
func.func @bad_result(%t: tensor<3x4xf32>, %d: tensor<3x4xf32>, %u: tensor<3x4xf32>) -> tensor<3xi32> {
  // expected-error @+1 {{result must be tensor<2xi32>}}
  %r = "tessera.spec_accept_tree_sample"(%t, %d, %u) : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3xi32>
  return %r : tensor<3xi32>
}
