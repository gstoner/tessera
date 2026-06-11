// LSA Graph→Schedule lowering (Gap-2 follow-on): tessera-lookahead-sparse-
// prefetch emits a schedule.prefetch{into=host} that the LSA op consumes (the
// cold-pool KV staging becomes a first-class IR value), then the real
// tpp-async-prefetch pass records it WITHOUT claiming overlap (into=host /
// overlap=none ⇒ overlapped=false, not hoisted). See
// docs/audit/domain/archive/lsa_scope.md.
//
// RUN: tessera-opt %s -tessera-lookahead-sparse-prefetch -tpp-async-prefetch | FileCheck %s

func.func @lsa_staged(
    %q: tensor<2x3x16x16xf32>, %k: tensor<2x3x16x16xf32>,
    %v: tensor<2x3x16x16xf32>) -> tensor<2x3x16x16xf32> {
  // CHECK-LABEL: func.func @lsa_staged

  // The prefetch is emitted before the attention, recording the host cold-pool
  // staging tier, and is left un-overlapped (synchronous staging today).
  // CHECK: schedule.prefetch
  // CHECK-SAME: into = "host"
  // CHECK-SAME: overlap = "none"
  // CHECK-SAME: tessera.lsa.staging = "host_cold_pool"
  // CHECK-SAME: tpp.prefetch.hoisted = false
  // CHECK-SAME: tpp.prefetch.overlapped = false

  // The LSA op now consumes the prefetch result (true dataflow edge) and is
  // tagged as having emitted its staging prefetch.
  // CHECK: tessera.lookahead_sparse_attention
  // CHECK-SAME: tessera.lsa.prefetch_emitted = true
  %0 = "tessera.lookahead_sparse_attention"(%q, %k, %v)
      {window_size = 6 : i64, block_size = 4 : i64, tau = 64 : i64,
       threshold = 5.000000e-01 : f64, causal = true}
      : (tensor<2x3x16x16xf32>, tensor<2x3x16x16xf32>,
         tensor<2x3x16x16xf32>) -> tensor<2x3x16x16xf32>
  return %0 : tensor<2x3x16x16xf32>
}
