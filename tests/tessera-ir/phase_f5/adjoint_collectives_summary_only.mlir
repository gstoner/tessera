// RUN: tessera-opt --tessera-adjoint-collective-insertion --verify-each=false %s | FileCheck %s
// RUN: tessera-opt --tessera-adjoint-collective-insertion --verify-each=false %s | FileCheck %s --check-prefix=PLAN
//
// Phase F5 — regression guard: a function-level `tessera.effect` SUMMARY is not
// per-arg effect information.
//
// EffectAnnotationPass always sets a function-level `tessera.effect` summary,
// even when no argument carries a per-arg effect. Effect-aware gating is a
// per-arg decision, so it must key off per-arg `tessera.effect` attrs. Keying
// off the summary would flip on effect-gating here, then treat every arg's
// missing per-arg effect as "pure" and skip it — silently dropping every
// gradient collective. The pass must instead take the weight_sharding-only
// fallback and still insert.

module {
  // CHECK: sym_name = "bwd_summary_only"
  func.func @bwd_summary_only(%A: tensor<4x8xf32>, %B: tensor<8x16xf32>)
      -> (tensor<4x16xf32>, tensor<4x8xf32>, tensor<8x16xf32>)
      attributes {
        tessera.effect = "memory",
        tessera.autodiff = "reverse",
        tessera.autodiff.arg_cotangents = ["%cotan_arg_0", "%cotan_arg_1"],
        tessera.weight_sharding = {arg_0 = "dp", arg_1 = "tp"}
      } {
    %out = "test.fwd"(%A, %B) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %dA = "test.cotan"(%A) : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %dB = "test.cotan"(%B) : (tensor<8x16xf32>) -> tensor<8x16xf32>

    // Both collectives are still inserted (not skipped as "pure").
    // CHECK: "tessera.collective.reduce_scatter"(%{{.*}}) {{.*}}axis = "dp"
    // CHECK: "tessera.collective.all_gather"(%{{.*}}) {{.*}}axis = "tp"
    func.return %out, %dA, %dB
        : tensor<4x16xf32>, tensor<4x8xf32>, tensor<8x16xf32>
  }
  // Sharding-only fallback (NOT effect-gated), since there are no per-arg effects.
  // PLAN-DAG: tessera.adjoint_collective_plan = "reduce_scatter:dp [sharding-only]"
  // PLAN-DAG: tessera.adjoint_collective_plan = "all_gather:tp [sharding-only]"
}
