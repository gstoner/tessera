// RUN: tessera-opt --tessera-adjoint-collective-insertion --verify-each=false %s | FileCheck %s
// RUN: tessera-opt --tessera-adjoint-collective-insertion --verify-each=false %s | FileCheck %s --check-prefix=PLAN
//
// Phase F5 — AdjointCollectiveInsertionPass, effect-aware path.
//
// The function is already in post-AutodiffPass form: it carries
// tessera.autodiff="reverse", a tessera.autodiff.arg_cotangents array, a
// tessera.weight_sharding dictionary, and a return that yields the original
// result followed by the per-arg cotangents. Every argument carries a
// memory-class tessera.effect, so each cotangent is synchronised with the
// collective its sharding kind selects:
//   arg_0 (dp)         → reduce_scatter on "dp"
//   arg_1 (tp)         → all_gather on "tp"
//   arg_2 (replicated) → all_reduce
//
// Placeholder producers use the unregistered `test` dialect (the cotangent
// SSA values are all this pass needs; the real backward graph is AutodiffPass's
// job upstream). Because the pass inserts unregistered `tessera.collective.*`
// marker ops, the module prints in generic form — CHECKs are substring-based.

module {
  // CHECK: sym_name = "bwd"
  func.func @bwd(%A: tensor<4x8xf32> {tessera.effect = "memory"},
                 %B: tensor<8x16xf32> {tessera.effect = "write"},
                 %C: tensor<4x16xf32> {tessera.effect = "reduce_sum"})
      -> (tensor<4x16xf32>, tensor<4x8xf32>, tensor<8x16xf32>, tensor<4x16xf32>)
      attributes {
        tessera.autodiff = "reverse",
        tessera.autodiff.arg_cotangents = ["%cotan_arg_0", "%cotan_arg_1", "%cotan_arg_2"],
        tessera.weight_sharding = {arg_0 = "dp", arg_1 = "tp", arg_2 = "replicated"}
      } {
    %out = "test.fwd"(%A, %B) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %dA = "test.cotan"(%A) : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %dB = "test.cotan"(%B) : (tensor<8x16xf32>) -> tensor<8x16xf32>
    %dC = "test.cotan"(%C) : (tensor<4x16xf32>) -> tensor<4x16xf32>

    // CHECK: "tessera.collective.reduce_scatter"(%{{.*}}) {{.*}}axis = "dp"
    // CHECK: "tessera.collective.all_gather"(%{{.*}}) {{.*}}axis = "tp"
    // CHECK: "tessera.collective.all_reduce"(%{{.*}}) {{.*}}axis = "dp"
    func.return %out, %dA, %dB, %dC
        : tensor<4x16xf32>, tensor<4x8xf32>, tensor<8x16xf32>, tensor<4x16xf32>
  }
  // Provenance: every insertion was driven by the memory-effect gate. These
  // land in the func header's arg_attrs (before the collective ops), so they
  // are checked under a separate prefix to avoid positional coupling.
  // PLAN-DAG: tessera.adjoint_collective_inserted
  // PLAN-DAG: tessera.adjoint_collective_plan = "reduce_scatter:dp [effect-gated]"
  // PLAN-DAG: tessera.adjoint_collective_plan = "all_gather:tp [effect-gated]"
  // PLAN-DAG: tessera.adjoint_collective_plan = "all_reduce [effect-gated]"
}
