// RUN: tessera-opt --tessera-autodiff-pipeline --verify-each=false %s | FileCheck %s
// RUN: tessera-opt --tessera-autodiff-pipeline --verify-each=false %s | FileCheck %s --check-prefix=PLAN
//
// End-to-end composition of the reverse-mode autodiff pipeline:
//   AutodiffPass (F4) → ActivationRematerializationPass (F2) → AdjointCollectiveInsertionPass (F5)
//
// A data-parallel training step. Both weights carry a memory-class effect, so
// after the backward graph is materialised (F4) and remat runs as a no-op
// (no `tessera.recompute` markers here), F5 synchronises each weight's gradient
// with a reduce_scatter on the dp mesh axis. This proves the three passes hand
// off correctly: the cotangents F4 exposes as trailing results become the
// operands F5 wraps in collectives.

module {
  // CHECK: sym_name = "train_step"
  func.func @train_step(%A: tensor<4x8xf32> {tessera.effect = "memory"},
                        %B: tensor<8x16xf32> {tessera.effect = "memory"})
      -> tensor<4x16xf32>
      attributes {
        tessera.autodiff = "reverse",
        tessera.weight_sharding = {arg_0 = "dp", arg_1 = "dp"}
      } {
    %C = "tessera.matmul"(%A, %B) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    func.return %C : tensor<4x16xf32>
  }
  // F4 emitted the transposed-matmul adjoints for the two weights ...
  // CHECK: tessera.matmul
  // ... and F5 wrapped each weight's cotangent in a dp reduce_scatter.
  // CHECK: "tessera.collective.reduce_scatter"(%{{.*}}) {{.*}}axis = "dp"
  // CHECK: "tessera.collective.reduce_scatter"(%{{.*}}) {{.*}}axis = "dp"

  // Both plans are effect-gated (the memory effect drove the insertion), and
  // F4's arg_cotangents record survives onto the function.
  // PLAN-DAG: tessera.adjoint_collective_inserted
  // PLAN-DAG: tessera.autodiff.arg_cotangents
  // PLAN-DAG: tessera.adjoint_collective_plan = "reduce_scatter:dp [effect-gated]"
}
