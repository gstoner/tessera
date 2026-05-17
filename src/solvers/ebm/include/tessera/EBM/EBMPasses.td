//===- EBMPasses.td ---------------------------------------*- tablegen -*-===//
// Pass declarations for Tessera EBM dialect.
//
// EBM5 ships a canonicalization pass (`tessera-ebm-canonicalize`) that
// resolves inner-loop schedule attributes and normalizes self_verify.
// EBM6 lowering passes are declared as stubs here; their implementations
// land with the EBM6 sprint.
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def EBMCanonicalize : Pass<"tessera-ebm-canonicalize", "mlir::ModuleOp"> {
  let summary =
    "Resolve inner-loop schedule metadata on tessera_ebm.* ops "
    "(K candidates × T steps × manifold).";
  let description = [{
    Walks every `tessera_ebm.*` op, attaches:
      tessera.ebm.canonical    : unit attr — present once the op is normalized
      tessera.ebm.manifold     : "euclidean" | "sphere" | "bivector"
                                 (mirrored from langevin_step args)

    Self-verify with `beta = 0` is normalized to a hard `argmin` form.
  }];
}

def EBMFuseEnergyGrad : Pass<"tessera-ebm-fuse-energy-grad", "mlir::ModuleOp"> {
  let summary =
    "[EBM6 stub] Fuse `energy` + `grad_y` evaluations so the inner loop "
    "reuses activations across both forward and backward passes.";
}

def EBMCheckpointInnerLoop : Pass<"tessera-ebm-checkpoint-inner-loop", "mlir::ModuleOp"> {
  let summary =
    "[EBM6 stub] Apply rematerialization to the T-step inner-loop trajectory; "
    "tunable budget.";
}

def EBMPipelineCandidates : Pass<"tessera-ebm-pipeline-candidates", "mlir::ModuleOp"> {
  let summary =
    "[EBM6 stub] Map the K-candidate dimension across streams / devices "
    "for self_verify parallelism.";
}
