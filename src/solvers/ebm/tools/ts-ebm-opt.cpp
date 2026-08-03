// ts-ebm-opt: mlir-opt-style driver that registers the Tessera EBM
// dialect plus its canonicalization pass (EBM5) and EBM6 lowering-pass
// stubs.
//
// Registered pass arguments:
//   --tessera-ebm-canonicalize           (EBM5, real)
//   --tessera-ebm-fuse-energy-grad       (EBM6 stub)
//   --tessera-ebm-checkpoint-inner-loop  (EXPERIMENTAL -- not in the
//                                        default pipeline; see W0.2)
//   --tessera-ebm-pipeline-candidates    (EBM6 stub)
//
// Canonical pipeline alias:
//   --tessera-ebm-pipeline
// which runs (in order):
//   canonicalize → fuse-energy-grad → pipeline-candidates.

#include "tessera/EBM/EBMDialect.h"
#include "tessera/EBM/EBMPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<tessera::ebm::EBMDialect>();
  // EBM6's CheckpointInnerLoop walks scf.for ops; register the SCF and
  // Func dialects so the IR parses end-to-end.
  // `arith` is used by every fixture that builds loop bounds; without it
  // ts-ebm-opt cannot parse its own tests (W0.1 — this driver had 6 lit
  // failures that were invisible because TESSERA_BUILD_EBM_BACKEND is OFF
  // by default).
  registry.insert<arith::ArithDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<func::FuncDialect>();

  registerPass(tessera::createEBMCanonicalizePass);
  registerPass(tessera::createEBMFuseEnergyGradPass);
  registerPass(tessera::createEBMCheckpointInnerLoopPass);
  registerPass(tessera::createEBMPipelineCandidatesPass);

  // W0.2 — `checkpoint-inner-loop` is NOT in the default pipeline.
  //
  // It annotates every step of every containing loop as rematerializable
  // with no liveness, demand, or differentiability analysis, and NOTHING
  // reads the attributes it writes (`tessera.ebm.checkpoint_loop`,
  // `checkpoint_budget`, `recompute_step` have zero consumers in the tree).
  // Under Decision #29 an unconsumed declaration must not ship in a default
  // path, and under Decision #10a an eligibility-marking pass must gate on a
  // demand analysis. It remains available as an explicitly experimental
  // standalone pass (`--tessera-ebm-checkpoint-inner-loop`) so its fixtures
  // keep running. Demand-aware loop rematerialization is W5.1.
  PassPipelineRegistration<> pipeline(
      "tessera-ebm-pipeline",
      "Canonicalize → fuse-energy-grad → pipeline-candidates.",
      [](OpPassManager &pm) {
        pm.addPass(tessera::createEBMCanonicalizePass());
        pm.addPass(tessera::createEBMFuseEnergyGradPass());
        pm.addPass(tessera::createEBMPipelineCandidatesPass());
      });

  return failed(MlirOptMain(argc, argv, "ts-ebm-opt\n", registry));
}
