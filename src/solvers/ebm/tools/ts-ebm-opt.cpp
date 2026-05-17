// ts-ebm-opt: mlir-opt-style driver that registers the Tessera EBM
// dialect plus its canonicalization pass (EBM5) and EBM6 lowering-pass
// stubs.
//
// Registered pass arguments:
//   --tessera-ebm-canonicalize           (EBM5, real)
//   --tessera-ebm-fuse-energy-grad       (EBM6 stub)
//   --tessera-ebm-checkpoint-inner-loop  (EBM6 stub)
//   --tessera-ebm-pipeline-candidates    (EBM6 stub)
//
// Canonical pipeline alias:
//   --tessera-ebm-pipeline
// which runs (in order):
//   canonicalize → fuse-energy-grad → checkpoint-inner-loop → pipeline-candidates.

#include "tessera/EBM/EBMDialect.h"
#include "tessera/EBM/EBMPasses.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<tessera::ebm::EBMDialect>();

  PassRegistration<>(tessera::createEBMCanonicalizePass);
  PassRegistration<>(tessera::createEBMFuseEnergyGradPass);
  PassRegistration<>(tessera::createEBMCheckpointInnerLoopPass);
  PassRegistration<>(tessera::createEBMPipelineCandidatesPass);

  PassPipelineRegistration<> pipeline(
      "tessera-ebm-pipeline",
      "Canonicalize → fuse-energy-grad → checkpoint-inner-loop → "
      "pipeline-candidates.",
      [](OpPassManager &pm) {
        pm.addPass(tessera::createEBMCanonicalizePass());
        pm.addPass(tessera::createEBMFuseEnergyGradPass());
        pm.addPass(tessera::createEBMCheckpointInnerLoopPass());
        pm.addPass(tessera::createEBMPipelineCandidatesPass());
      });

  return failed(MlirOptMain(argc, argv, "ts-ebm-opt\n", registry));
}
