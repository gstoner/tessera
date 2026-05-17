// ts-clifford-opt: minimal mlir-opt-style driver that registers the
// Tessera Clifford dialect plus its annotation pass (GA7) and GA8
// lowering-pass stubs.
//
// Registered pass arguments:
//   --tessera-clifford-annotate-algebra      (GA7, real)
//   --tessera-clifford-expand-product-table  (GA8 stub)
//   --tessera-clifford-grade-fusion          (GA8 stub)
//   --tessera-clifford-rotor-sandwich-fold   (GA8 stub)
//
// Canonical pipeline alias:
//   --tessera-clifford-pipeline
// which runs (in order):
//   annotate → expand-product-table → grade-fusion → rotor-sandwich-fold.
//
// The stubs are no-op walks that emit per-op remarks describing where the
// real implementation will land. The annotation pass is fully implemented
// and gates downstream lowering on the v1 signature allow-list.

#include "tessera/Clifford/CliffordDialect.h"
#include "tessera/Clifford/CliffordPasses.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<tessera::clifford::CliffordDialect>();

  // Individual passes.
  PassRegistration<>(tessera::createCliffordAnnotateAlgebraPass);
  PassRegistration<>(tessera::createCliffordExpandProductTablePass);
  PassRegistration<>(tessera::createCliffordGradeFusionPass);
  PassRegistration<>(tessera::createCliffordRotorSandwichFoldPass);

  // Canonical end-to-end pipeline alias.
  PassPipelineRegistration<> pipeline(
      "tessera-clifford-pipeline",
      "Annotate → expand-product-table → grade-fusion → rotor-sandwich-fold.",
      [](OpPassManager &pm) {
        pm.addPass(tessera::createCliffordAnnotateAlgebraPass());
        pm.addPass(tessera::createCliffordExpandProductTablePass());
        pm.addPass(tessera::createCliffordGradeFusionPass());
        pm.addPass(tessera::createCliffordRotorSandwichFoldPass());
      });

  return failed(MlirOptMain(argc, argv, "ts-clifford-opt\n", registry));
}
