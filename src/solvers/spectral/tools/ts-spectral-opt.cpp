// ts-spectral-opt: minimal mlir-opt-style driver that registers the Tessera
// spectral dialect plus all six planning / lowering passes.
//
// Registered pass arguments:
//   --tessera-legalize-spectral
//   --tessera-spectral-mxp
//   --tessera-spectral-transpose-plan
//   --tessera-spectral-autotune
//   --tessera-spectral-distributed
//   --lower-spectral-to-target-ir
//
// Canonical full-stack pipeline alias:
//   --tessera-spectral-pipeline
// which runs (in order):
//   legalize → mxp → transpose-plan → autotune → distributed → lower-to-target-ir.
//
// The implicit ordering matches what every fft/ifft op needs: later passes
// read attributes set by earlier ones (e.g., autotune reads
// tessera.spectral.stages from legalize; lower-to-target-ir reads
// tessera.target_ir.backend from autotune's module-level target hint).

#include "tessera/Spectral/SpectralDialect.h"
#include "tessera/Spectral/SpectralPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<tessera::spectral::SpectralDialect>();
  registry.insert<func::FuncDialect>(); // test IR wraps ops in func.func

  // Individual passes.  registerPass() takes a factory returning
  // unique_ptr<Pass>; PassRegistration<> now requires a concrete pass type,
  // which these opaque factories intentionally do not expose.
  registerPass(tessera::createLegalizeSpectralPass);
  registerPass(tessera::createSpectralMXPPass);
  registerPass(tessera::createSpectralTransposePlanPass);
  registerPass(tessera::createSpectralAutotunePass);
  registerPass(tessera::createSpectralDistributedPass);
  registerPass(tessera::createLowerSpectralToTargetIRPass);

  // Canonical end-to-end pipeline alias.
  PassPipelineRegistration<> pipeline(
      "tessera-spectral-pipeline",
      "Legalize → MXP → transpose-plan → autotune → distributed → "
      "lower-to-target-ir.",
      [](OpPassManager &pm) {
        pm.addPass(tessera::createLegalizeSpectralPass());
        pm.addPass(tessera::createSpectralMXPPass());
        pm.addPass(tessera::createSpectralTransposePlanPass());
        pm.addPass(tessera::createSpectralAutotunePass());
        pm.addPass(tessera::createSpectralDistributedPass());
        pm.addPass(tessera::createLowerSpectralToTargetIRPass());
      });

  // Lightweight cleanup hook preserved for backwards compat.
  PassPipelineRegistration<> cleanup(
      "tessera-spectral-cleanup", "No-op cleanup for now",
      [](OpPassManager &pm) {});

  return failed(MlirOptMain(argc, argv, "ts-spectral-opt\n", registry));
}
