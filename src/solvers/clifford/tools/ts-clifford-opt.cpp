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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<tessera::clifford::CliffordDialect>();
  // GA8 lowering passes emit arith + tensor IR; register both so the
  // lowered modules are valid + printable.
  registry.insert<arith::ArithDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<func::FuncDialect>();

  // Individual passes.
  PassRegistration<>(tessera::createCliffordAnnotateAlgebraPass);
  PassRegistration<>(tessera::createCliffordExpandProductTablePass);
  PassRegistration<>(tessera::createCliffordGradeFusionPass);
  PassRegistration<>(tessera::createCliffordRotorSandwichFoldPass);

  // Canonical end-to-end pipeline alias.
  // Pass ordering rationale:
  //   - annotate first: validates the v1 signature allow-list and
  //     attaches `tessera.clifford.canonical` markers.
  //   - rotor-sandwich-fold second: matches `gp(gp(R, x), reverse(R))`
  //     chains and rewrites to a single `rotor_sandwich`. Must run
  //     before grade-fusion (which would obscure the chain by
  //     attaching `output_grades` attrs to the inner geo_products).
  //   - grade-fusion third: attaches `output_grades` on any
  //     geo_product whose only consumer is a `grade` op.
  //   - expand-product-table last: lowers every surviving geo_product
  //     to an unrolled arith.mulf + arith.addf sequence driven by the
  //     compile-time-known Cayley table, honoring `output_grades` for
  //     grade-restricted contractions.
  PassPipelineRegistration<> pipeline(
      "tessera-clifford-pipeline",
      "Annotate → rotor-sandwich-fold → grade-fusion → expand-product-table.",
      [](OpPassManager &pm) {
        pm.addPass(tessera::createCliffordAnnotateAlgebraPass());
        pm.addPass(tessera::createCliffordRotorSandwichFoldPass());
        pm.addPass(tessera::createCliffordGradeFusionPass());
        pm.addPass(tessera::createCliffordExpandProductTablePass());
      });

  return failed(MlirOptMain(argc, argv, "ts-clifford-opt\n", registry));
}
