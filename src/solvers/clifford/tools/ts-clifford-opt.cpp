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
  registerPass(tessera::createCliffordAnnotateAlgebraPass);
  registerPass(tessera::createCliffordExpandProductTablePass);
  registerPass(tessera::createCliffordGradeFusionPass);
  registerPass(tessera::createCliffordRotorSandwichFoldPass);

  // Canonical end-to-end pipeline alias.
  // Pass ordering: annotate → rotor-sandwich-fold → grade-fusion →
  // expand-product-table.  rotor-sandwich-fold MUST run before
  // grade-fusion (otherwise output_grades on the inner geo_products
  // obscures the sandwich pattern).
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
