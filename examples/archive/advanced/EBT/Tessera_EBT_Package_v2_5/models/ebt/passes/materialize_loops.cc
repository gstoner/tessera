#include "materialize_loops.h"
// TODO(mlir): include MLIR headers and Tessera EBT dialect.
/*
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;
*/
namespace tessera { namespace ebt {
// Pseudocode body to show intent:
struct MaterializeLoopsPass /*: public PassWrapper<MaterializeLoopsPass, OperationPass<ModuleOp>>*/ {
  MaterializeLoopsOptions opts;
  // void runOnOperation() override {
  //   ModuleOp m = getOperation();
  //   m.walk([&](func::FuncOp f){
  //     // Detect EBT driver (heuristic or attr)
  //     // Insert scf.for over K, then nested scf.for over T.
  //     // Clone body: grad_y + inner_step stay inside T loop.
  //     // Attach mapping attrs for downstream schedulers.
  //   });
  // }
};
mlir::Pass* createMaterializeLoopsPass(const MaterializeLoopsOptions& o) {
  // return new MaterializeLoopsPass{o};
  return nullptr;
}
void registerMaterializeLoopsPipeline() {
  // PassPipelineRegistration<MaterializeLoopsOptions>(
  //   "tessera-ebt-materialize-loops",
  //   "Materialize EBT KxT loops",
  //   [](OpPassManager& pm, const MaterializeLoopsOptions& o){
  //     pm.addPass(createMaterializeLoopsPass(o));
  //   });
}
}} // ns
