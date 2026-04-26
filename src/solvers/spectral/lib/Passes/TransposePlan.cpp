//===- TransposePlan.cpp.cpp -------------------------------------------*- C++ -*-===//
// Plan transposes and vector shapes
//===----------------------------------------------------------------------===//
#include "tessera/Spectral/SpectralPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace tessera {
namespace transposeplan {
struct TransposePlanPass : public PassWrapper<TransposePlanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransposePlanPass)
  void runOnOperation() override {
    // TODO: implement
  }
};
} // namespace
std::unique_ptr<mlir::Pass> createSpectralTransposePlanPass() {
  return std::make_unique<TransposePlanPass>();
}
} // namespace tessera
