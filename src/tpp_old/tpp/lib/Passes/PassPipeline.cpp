
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createLegalizeSpaceTimePass();
std::unique_ptr<mlir::Pass> createHaloInferPass();
std::unique_ptr<mlir::Pass> createFuseStencilTimePass();
std::unique_ptr<mlir::Pass> createAsyncPrefetchPass();
std::unique_ptr<mlir::Pass> createVectorizeTPPPass();
std::unique_ptr<mlir::Pass> createDistributeHaloPass();
std::unique_ptr<mlir::Pass> createLowerTPPToTargetIRPass();

extern "C" void registerTPPPipelineAlias(mlir::PassPipelineRegistration<>* outReg) {
  static mlir::PassPipelineRegistration<> reg(
    "tpp-space-time",
    "TPP space–time pipeline",
    [](mlir::OpPassManager &pm) {
      pm.addPass(createLegalizeSpaceTimePass());
      pm.addPass(createHaloInferPass());
      pm.addPass(createFuseStencilTimePass());
      pm.addPass(createAsyncPrefetchPass());
      pm.addPass(createVectorizeTPPPass());
      pm.addPass(createDistributeHaloPass());
      pm.addPass(createLowerTPPToTargetIRPass());
    });
  if (outReg) *outReg = reg;
}
