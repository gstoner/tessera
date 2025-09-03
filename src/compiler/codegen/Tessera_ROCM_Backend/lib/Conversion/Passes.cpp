#include "TesseraROCM/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;
namespace {
struct LowerT2ROCDL : public PassWrapper<LowerT2ROCDL, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerT2ROCDL)
  void runOnOperation() override;
  StringRef getArgument() const final { return "lower-tessera-target-to-rocdl"; }
};
struct LowerKernelABI : public PassWrapper<LowerKernelABI, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerKernelABI)
  void runOnOperation() override;
  StringRef getArgument() const final { return "lower-tessera-kernel-abi"; }
};
}
void LowerT2ROCDL::runOnOperation() {}
void LowerKernelABI::runOnOperation() {}
namespace mlir::tessera_rocm {
std::unique_ptr<Pass> createLowerTesseraTargetToROCDLPass(){ return std::make_unique<LowerT2ROCDL>(); }
std::unique_ptr<Pass> createLowerKernelABIPass(){ return std::make_unique<LowerKernelABI>(); }
void registerTesseraROCMPasses(){ PassRegistration<LowerT2ROCDL>(); PassRegistration<LowerKernelABI>(); }
}
