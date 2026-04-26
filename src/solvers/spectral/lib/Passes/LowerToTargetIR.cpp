//===- LowerToTargetIR.cpp.cpp -------------------------------------------*- C++ -*-===//
// Lower to Target-IR backends
//===----------------------------------------------------------------------===//
#include "tessera/Spectral/SpectralPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace tessera {
namespace lowertotargetir {
struct LowerToTargetIRPass : public PassWrapper<LowerToTargetIRPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToTargetIRPass)
  void runOnOperation() override {
    // TODO: implement
  }
};
} // namespace
std::unique_ptr<mlir::Pass> createLowerSpectralToTargetIRPass() {
  return std::make_unique<LowerToTargetIRPass>();
}
} // namespace tessera
