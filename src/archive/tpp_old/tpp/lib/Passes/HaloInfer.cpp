
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {
struct HaloInfer : public PassWrapper<HaloInfer, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "tpp-halo-infer"; }
  StringRef getDescription() const final { return "Infer halos from stencil/BC usage"; }
  void runOnOperation() final {
    // TODO: implement
  }
};
} // namespace

std::unique_ptr<Pass> createHaloInferPass() { return std::make_unique<HaloInfer>(); }
