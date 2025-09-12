
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {
struct DistributeHalo : public PassWrapper<DistributeHalo, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "tpp-distribute-halo"; }
  StringRef getDescription() const final { return "Create overlapped halo exchanges"; }
  void runOnOperation() final {
    // TODO: implement
  }
};
} // namespace

std::unique_ptr<Pass> createDistributeHaloPass() { return std::make_unique<DistributeHalo>(); }
