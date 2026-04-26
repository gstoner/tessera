
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {
struct LowerTPPToTargetIR : public PassWrapper<LowerTPPToTargetIR, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "lower-tpp-to-target-ir"; }
  StringRef getDescription() const final { return "Lower to Target-IR"; }
  void runOnOperation() final {
    // TODO: implement
  }
};
} // namespace

std::unique_ptr<Pass> createLowerTPPToTargetIRPass() { return std::make_unique<LowerTPPToTargetIR>(); }
