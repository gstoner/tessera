
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {
struct LegalizeSpaceTime : public PassWrapper<LegalizeSpaceTime, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "tpp-legalize-space-time"; }
  StringRef getDescription() const final { return "Normalize/annotate TPP space–time constructs"; }
  void runOnOperation() final {
    // TODO: implement
  }
};
} // namespace

std::unique_ptr<Pass> createLegalizeSpaceTimePass() { return std::make_unique<LegalizeSpaceTime>(); }
