
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace {
struct LegalizeSpaceTime : public PassWrapper<LegalizeSpaceTime, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "tpp-legalize-space-time"; }
  StringRef getDescription() const final { return "Normalize/annotate TPP space–time constructs"; }
  void runOnOperation() final {
    // Placeholder: in v0.2 we keep as no-op; future: type normalization.
  }
};
} // namespace
std::unique_ptr<Pass> createLegalizeSpaceTimePass(){ return std::make_unique<LegalizeSpaceTime>(); }
