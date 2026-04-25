
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace {
struct DistributeHalo : public PassWrapper<DistributeHalo, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "tpp-distribute-halo"; }
  StringRef getDescription() const final { return "Create overlapped halo exchanges (stub)"; }
  void runOnOperation() final {}
};
}
std::unique_ptr<Pass> createDistributeHaloPass(){ return std::make_unique<DistributeHalo>(); }
