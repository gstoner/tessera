#include "tessera_empirical_search_pass.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tessera {
namespace {
struct EmpiricalSearchPass : public PassWrapper<EmpiricalSearchPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // TODO: Snapshot IR → call agent → annotate module with best variant metadata.
  }
};
} // namespace

std::unique_ptr<Pass> createEmpiricalSearchPass() { return std::make_unique<EmpiricalSearchPass>(); }
} // namespace tessera
