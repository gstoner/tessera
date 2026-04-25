#include "tessera/Dialect/Neighbors/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace tessera { namespace neighbors {

struct DynamicTopologyPass : public PassWrapper<DynamicTopologyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DynamicTopologyPass)
  StringRef getArgument() const final { return "tessera-topology-dynamic"; }
  StringRef getDescription() const final { return "Insert fences/replan hooks for dynamic topology"; }
  void runOnOperation() override {
    // TODO: detect topology mutations, insert fence tokens and replan calls.
  }
};

void registerDynamicTopologyPass() { PassRegistration<DynamicTopologyPass>(); }

}} // namespace
