#include "tessera/Dialect/Neighbors/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace tessera { namespace neighbors {

struct PipelineOverlapPass : public PassWrapper<PipelineOverlapPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PipelineOverlapPass)
  StringRef getArgument() const final { return "tessera-pipeline-overlap"; }
  StringRef getDescription() const final { return "Insert async overlap of halo exchange and compute"; }
  void runOnOperation() override {
    // TODO: place pack/comm/unpack ops on distinct streams via attributes/tokens.
  }
};

void registerPipelineOverlapPass() { PassRegistration<PipelineOverlapPass>(); }

}} // namespace
