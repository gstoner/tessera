#include "tessera/Transforms/ParallelDecodeExpand.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct ParallelDecodeExpandPass : public PassWrapper<ParallelDecodeExpandPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelDecodeExpandPass)
  ParallelDecodeExpandPass() = default;
  StringRef getArgument() const override { return "tessera-parallel-decode-expand"; }
  StringRef getDescription() const override { return "Expand tessera.graph.parallel_decode{K} into K branch regions with shared cache handles."; }
  void runOnOperation() override {
    ModuleOp m = getOperation();
    // TODO: Find tessera.graph.parallel_decode ops, clone subgraphs into per-branch regions,
    // wire deps and produce a branches SSA handle.
    m.walk([&](Operation *op) {
      (void)op; // placeholder
    });
  }
};
} // namespace

std::unique_ptr<Pass> tessera::createParallelDecodeExpandPass() { return std::make_unique<ParallelDecodeExpandPass>(); }

void tessera::registerParallelDecodeExpandPass() {
  PassRegistration<ParallelDecodeExpandPass>();
}
