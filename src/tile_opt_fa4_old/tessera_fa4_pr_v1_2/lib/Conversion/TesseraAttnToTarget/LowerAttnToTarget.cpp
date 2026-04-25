//===- LowerAttnToTarget.cpp (v1.2) ----------------------------------------===//
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace tessera { namespace attn {

struct LowerAttnToTargetPass : public PassWrapper<LowerAttnToTargetPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAttnToTargetPass)
  void runOnOperation() override {
    // Lower lse.save to a memref/tensor materialization op; lse.load to a simple read.
  }
};

std::unique_ptr<Pass> createLowerAttnToTargetPass() { return std::make_unique<LowerAttnToTargetPass>(); }

}} // ns
