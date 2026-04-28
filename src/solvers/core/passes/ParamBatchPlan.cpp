//===- ParamBatchPlan.cpp — batch parameter sweeps -----------------------*- C++ -*-===//
//
// Walks tessera_solver.param_sweep ops and groups them into execution batches.
// Each sweep axis represents one independent configuration dimension (e.g.
// learning_rate, regularization).  Batching reduces kernel-launch overhead.
//
// Output attrs on each param_sweep op:
//   tessera_solver.param_batch_id   — which batch this op belongs to (int64)
//   tessera_solver.param_batch_size — number of configs in the batch (int64)
//   tessera_solver.param_sweep_total — total sweep configs (int64)
//
// On the module:
//   tessera_solver.num_param_batches — total number of batches created
//
//===----------------------------------------------------------------------===//

#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <vector>

using namespace mlir;

namespace {

struct ParamBatchPlanPass
    : PassWrapper<ParamBatchPlanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParamBatchPlanPass)

  Option<int> batchSize{
      *this, "param-batch-size",
      llvm::cl::desc("Number of sweep configurations per execution batch"),
      llvm::cl::init(8)};

  StringRef getArgument() const final { return "tessera-param-batch-plan"; }
  StringRef getDescription() const final {
    return "Group tessera_solver.param_sweep ops into execution batches";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    // Collect all param_sweep ops.
    std::vector<Operation *> sweeps;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_solver.param_sweep")
        sweeps.push_back(op);
    });

    if (sweeps.empty())
      return;

    int64_t total = static_cast<int64_t>(sweeps.size());
    int64_t bs = std::max(1, batchSize.getValue());
    int64_t numBatches = (total + bs - 1) / bs;

    for (int64_t i = 0; i < total; ++i) {
      int64_t batchId = i / bs;
      // The last batch may be smaller.
      int64_t thisBatchSize = (batchId < numBatches - 1)
                                  ? bs
                                  : total - batchId * bs;

      sweeps[i]->setAttr(
          "tessera_solver.param_batch_id",
          IntegerAttr::get(IntegerType::get(ctx, 64), batchId));
      sweeps[i]->setAttr(
          "tessera_solver.param_batch_size",
          IntegerAttr::get(IntegerType::get(ctx, 64), thisBatchSize));
      sweeps[i]->setAttr(
          "tessera_solver.param_sweep_total",
          IntegerAttr::get(IntegerType::get(ctx, 64), total));
    }

    mod->setAttr(
        "tessera_solver.num_param_batches",
        IntegerAttr::get(IntegerType::get(ctx, 64), numBatches));
  }
};

} // namespace

namespace tessera {
namespace passes {
std::unique_ptr<Pass> createParamBatchPlanPass() {
  return std::make_unique<ParamBatchPlanPass>();
}
} // namespace passes
} // namespace tessera
