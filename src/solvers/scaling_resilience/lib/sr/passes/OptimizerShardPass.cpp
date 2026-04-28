//===- OptimizerShardPass.cpp — ZeRO-2 optimizer state partitioning ------*- C++ -*-===//
//
// Implements ZeRO stage-2 partitioning: momentum and variance (optimizer state)
// are evenly divided across the DP mesh axis.
//
// For each tessera.optimizer.* op or function arg tagged
// tessera_sr.optimizer_state = "momentum" | "variance":
//
//   tessera_sr.sharded         — UnitAttr (marks as partitioned)
//   tessera_sr.shard_axis      — axis name (default "dp")
//   tessera_sr.shard_rank      — which slice this rank holds (set to * for IR)
//   tessera_sr.zero_stage      — 2 (int64)
//   tessera_sr.partition_count — num_dp_ranks (int64)
//
// On the module:
//   tessera_sr.zero_stage = 2
//   tessera_sr.dp_axis    = "dp"
//
//===----------------------------------------------------------------------===//

#include "tessera/sr/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct OptimizerShardPass
    : public PassWrapper<OptimizerShardPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizerShardPass)

  Option<std::string> dpAxis{
      *this, "dp-axis",
      llvm::cl::desc("Data-parallel mesh axis name"),
      llvm::cl::init(std::string("dp"))};

  Option<int> numDPRanks{
      *this, "num-dp-ranks",
      llvm::cl::desc("Number of data-parallel ranks for ZeRO partitioning"),
      llvm::cl::init(1)};

  Option<int> zeroStage{
      *this, "zero-stage",
      llvm::cl::desc("ZeRO stage: 1, 2, or 3"),
      llvm::cl::init(2)};

  StringRef getArgument() const final { return "tessera-optimizer-shard"; }
  StringRef getDescription() const final {
    return "ZeRO-2: partition optimizer states (momentum/variance) across DP "
           "mesh";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    int64_t nRanks = numDPRanks;
    if (auto attr = mod->getAttrOfType<IntegerAttr>("tessera.num_dp_ranks"))
      nRanks = attr.getInt();
    if (nRanks < 1) nRanks = 1;

    StringRef axis = dpAxis;
    if (auto attr = mod->getAttrOfType<StringAttr>("tessera.dp_axis"))
      axis = attr.getValue();

    int64_t stage = zeroStage;

    mod.walk([&](Operation *op) {
      StringRef opName = op->getName().getStringRef();

      // Match optimizer state ops.
      bool isOptimizerOp =
          opName.contains("optimizer") || opName.contains("momentum") ||
          opName.contains("variance") || opName.contains("adam") ||
          opName.endswith("optimizer.shard") ||
          op->hasAttr("tessera_sr.optimizer_state");

      if (!isOptimizerOp)
        return;

      op->setAttr("tessera_sr.sharded", UnitAttr::get(ctx));
      op->setAttr("tessera_sr.shard_axis", StringAttr::get(ctx, axis));
      op->setAttr("tessera_sr.zero_stage",
                  IntegerAttr::get(IntegerType::get(ctx, 64), stage));
      op->setAttr("tessera_sr.partition_count",
                  IntegerAttr::get(IntegerType::get(ctx, 64), nRanks));

      // Stage 3: also tag parameter tensors.
      if (stage >= 3) {
        op->setAttr("tessera_sr.params_sharded", UnitAttr::get(ctx));
      }
    });

    // Annotate module.
    mod->setAttr("tessera_sr.zero_stage",
                 IntegerAttr::get(IntegerType::get(ctx, 64), stage));
    mod->setAttr("tessera_sr.dp_axis", StringAttr::get(ctx, axis));
    mod->setAttr("tessera_sr.num_dp_ranks",
                 IntegerAttr::get(IntegerType::get(ctx, 64), nRanks));
  }
};

} // namespace

std::unique_ptr<Pass> mlir::tessera::sr::createOptimizerShardPass() {
  return std::make_unique<OptimizerShardPass>();
}
