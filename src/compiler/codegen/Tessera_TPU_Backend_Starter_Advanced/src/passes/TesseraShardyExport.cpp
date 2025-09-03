#include "tessera/tpu/TesseraTPUPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct ShardyExportPass
    : PassWrapper<ShardyExportPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "tessera-export-shardy"; }
  StringRef getDescription() const override { return "Attach Shardy-native sharding attrs (sdy.tensor_sharding, sdy.mesh)."; }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();

    // Example: attach a mesh annotation at module scope, and per-op shardings.
    // In a real pipeline, you'd translate Tessera's #mesh(...) and distribute(...) to these.
    if (!m->hasAttr("sdy.mesh")) {
      // Annotate a 2D mesh: data x model
      auto meshStr = StringAttr::get(ctx, "mesh = {axes = [\"data\",\"model\"], shape = [D,M]}");
      m->setAttr("sdy.mesh", meshStr);
    }

    for (auto func : m.getOps<FuncOp>()) {
      for (auto &op : func.getBody().front()) {
        // Where we see stablehlo ops, attach a replicated sharding as a placeholder.
        if (op.getName().getStringRef().startswith("stablehlo.")) {
          op.setAttr("sdy.tensor_sharding", StringAttr::get(ctx, "{sharding = replicated}"));
        }
      }
    }
  }
};
} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createExportShardyPass() {
  return std::make_unique<ShardyExportPass>();
}
} // namespace tessera
