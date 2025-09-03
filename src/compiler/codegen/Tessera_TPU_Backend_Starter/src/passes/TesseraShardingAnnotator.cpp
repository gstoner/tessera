#include "tessera/tpu/TesseraTPUPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct AnnotateShardingPass
    : PassWrapper<AnnotateShardingPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "tessera-annotate-sharding"; }
  StringRef getDescription() const override { return "Attach prototype sharding attrs (mhlo.sharding) for GSPMD."; }

  // Very simple demo: add a replicated sharding to all functions' arguments.
  void runOnOperation() override {
    ModuleOp m = getOperation();
    for (auto func : m.getOps<FuncOp>()) {
      for (auto &op : func.getBody().front()) {
        // Annotate dot_general/add if present.
        if (op.getName().getStringRef().contains("stablehlo.")) {
          // Example sharding: "replicated"
          op.setAttr("mhlo.sharding", StringAttr::get(m.getContext(), "{replicated}"));
        }
      }
    }
  }
};
} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createAnnotateShardingPass() {
  return std::make_unique<AnnotateShardingPass>();
}
} // namespace tessera
