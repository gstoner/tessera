
// Plan sliding-window sizes (W) and stride per layer/sequence tile.
// cmd: -tessera-atlas-window-plan
#include "mlir/Pass/Pass.h"
using namespace mlir;

namespace {
struct AtlasWindowPlannerPass : PassWrapper<AtlasWindowPlannerPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AtlasWindowPlannerPass)
  StringRef getArgument() const final { return "tessera-atlas-window-plan"; }
  StringRef getDescription() const final { return "Plan Omega-rule sliding windows and annotate ops."; }
  void runOnOperation() override {
    // TODO: Heuristics: fit in HBM/L2, align to KV cache paging, overlap compute/comm.
  }
};
} // namespace

std::unique_ptr<Pass> createAtlasWindowPlannerPass() { return std::make_unique<AtlasWindowPlannerPass>(); }
