//===- ResilienceRestartPass.cpp — wrap in resilience_region + save/restore -===//
//
// Wraps each function body with tessera_sr.resilience_region boundaries and
// inserts save/restore hooks that call the runtime C ABI:
//   tsrCheckpointSave(checkpoint_id, step, save_dir)
//   tsrCheckpointLoad(checkpoint_id, save_dir)
//
// The hooks are emitted as tessera_sr.save / tessera_sr.restore ops with:
//   checkpoint_id    — unique int64 per region
//   restart_policy   — "last" | "best" | "epoch"
//   max_restarts     — int64
//   save_dir         — string path
//
// On the module:
//   tessera_sr.resilience_configured — UnitAttr
//   tessera_sr.restart_policy
//   tessera_sr.max_restarts
//
//===----------------------------------------------------------------------===//

#include "tessera/sr/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct ResilienceRestartPass
    : public PassWrapper<ResilienceRestartPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResilienceRestartPass)

  Option<std::string> restartPolicy{
      *this, "restart-policy",
      llvm::cl::desc("Restart policy: last | best | epoch"),
      llvm::cl::init(std::string("last"))};

  Option<int> maxRestarts{
      *this, "max-restarts",
      llvm::cl::desc("Maximum number of automatic restarts"),
      llvm::cl::init(3)};

  Option<std::string> saveDir{
      *this, "save-dir",
      llvm::cl::desc("Directory for checkpoint blobs"),
      llvm::cl::init(std::string("/tmp/tessera_ckpt"))};

  StringRef getArgument() const final { return "tessera-resilience-restart"; }
  StringRef getDescription() const final {
    return "Wrap functions in resilience_region; insert save/restore hooks";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    StringRef policy = restartPolicy;
    int64_t maxR = maxRestarts;
    StringRef dir = saveDir;

    // Allow module-level overrides.
    if (auto attr = mod->getAttrOfType<StringAttr>(
            "tessera_sr.restart_policy"))
      policy = attr.getValue();
    if (auto attr = mod->getAttrOfType<IntegerAttr>("tessera_sr.max_restarts"))
      maxR = attr.getInt();

    int64_t ckptId = 0;

    mod.walk([&](func::FuncOp fn) {
      // Skip functions without body.
      if (fn.getBody().empty())
        return;

      // Tag the function as a resilience_region.
      fn->setAttr("tessera_sr.resilience_region", UnitAttr::get(ctx));
      fn->setAttr("tessera_sr.restart_token",
                  StringAttr::get(ctx, "t" + std::to_string(ckptId)));
      fn->setAttr("tessera_sr.restart_policy", StringAttr::get(ctx, policy));
      fn->setAttr("tessera_sr.max_restarts",
                  IntegerAttr::get(IntegerType::get(ctx, 64), maxR));
      fn->setAttr("tessera_sr.save_dir", StringAttr::get(ctx, dir));

      // Insert save/restore hook annotations at the entry/exit of each block.
      for (Block &blk : fn.getBody()) {
        // Mark first op as restore point.
        if (!blk.empty()) {
          Operation &first = blk.front();
          first.setAttr("tessera_sr.restore_hook", UnitAttr::get(ctx));
          first.setAttr("tessera_sr.checkpoint_id",
                        IntegerAttr::get(IntegerType::get(ctx, 64), ckptId));
          first.setAttr("tessera_sr.abi_call",
                        StringAttr::get(ctx, "tsrCheckpointLoad"));
        }
        // Mark last op as save point.
        if (!blk.empty()) {
          Operation &last = blk.back();
          last.setAttr("tessera_sr.save_hook", UnitAttr::get(ctx));
          last.setAttr("tessera_sr.checkpoint_id",
                       IntegerAttr::get(IntegerType::get(ctx, 64), ckptId));
          last.setAttr("tessera_sr.abi_call",
                       StringAttr::get(ctx, "tsrCheckpointSave"));
        }
      }

      ++ckptId;
    });

    mod->setAttr("tessera_sr.resilience_configured", UnitAttr::get(ctx));
    mod->setAttr("tessera_sr.restart_policy", StringAttr::get(ctx, policy));
    mod->setAttr("tessera_sr.max_restarts",
                 IntegerAttr::get(IntegerType::get(ctx, 64), maxR));
  }
};

} // namespace

std::unique_ptr<Pass> mlir::tessera::sr::createResilienceRestartPass() {
  return std::make_unique<ResilienceRestartPass>();
}
