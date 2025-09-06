#include "tessera/Transforms/KVCacheBlockify.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct KVCacheBlockifyPass : public PassWrapper<KVCacheBlockifyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KVCacheBlockifyPass)
  KVCacheBlockifyPass() = default;
  StringRef getArgument() const override { return "tessera-kv-cache-blockify"; }
  StringRef getDescription() const override { return "Insert kv_block_read/pack around attention ops and add window attrs."; }
  void runOnOperation() override {
    ModuleOp m = getOperation();
    // TODO: Pattern: tile.attn_bidir -> kv_block_read + attn_bidir + kv_block_pack (if writing new blocks).
    m.walk([&](Operation *op) {
      (void)op; // placeholder
    });
  }
};
} // namespace

std::unique_ptr<Pass> tessera::createKVCacheBlockifyPass() { return std::make_unique<KVCacheBlockifyPass>(); }

void tessera::registerKVCacheBlockifyPass() {
  PassRegistration<KVCacheBlockifyPass>();
}
