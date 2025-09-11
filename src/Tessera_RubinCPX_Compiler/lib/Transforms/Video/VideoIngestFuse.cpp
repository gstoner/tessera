
#include "tessera/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
namespace {
struct FuseVideoIngestPass
    : public PassWrapper<FuseVideoIngestPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseVideoIngestPass)
  StringRef getArgument() const override { return "tessera-fuse-video-ingest"; }
  StringRef getDescription() const override {
    return "Fuse video.decode -> patchify -> tokenizer -> attn.prefill_fused on CPX to retain data in GDDR7";
  }
  void runOnOperation() override {
    // TODO: Pattern: cluster matching video.decode -> ... -> attn.prefill_fused and outline a fused region.
  }
};
}

namespace tessera {
std::unique_ptr<mlir::Pass> createFuseVideoIngestPass() {
  return std::make_unique<FuseVideoIngestPass>();
}
} // namespace tessera
