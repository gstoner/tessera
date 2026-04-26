
#include "tessera/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BlockAndValueMapping.h"

using namespace mlir;

static bool isOp(Operation* op, StringRef name) {
  return op && op->getName().getStringRef() == name;
}

namespace {
struct FuseVideoIngest : RewritePattern {
  FuseVideoIngest(MLIRContext *ctx) : RewritePattern("tessera.target.cpx.video.decode", 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *decode, PatternRewriter &rewriter) const override {
    // Expect chain: video.decode -> (tessera.video.patchify) -> (tessera.video.tokenize) -> attn.prefill_fused
    Operation *patch = nullptr, *tok = nullptr, *prefill = nullptr;
    if (!decode->getResult(0).use_empty()) {
      for (Operation *user : decode->getResult(0).getUsers()) {
        if (user->getName().getStringRef() == "tessera.video.patchify") patch = user;
      }
    }
    if (!patch || patch->getResult(0).use_empty()) return failure();
    for (Operation *user : patch->getResult(0).getUsers()) {
      if (user->getName().getStringRef() == "tessera.video.tokenize") tok = user;
    }
    if (!tok || tok->getResult(0).use_empty()) return failure();
    for (Operation *user : tok->getResult(0).getUsers()) {
      if (user->getName().getStringRef() == "tessera.target.cpx.attn.prefill_fused") prefill = user;
    }
    if (!prefill) return failure();

    Location loc = decode->getLoc();
    // Construct fused region op
    OperationState st(loc, "tessera.target.cpx.video.ingest_fused");
    Region* region = st.addRegion();
    auto fused = rewriter.create(st);
    Region &r = fused->getRegion(0);
    r.push_back(new Block());
    rewriter.setInsertionPointToStart(&r.front());

    // Clone the chain into the region
    BlockAndValueMapping bvm;
    auto d2 = rewriter.clone(*decode, bvm);
    auto p2 = rewriter.clone(*patch, bvm);
    auto t2 = rewriter.clone(*tok, bvm);
    auto a2 = rewriter.clone(*prefill, bvm);

    // Return terminator
    rewriter.create<func::ReturnOp>(loc, ValueRange{});
    rewriter.setInsertionPointAfter(fused);

    // Replace consumers of prefill result (if any) with nothing in skeleton
    prefill->dropAllUses();
    // Erase original ops
    prefill->erase(); tok->erase(); patch->erase(); decode->erase();

    return success();
  }
};

struct FuseVideoIngestPass
  : public PassWrapper<FuseVideoIngestPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseVideoIngestPass)
  StringRef getArgument() const override { return "tessera-fuse-video-ingest"; }
  StringRef getDescription() const override { return "Fuse decode->patchify->tokenize->prefill into video.ingest_fused"; }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseVideoIngest>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createFuseVideoIngestPass() { return std::make_unique<FuseVideoIngestPass>(); }
} // namespace tessera
