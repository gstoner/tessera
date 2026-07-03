//===- MatmulGeluFusionToAppleGPU.cpp ------------------------------------===//
//
// Phase 8.4.7 — Apple GPU MSL fusion for the MLP block activation:
// matmul -> gelu. Pattern matches a 2-op SSA chain
//
//   %m = tessera.matmul %A, %B           : (M, K) x (K, N) -> (M, N)
//   %o = tessera.gelu   %m               : (M, N) -> (M, N)
//
// and replaces both with a single func.call into the Apple-GPU runtime
// shim's matmul_gelu_f32 kernel. Mirrors the Phase 8.4.3 matmul -> softmax
// fusion structurally — just a different post-matmul postlude.
//
// Constraints (Phase 8.4.7):
//   - rank-2 f32 inputs/output
//   - static shapes
//   - matmul result has exactly one use (the gelu)
//   - N <= 256 (per-thread stack array bound)
//
//===----------------------------------------------------------------------===//

#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/FusionChainUtils.h"
#include "Tessera/Target/Apple/LoweringUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace ::mlir;

namespace tessera {
namespace apple {

namespace {

// Optimizing-Compiler Plan F2a — matmul -> gelu lowers to the generic
// SYNTHESIZED epilogue kernel (the epilogue carried as a region descriptor
// attribute), retiring the per-epilogue hand-written matmul_gelu_f32 kernel.
constexpr llvm::StringLiteral kSynthEpilogueF32Symbol =
    "tessera_apple_gpu_synth_matmul_epilogue_f32";



struct LowerMatmulGeluFusionToAppleGPU : public RewritePattern {
  LowerMatmulGeluFusionToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.gelu", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *geluOp,
                                PatternRewriter &rewriter) const override {
    if (geluOp->getNumOperands() < 1) return failure();
    Value geluIn = geluOp->getOperand(0);

    // Decision #19 — consume the compiler's fusion descriptor when present.
    bool descriptorDriven =
        tessera::apple::fusionDescriptorDriven(geluOp, "matmul_gelu");

    auto gTy = dyn_cast<RankedTensorType>(geluIn.getType());
    if (!gTy || gTy.getRank() != 2)
      return rewriter.notifyMatchFailure(geluOp, "matmul_gelu fusion: rank-2 only");
    if (!gTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(geluOp, "matmul_gelu fusion: f32 only");

    auto matmulOr = tessera::apple::walkChainProducer(
        rewriter, geluOp, geluIn, "tessera.matmul", descriptorDriven);
    if (failed(matmulOr))
      return failure();
    Operation *matmulOp = *matmulOr;
    if (matmulOp->getNumOperands() < 2) return failure();
    Value lhs = matmulOp->getOperand(0);
    Value rhs = matmulOp->getOperand(1);

    auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsTy || !rhsTy || lhsTy.getRank() != 2 || rhsTy.getRank() != 2)
      return rewriter.notifyMatchFailure(geluOp, "matmul_gelu fusion: matmul inputs not rank-2");
    if (!lhsTy.getElementType().isF32() || !rhsTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(geluOp, "matmul_gelu fusion: matmul inputs not f32");
    if (lhsTy.isDynamicDim(0) || lhsTy.isDynamicDim(1) ||
        rhsTy.isDynamicDim(0) || rhsTy.isDynamicDim(1))
      return rewriter.notifyMatchFailure(geluOp, "matmul_gelu fusion: requires static shapes");

    int64_t M = lhsTy.getDimSize(0);
    int64_t K = lhsTy.getDimSize(1);
    int64_t N = rhsTy.getDimSize(1);
    if (rhsTy.getDimSize(0) != K)
      return rewriter.notifyMatchFailure(geluOp, "matmul_gelu fusion: matmul K mismatch");
    if (N > 256)
      return rewriter.notifyMatchFailure(
          geluOp, "matmul_gelu fusion: GPU kernel limited to N <= 256");

    Location loc = geluOp->getLoc();
    ModuleOp mod = geluOp->getParentOfType<ModuleOp>();
    Type f32Ty = rewriter.getF32Type();

    auto aMemTy = MemRefType::get({M, K}, f32Ty);
    auto bMemTy = MemRefType::get({K, N}, f32Ty);
    auto oMemTy = MemRefType::get({M, N}, f32Ty);
    auto outTensorTy = RankedTensorType::get({M, N}, f32Ty);

    SmallVector<NamedAttribute> desc{
        rewriter.getNamedAttr("tessera.fusion.kernel",
                              rewriter.getStringAttr("synth_matmul_epilogue")),
        rewriter.getNamedAttr("tessera.fusion.epilogue",
                              rewriter.getStringAttr("gelu")),
        rewriter.getNamedAttr(
            "tessera.fusion.source",
            rewriter.getStringAttr(descriptorDriven ? "descriptor"
                                                    : "rediscovered"))};

    rewriter.setInsertionPoint(matmulOp);
    Value result = tessera::common::emitFusionCall(
        rewriter, loc, mod, kSynthEpilogueF32Symbol,
        {{lhs, aMemTy}, {rhs, bMemTy}}, oMemTy, outTensorTy, {M, N, K}, desc);
    rewriter.replaceOp(geluOp, result);
    rewriter.eraseOp(matmulOp);
    return success();
  }
};

struct LowerMatmulGeluFusionToAppleGPUPass
    : public PassWrapper<LowerMatmulGeluFusionToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMatmulGeluFusionToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-matmul-gelu-fusion-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Fuse tessera.matmul -> tessera.gelu (rank-2, f32, N <= 256) into "
           "a single Apple GPU runtime call (custom MSL kernel)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerMatmulGeluFusionToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerMatmulGeluFusionToAppleGPUPass() {
  return std::make_unique<LowerMatmulGeluFusionToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
