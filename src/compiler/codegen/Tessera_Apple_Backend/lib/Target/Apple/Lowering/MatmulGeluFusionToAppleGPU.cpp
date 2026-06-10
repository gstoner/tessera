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

constexpr llvm::StringLiteral kMatmulGeluF32Symbol =
    "tessera_apple_gpu_matmul_gelu_f32";



struct LowerMatmulGeluFusionToAppleGPU : public RewritePattern {
  LowerMatmulGeluFusionToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.gelu", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *geluOp,
                                PatternRewriter &rewriter) const override {
    if (geluOp->getNumOperands() < 1) return failure();
    Value geluIn = geluOp->getOperand(0);

    auto gTy = dyn_cast<RankedTensorType>(geluIn.getType());
    if (!gTy || gTy.getRank() != 2)
      return rewriter.notifyMatchFailure(geluOp, "matmul_gelu fusion: rank-2 only");
    if (!gTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(geluOp, "matmul_gelu fusion: f32 only");

    Operation *defOp = geluIn.getDefiningOp();
    if (!defOp)
      return rewriter.notifyMatchFailure(geluOp, "matmul_gelu fusion: no defining op");
    if (defOp->getName().getStringRef() != "tessera.matmul")
      return rewriter.notifyMatchFailure(geluOp, "matmul_gelu fusion: defining op is not tessera.matmul");
    if (!geluIn.hasOneUse())
      return rewriter.notifyMatchFailure(geluOp, "matmul_gelu fusion: matmul result has multiple uses");

    Operation *matmulOp = defOp;
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
    MLIRContext *ctx = geluOp->getContext();

    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();
    Type f32Ty = rewriter.getF32Type();

    auto aMemTy = MemRefType::get({M, K}, f32Ty);
    auto bMemTy = MemRefType::get({K, N}, f32Ty);
    auto oMemTy = MemRefType::get({M, N}, f32Ty);

    rewriter.setInsertionPoint(matmulOp);
    Value aPtr = extractPtr(rewriter, loc, lhs, aMemTy);
    Value bPtr = extractPtr(rewriter, loc, rhs, bMemTy);
    auto oAlloc = rewriter.create<memref::AllocOp>(loc, oMemTy);
    Value oPtr;
    {
      auto pi =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, oAlloc);
      oPtr = rewriter.create<arith::IndexCastOp>(loc, i64Ty, pi);
    }

    Value Mv = rewriter.create<arith::ConstantIntOp>(loc, M, 32);
    Value Nv = rewriter.create<arith::ConstantIntOp>(loc, N, 32);
    Value Kv = rewriter.create<arith::ConstantIntOp>(loc, K, 32);

    FunctionType fnTy = FunctionType::get(
        ctx, {i64Ty, i64Ty, i64Ty, i32Ty, i32Ty, i32Ty}, {});
    ensureExternalDecl(mod, kMatmulGeluF32Symbol, fnTy);

    rewriter.create<func::CallOp>(
        loc, kMatmulGeluF32Symbol, TypeRange{},
        ValueRange{aPtr, bPtr, oPtr, Mv, Nv, Kv});

    auto outTensorTy = RankedTensorType::get({M, N}, f32Ty);
    Value result =
        rewriter.create<bufferization::ToTensorOp>(loc, outTensorTy, oAlloc);
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
