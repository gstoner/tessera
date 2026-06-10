//===- RowOpToAppleGPU.cpp - Lower Tier-1 row ops to MPSGraph lane -------===//
//
// 2026-05-29 — Apple GPU MPSGraph lane.
//
// Lowers the last-axis row ops to MPSGraph-backed runtime calls:
//
//   tessera.layer_norm   -> tessera_apple_gpu_layer_norm_f32(x, g, b, out, rows, cols, eps)
//   tessera.rmsnorm[_safe]-> tessera_apple_gpu_rmsnorm_gpu_f32(x, g, out, rows, cols, eps)
//   tessera.log_softmax  -> tessera_apple_gpu_log_softmax_f32(x, out, rows, cols)
//
// tessera.layer_norm / tessera.rmsnorm are *unweighted* (gamma=1, beta=0) in
// the Tessera op surface, so the gamma/beta pointers are passed as null; the
// runtime builds the unweighted graph. Any rank folds to [rows, cols] with
// cols = last dim.
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

// kind: 0 layer_norm, 1 rmsnorm, 3 log_softmax (matching the runtime rowop).
static int rowKind(StringRef name) {
  return llvm::StringSwitch<int>(name)
      .Case("tessera.layer_norm", 0)
      .Case("tessera.rmsnorm", 1)
      .Case("tessera.rmsnorm_safe", 1)
      .Case("tessera.log_softmax", 3)
      .Default(-1);
}



struct LowerRowOpToAppleGPU : public RewritePattern {
  LowerRowOpToAppleGPU(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    StringRef name = op->getName().getStringRef();
    int kind = rowKind(name);
    if (kind < 0)
      return failure();
    if (op->getNumOperands() < 1 || op->getNumResults() != 1)
      return failure();
    Value x = op->getOperand(0);
    auto xTy = dyn_cast<RankedTensorType>(x.getType());
    if (!xTy || !xTy.hasStaticShape() || xTy.getRank() < 1)
      return rewriter.notifyMatchFailure(op, "AppleGPU row-op lane needs a static-shape tensor");
    Type elem = xTy.getElementType();
    bool isF16 = elem.isF16();
    if (!elem.isF32() && !isF16)
      return rewriter.notifyMatchFailure(op, "AppleGPU row-op lane supports f32/f16");

    auto shape = xTy.getShape();
    int64_t cols = shape[xTy.getRank() - 1];
    int64_t rows = 1;
    for (int64_t i = 0; i + 1 < xTy.getRank(); ++i)
      rows *= shape[i];

    // eps: explicit attr wins; otherwise the Tessera default (1e-5, or 1e-6
    // for rmsnorm_safe).
    double eps = name == "tessera.rmsnorm_safe" ? 1e-6 : 1e-5;
    if (auto e = op->getAttrOfType<FloatAttr>("eps"))
      eps = e.getValueAsDouble();

    Location loc = op->getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = op->getContext();
    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();
    Type f32Ty = rewriter.getF32Type();
    auto memTy = MemRefType::get(shape, elem);

    Value xPtr = extractPtr(rewriter, loc, x, memTy);
    auto outAlloc = rewriter.create<memref::AllocOp>(loc, memTy);
    auto outIdx =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outAlloc);
    Value outPtr = rewriter.create<arith::IndexCastOp>(loc, i64Ty, outIdx);
    Value rowsV = rewriter.create<arith::ConstantIntOp>(loc, rows, 32);
    Value colsV = rewriter.create<arith::ConstantIntOp>(loc, cols, 32);
    Value nullPtr = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);

    if (kind == 0) {  // layer_norm (unweighted: gamma=beta=null)
      StringRef sym = isF16 ? "tessera_apple_gpu_layer_norm_f16"
                            : "tessera_apple_gpu_layer_norm_f32";
      Value epsV = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr((float)eps));
      FunctionType fnTy = FunctionType::get(
          ctx, {i64Ty, i64Ty, i64Ty, i64Ty, i32Ty, i32Ty, f32Ty}, {});
      ensureExternalDecl(mod, sym, fnTy);
      rewriter.create<func::CallOp>(
          loc, sym, TypeRange{},
          ValueRange{xPtr, nullPtr, nullPtr, outPtr, rowsV, colsV, epsV});
    } else if (kind == 1) {  // rmsnorm (unweighted: gamma=null)
      StringRef sym = isF16 ? "tessera_apple_gpu_rmsnorm_gpu_f16"
                            : "tessera_apple_gpu_rmsnorm_gpu_f32";
      Value epsV = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr((float)eps));
      FunctionType fnTy = FunctionType::get(
          ctx, {i64Ty, i64Ty, i64Ty, i32Ty, i32Ty, f32Ty}, {});
      ensureExternalDecl(mod, sym, fnTy);
      rewriter.create<func::CallOp>(
          loc, sym, TypeRange{},
          ValueRange{xPtr, nullPtr, outPtr, rowsV, colsV, epsV});
    } else {  // log_softmax
      StringRef sym = isF16 ? "tessera_apple_gpu_log_softmax_f16"
                            : "tessera_apple_gpu_log_softmax_f32";
      FunctionType fnTy =
          FunctionType::get(ctx, {i64Ty, i64Ty, i32Ty, i32Ty}, {});
      ensureExternalDecl(mod, sym, fnTy);
      rewriter.create<func::CallOp>(loc, sym, TypeRange{},
                                    ValueRange{xPtr, outPtr, rowsV, colsV});
    }

    Value result = rewriter.create<bufferization::ToTensorOp>(
        loc, RankedTensorType::get(shape, elem), outAlloc);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerRowOpToAppleGPUPass
    : public PassWrapper<LowerRowOpToAppleGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRowOpToAppleGPUPass)

  StringRef getArgument() const override { return "tessera-rowop-to-apple_gpu"; }
  StringRef getDescription() const override {
    return "Lower Tier-1 last-axis row ops (layer_norm/rmsnorm/rmsnorm_safe/"
           "log_softmax) to Apple GPU MPSGraph runtime calls";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerRowOpToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerRowOpToAppleGPUPass() {
  return std::make_unique<LowerRowOpToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
