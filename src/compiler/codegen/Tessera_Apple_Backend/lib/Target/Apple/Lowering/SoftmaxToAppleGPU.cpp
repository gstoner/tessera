//===- SoftmaxToAppleGPU.cpp - Lower tessera.softmax to MSL kernel -------===//
//
// Phase 8.4.2 — Apple GPU custom MSL softmax kernel.
//
// Replaces tessera.softmax ops (rank-2, f32, axis=-1) with calls to the
// Apple-GPU runtime shim:
//
//   tessera_apple_gpu_softmax_f32(X, Out, M, K)
//
// One thread per row in the runtime kernel; the IR-level ABI is the same
// flat memref-pointer pattern as RopeToAppleGPU and FlashAttnToAppleGPU.
// Higher-rank tensors fall through to the artifact-only path until a
// follow-up phase generalizes the kernel.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Target/Apple/Passes.h"

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

constexpr llvm::StringLiteral kSoftmaxF32Symbol =
    "tessera_apple_gpu_softmax_f32";
constexpr llvm::StringLiteral kSoftmaxF16Symbol =
    "tessera_apple_gpu_softmax_f16";
constexpr llvm::StringLiteral kSoftmaxBF16Symbol =
    "tessera_apple_gpu_softmax_bf16";

static func::FuncOp ensureExternalDecl(ModuleOp mod, StringRef name,
                                       FunctionType fnTy) {
  if (auto fn = mod.lookupSymbol<func::FuncOp>(name))
    return fn;
  OpBuilder b(mod.getBodyRegion());
  b.setInsertionPointToStart(mod.getBody());
  auto fn = b.create<func::FuncOp>(mod.getLoc(), name, fnTy);
  fn.setPrivate();
  return fn;
}

static Value extractPtr(OpBuilder &b, Location loc, Value tensor,
                        MemRefType memTy) {
  auto buf = b.create<bufferization::ToBufferOp>(loc, memTy, tensor);
  auto ptrIdx = b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buf);
  return b.create<arith::IndexCastOp>(loc, b.getI64Type(), ptrIdx);
}

struct LowerSoftmaxToAppleGPU : public RewritePattern {
  LowerSoftmaxToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.softmax", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 1)
      return failure();
    Value x = op->getOperand(0);
    auto xTy = dyn_cast<RankedTensorType>(x.getType());
    if (!xTy || xTy.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "AppleGPU softmax MSL path is rank-2 only in Phase 8.4.2");
    Type xElem = xTy.getElementType();
    StringRef symbol;
    if (xElem.isF32()) {
      symbol = kSoftmaxF32Symbol;
    } else if (xElem.isF16()) {
      symbol = kSoftmaxF16Symbol;
    } else if (xElem.isBF16()) {
      symbol = kSoftmaxBF16Symbol;
    } else {
      return rewriter.notifyMatchFailure(
          op, "AppleGPU softmax MSL path supports f32, f16, and bf16 in Phase 8.4.4.1");
    }
    if (xTy.isDynamicDim(0) || xTy.isDynamicDim(1))
      return rewriter.notifyMatchFailure(
          op, "AppleGPU softmax MSL path requires static shapes");

    // axis: defaults to -1 (innermost). Other axes fall back to the
    // artifact path until the runtime kernel grows axis support.
    int64_t axis = -1;
    if (auto attr = op->getAttrOfType<IntegerAttr>("axis"))
      axis = attr.getInt();
    if (axis != -1 && axis != xTy.getRank() - 1)
      return rewriter.notifyMatchFailure(
          op, "AppleGPU softmax MSL path requires axis=-1 in Phase 8.4.2");

    int64_t M = xTy.getDimSize(0);
    int64_t K = xTy.getDimSize(1);

    Location loc = op->getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = op->getContext();

    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();

    auto memTy = MemRefType::get({M, K}, xElem);
    Value xPtr = extractPtr(rewriter, loc, x, memTy);
    auto outAlloc = rewriter.create<memref::AllocOp>(loc, memTy);
    Value outPtr;
    {
      auto pi =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outAlloc);
      outPtr = rewriter.create<arith::IndexCastOp>(loc, i64Ty, pi);
    }

    Value Mv = rewriter.create<arith::ConstantIntOp>(loc, M, 32);
    Value Kv = rewriter.create<arith::ConstantIntOp>(loc, K, 32);

    FunctionType fnTy =
        FunctionType::get(ctx, {i64Ty, i64Ty, i32Ty, i32Ty}, {});
    ensureExternalDecl(mod, symbol, fnTy);

    rewriter.create<func::CallOp>(
        loc, symbol, TypeRange{},
        ValueRange{xPtr, outPtr, Mv, Kv});

    auto outTensorTy = RankedTensorType::get({M, K}, xElem);
    Value result =
        rewriter.create<bufferization::ToTensorOp>(loc, outTensorTy, outAlloc);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerSoftmaxToAppleGPUPass
    : public PassWrapper<LowerSoftmaxToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerSoftmaxToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-softmax-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Lower tessera.softmax (rank-2, f32/f16/bf16, axis=-1) to Apple "
           "GPU runtime calls (custom MSL kernel)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerSoftmaxToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerSoftmaxToAppleGPUPass() {
  return std::make_unique<LowerSoftmaxToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
