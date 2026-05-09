//===- GeluToAppleGPU.cpp - Lower tessera.gelu to MSL kernel -------------===//
//
// Phase 8.4.2 — Apple GPU custom MSL gelu kernel.
//
// Replaces tessera.gelu ops (rank-2, f32) with calls to the Apple-GPU
// runtime shim:
//
//   tessera_apple_gpu_gelu_f32(X, Out, N)
//
// The kernel is rank-agnostic at the runtime layer (one thread per element);
// the Phase 8.4.2 lowering pass restricts to rank-2 to keep the memref layout
// straightforward. Higher-rank tensors fall through to the artifact-only path.
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

constexpr llvm::StringLiteral kGeluF32Symbol = "tessera_apple_gpu_gelu_f32";
constexpr llvm::StringLiteral kGeluF16Symbol = "tessera_apple_gpu_gelu_f16";
constexpr llvm::StringLiteral kGeluBF16Symbol = "tessera_apple_gpu_gelu_bf16";

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

struct LowerGeluToAppleGPU : public RewritePattern {
  LowerGeluToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.gelu", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 1)
      return failure();
    Value x = op->getOperand(0);
    auto xTy = dyn_cast<RankedTensorType>(x.getType());
    if (!xTy || xTy.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "AppleGPU gelu MSL path is rank-2 only in Phase 8.4.2");
    Type xElem = xTy.getElementType();
    StringRef symbol;
    if (xElem.isF32()) {
      symbol = kGeluF32Symbol;
    } else if (xElem.isF16()) {
      symbol = kGeluF16Symbol;
    } else if (xElem.isBF16()) {
      symbol = kGeluBF16Symbol;
    } else {
      return rewriter.notifyMatchFailure(
          op, "AppleGPU gelu MSL path supports f32, f16, and bf16 in Phase 8.4.4.1");
    }
    if (xTy.isDynamicDim(0) || xTy.isDynamicDim(1))
      return rewriter.notifyMatchFailure(
          op, "AppleGPU gelu MSL path requires static shapes");

    int64_t M = xTy.getDimSize(0);
    int64_t K = xTy.getDimSize(1);
    int64_t N = M * K;

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

    Value Nv = rewriter.create<arith::ConstantIntOp>(loc, N, 32);

    FunctionType fnTy =
        FunctionType::get(ctx, {i64Ty, i64Ty, i32Ty}, {});
    ensureExternalDecl(mod, symbol, fnTy);

    rewriter.create<func::CallOp>(
        loc, symbol, TypeRange{}, ValueRange{xPtr, outPtr, Nv});

    auto outTensorTy = RankedTensorType::get({M, K}, xElem);
    Value result =
        rewriter.create<bufferization::ToTensorOp>(loc, outTensorTy, outAlloc);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerGeluToAppleGPUPass
    : public PassWrapper<LowerGeluToAppleGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGeluToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-gelu-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Lower tessera.gelu (rank-2, f32/f16/bf16) to Apple GPU runtime "
           "calls (custom MSL kernel)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerGeluToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerGeluToAppleGPUPass() {
  return std::make_unique<LowerGeluToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
