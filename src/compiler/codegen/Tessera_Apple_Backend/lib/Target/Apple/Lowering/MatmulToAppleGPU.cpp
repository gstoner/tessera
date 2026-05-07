//===- MatmulToAppleGPU.cpp - Lower tessera.matmul to MPS runtime --------===//
//
// Phase 8.3 — Apple GPU native execution via Metal Performance Shaders.
//
// Replaces tessera.matmul ops (with static f32 input tensors) with calls to
// the Apple-GPU runtime shim:
//
//   tessera_apple_gpu_mps_matmul_f32  (A: f32, B: f32 -> C: f32, row-major)
//
// Mirrors MatmulToAppleCPU.cpp exactly — the only difference is the call
// target. The runtime shim (apple_gpu_runtime.mm) builds a MetalDeviceContext,
// allocates MTLBuffers in shared storage, encodes an MPSMatrixMultiplication,
// commits + waits, and copies the result back to host memory.
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

constexpr llvm::StringLiteral kGemmF32Symbol =
    "tessera_apple_gpu_mps_matmul_f32";

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

struct LowerMatmulToAppleGPU : public RewritePattern {
  LowerMatmulToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.matmul", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2)
      return failure();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);

    auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsTy || !rhsTy || lhsTy.getRank() != 2 || rhsTy.getRank() != 2)
      return failure();

    Type lhsElem = lhsTy.getElementType();
    Type rhsElem = rhsTy.getElementType();
    if (!lhsElem.isF32() || !rhsElem.isF32())
      return rewriter.notifyMatchFailure(
          op, "AppleGPU MPS path is f32-only in Phase 8.3");

    if (lhsTy.isDynamicDim(0) || lhsTy.isDynamicDim(1) ||
        rhsTy.isDynamicDim(0) || rhsTy.isDynamicDim(1))
      return rewriter.notifyMatchFailure(
          op, "AppleGPU MPS path requires static shapes");

    int64_t M = lhsTy.getDimSize(0);
    int64_t K = lhsTy.getDimSize(1);
    int64_t N = rhsTy.getDimSize(1);
    if (rhsTy.getDimSize(0) != K)
      return rewriter.notifyMatchFailure(op, "matmul shape mismatch");

    Location loc = op->getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = op->getContext();

    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();
    Type f32Ty = rewriter.getF32Type();

    auto lhsMemTy = MemRefType::get({M, K}, f32Ty);
    auto rhsMemTy = MemRefType::get({K, N}, f32Ty);
    auto outMemTy = MemRefType::get({M, N}, f32Ty);

    Value aPtr = extractPtr(rewriter, loc, lhs, lhsMemTy);
    Value bPtr = extractPtr(rewriter, loc, rhs, rhsMemTy);
    auto cAlloc = rewriter.create<memref::AllocOp>(loc, outMemTy);
    Value cPtr;
    {
      auto pi =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, cAlloc);
      cPtr = rewriter.create<arith::IndexCastOp>(loc, i64Ty, pi);
    }

    Value Mv = rewriter.create<arith::ConstantIntOp>(loc, M, 32);
    Value Nv = rewriter.create<arith::ConstantIntOp>(loc, N, 32);
    Value Kv = rewriter.create<arith::ConstantIntOp>(loc, K, 32);

    FunctionType gemmFnTy = FunctionType::get(
        ctx, {i64Ty, i64Ty, i64Ty, i32Ty, i32Ty, i32Ty}, {});
    ensureExternalDecl(mod, kGemmF32Symbol, gemmFnTy);

    rewriter.create<func::CallOp>(
        loc, kGemmF32Symbol, TypeRange{},
        ValueRange{aPtr, bPtr, cPtr, Mv, Nv, Kv});

    auto outTensorTy = RankedTensorType::get({M, N}, f32Ty);
    Value result =
        rewriter.create<bufferization::ToTensorOp>(loc, outTensorTy, cAlloc);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerMatmulToAppleGPUPass
    : public PassWrapper<LowerMatmulToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMatmulToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-matmul-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Lower tessera.matmul (rank-2, f32) to Apple GPU runtime calls "
           "(MPSMatrixMultiplication)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerMatmulToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerMatmulToAppleGPUPass() {
  return std::make_unique<LowerMatmulToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
