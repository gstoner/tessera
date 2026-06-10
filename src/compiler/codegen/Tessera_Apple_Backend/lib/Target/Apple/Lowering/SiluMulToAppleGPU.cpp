//===- SiluMulToAppleGPU.cpp - Lower tessera.silu_mul to MPSGraph lane ---===//
//
// 2026-05-29 — Apple GPU MPSGraph lane.
//
// Lowers the SwiGLU gate `tessera.silu_mul(a, b) = silu(a) * b` to the
// op-coded binary runtime call:
//
//   tessera_apple_gpu_mpsgraph_binary_f32(int32 op, first, second, out, n)
//
// The runtime opcode 6 computes `first * silu(second)`, so we pass
// (first=b, second=a) to get `silu(a) * b`. Elementwise => rank-agnostic.
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



struct LowerSiluMulToAppleGPU : public RewritePattern {
  LowerSiluMulToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.silu_mul", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2 || op->getNumResults() != 1)
      return failure();
    Value a = op->getOperand(0);
    Value b = op->getOperand(1);
    auto aTy = dyn_cast<RankedTensorType>(a.getType());
    auto bTy = dyn_cast<RankedTensorType>(b.getType());
    if (!aTy || !bTy || !aTy.hasStaticShape() || aTy != bTy)
      return rewriter.notifyMatchFailure(op, "silu_mul needs matching static-shape operands");
    Type elem = aTy.getElementType();
    if (!elem.isF32())
      return rewriter.notifyMatchFailure(op, "AppleGPU silu_mul lane is f32 (f16/bf16 dispatch at runtime)");

    int64_t n = 1;
    for (int64_t d : aTy.getShape())
      n *= d;

    Location loc = op->getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = op->getContext();
    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();
    auto memTy = MemRefType::get(aTy.getShape(), elem);

    // opcode 6 = first * silu(second); pass (first=b, second=a).
    Value firstPtr = extractPtr(rewriter, loc, b, memTy);
    Value secondPtr = extractPtr(rewriter, loc, a, memTy);
    auto outAlloc = rewriter.create<memref::AllocOp>(loc, memTy);
    auto outIdx =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outAlloc);
    Value outPtr = rewriter.create<arith::IndexCastOp>(loc, i64Ty, outIdx);
    Value opc = rewriter.create<arith::ConstantIntOp>(loc, 6, 32);
    Value nv = rewriter.create<arith::ConstantIntOp>(loc, n, 64);

    FunctionType fnTy =
        FunctionType::get(ctx, {i32Ty, i64Ty, i64Ty, i64Ty, i64Ty}, {});
    ensureExternalDecl(mod, "tessera_apple_gpu_mpsgraph_binary_f32", fnTy);
    rewriter.create<func::CallOp>(
        loc, StringRef("tessera_apple_gpu_mpsgraph_binary_f32"), TypeRange{},
        ValueRange{opc, firstPtr, secondPtr, outPtr, nv});

    Value result = rewriter.create<bufferization::ToTensorOp>(
        loc, RankedTensorType::get(aTy.getShape(), elem), outAlloc);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerSiluMulToAppleGPUPass
    : public PassWrapper<LowerSiluMulToAppleGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerSiluMulToAppleGPUPass)

  StringRef getArgument() const override { return "tessera-silu-mul-to-apple_gpu"; }
  StringRef getDescription() const override {
    return "Lower tessera.silu_mul (SwiGLU gate) to Apple GPU MPSGraph "
           "binary runtime call";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerSiluMulToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerSiluMulToAppleGPUPass() {
  return std::make_unique<LowerSiluMulToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
