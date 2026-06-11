//===- SwigluFusionToAppleGPU.cpp -----------------------------*- C++ -*-===//
//
// Phase 8.4.8 — Apple GPU SwiGLU fused MLP-block kernel.
//
// Lowers the `tessera.swiglu_fused` op (produced by Stage 2b's Schedule
// IR fusion recognizer at src/transforms/lib/SwigluFusionPass.cpp) into a
// single Apple GPU runtime call:
//
//   tessera_apple_gpu_swiglu_{f32,f16,bf16}(X, Wg, Wu, Wd, O,
//                                            M, K, H, K_out)
//
// Constraints:
//   - rank-2 inputs/output throughout
//   - matching element types across X, Wg, Wu, Wd, output (f32, f16, bf16)
//   - static shapes
//   - hidden dim H ≤ 256 AND output dim K_out ≤ 256 (per-thread stack
//     arrays in the MSL kernel, mirroring matmul→softmax→matmul / matmul
//     →gelu)
//
// The pass runs first in the Apple GPU pipeline (longest fusion wins, per
// the ordering rule in `docs/apple_gpu_overview.md`). If any constraint
// fails, the fused op falls through to the next pass; if no Apple GPU
// pattern claims it, it stays as `tessera.swiglu_fused` and the runtime
// path falls back to per-op execution.
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

constexpr llvm::StringLiteral kSwigluF32Symbol =
    "tessera_apple_gpu_swiglu_f32";
constexpr llvm::StringLiteral kSwigluF16Symbol =
    "tessera_apple_gpu_swiglu_f16";
constexpr llvm::StringLiteral kSwigluBF16Symbol =
    "tessera_apple_gpu_swiglu_bf16";



struct LowerSwigluFusedToAppleGPU : public RewritePattern {
  LowerSwigluFusedToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.swiglu_fused", /*benefit=*/3, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 4)
      return rewriter.notifyMatchFailure(op,
                                         "swiglu_fused: expected 4 operands");
    Value x = op->getOperand(0);
    Value wGate = op->getOperand(1);
    Value wUp = op->getOperand(2);
    Value wDown = op->getOperand(3);

    auto xTy = dyn_cast<RankedTensorType>(x.getType());
    auto wgTy = dyn_cast<RankedTensorType>(wGate.getType());
    auto wuTy = dyn_cast<RankedTensorType>(wUp.getType());
    auto wdTy = dyn_cast<RankedTensorType>(wDown.getType());
    if (!xTy || !wgTy || !wuTy || !wdTy)
      return rewriter.notifyMatchFailure(op,
                                         "swiglu_fused: non-ranked tensor");
    if (xTy.getRank() != 2 || wgTy.getRank() != 2 || wuTy.getRank() != 2 ||
        wdTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "swiglu_fused: rank-2 only");

    Type elem = xTy.getElementType();
    if (elem != wgTy.getElementType() || elem != wuTy.getElementType() ||
        elem != wdTy.getElementType())
      return rewriter.notifyMatchFailure(
          op, "swiglu_fused: all four operands must share element type");

    StringRef symbol;
    if (elem.isF32()) {
      symbol = kSwigluF32Symbol;
    } else if (elem.isF16()) {
      symbol = kSwigluF16Symbol;
    } else if (elem.isBF16()) {
      symbol = kSwigluBF16Symbol;
    } else {
      return rewriter.notifyMatchFailure(
          op, "swiglu_fused: only f32/f16/bf16 supported in Phase 8.4.8");
    }

    if (xTy.isDynamicDim(0) || xTy.isDynamicDim(1) ||
        wgTy.isDynamicDim(0) || wgTy.isDynamicDim(1) ||
        wuTy.isDynamicDim(0) || wuTy.isDynamicDim(1) ||
        wdTy.isDynamicDim(0) || wdTy.isDynamicDim(1))
      return rewriter.notifyMatchFailure(op,
                                         "swiglu_fused: requires static shapes");

    int64_t M = xTy.getDimSize(0);
    int64_t K = xTy.getDimSize(1);
    int64_t H = wgTy.getDimSize(1);
    int64_t Kout = wdTy.getDimSize(1);

    if (wgTy.getDimSize(0) != K || wuTy.getDimSize(0) != K)
      return rewriter.notifyMatchFailure(
          op, "swiglu_fused: gate/up matmul K dim must match X");
    if (wuTy.getDimSize(1) != H)
      return rewriter.notifyMatchFailure(
          op, "swiglu_fused: gate and up must produce matching hidden dim");
    if (wdTy.getDimSize(0) != H)
      return rewriter.notifyMatchFailure(
          op, "swiglu_fused: down matmul K dim must match hidden dim");

    // The MSL kernel allocates per-thread stack arrays of size H and Kout.
    // Cap these to mirror the existing MLP-block fusions; larger shapes
    // fall through to the per-op path.
    if (H > 256 || Kout > 256)
      return rewriter.notifyMatchFailure(
          op, "swiglu_fused: GPU kernel limited to H <= 256 and Kout <= 256");

    Location loc = op->getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = op->getContext();

    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();

    auto xMemTy = MemRefType::get({M, K}, elem);
    auto wgMemTy = MemRefType::get({K, H}, elem);
    auto wuMemTy = MemRefType::get({K, H}, elem);
    auto wdMemTy = MemRefType::get({H, Kout}, elem);
    auto oMemTy = MemRefType::get({M, Kout}, elem);

    rewriter.setInsertionPoint(op);
    Value xPtr = extractPtr(rewriter, loc, x, xMemTy);
    Value wgPtr = extractPtr(rewriter, loc, wGate, wgMemTy);
    Value wuPtr = extractPtr(rewriter, loc, wUp, wuMemTy);
    Value wdPtr = extractPtr(rewriter, loc, wDown, wdMemTy);
    auto oAlloc = rewriter.create<memref::AllocOp>(loc, oMemTy);
    Value oPtr;
    {
      auto pi =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, oAlloc);
      oPtr = rewriter.create<arith::IndexCastOp>(loc, i64Ty, pi);
    }

    Value Mv = rewriter.create<arith::ConstantIntOp>(loc, M, 32);
    Value Kv = rewriter.create<arith::ConstantIntOp>(loc, K, 32);
    Value Hv = rewriter.create<arith::ConstantIntOp>(loc, H, 32);
    Value Kov = rewriter.create<arith::ConstantIntOp>(loc, Kout, 32);

    FunctionType fnTy = FunctionType::get(
        ctx, {i64Ty, i64Ty, i64Ty, i64Ty, i64Ty, i32Ty, i32Ty, i32Ty, i32Ty},
        {});
    ensureExternalDecl(mod, symbol, fnTy);

    auto callOp = rewriter.create<func::CallOp>(
        loc, symbol, TypeRange{},
        ValueRange{xPtr, wgPtr, wuPtr, wdPtr, oPtr, Mv, Kv, Hv, Kov});
    // Decision #19 — emit the fusion descriptor. swiglu lowers a pre-fused
    // tessera.swiglu_fused op, so the op itself is the descriptor (no chain
    // re-discovery): source = "composite_op".
    callOp->setAttr("tessera.fusion.kernel", rewriter.getStringAttr("swiglu"));
    callOp->setAttr("tessera.fusion.source", rewriter.getStringAttr("composite_op"));

    auto outTensorTy = RankedTensorType::get({M, Kout}, elem);
    Value result =
        rewriter.create<bufferization::ToTensorOp>(loc, outTensorTy, oAlloc);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerSwigluFusionToAppleGPUPass
    : public PassWrapper<LowerSwigluFusionToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerSwigluFusionToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-swiglu-fusion-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Lower tessera.swiglu_fused (rank-2, f32/f16/bf16, H <= 256, "
           "Kout <= 256) to an Apple GPU runtime call (custom MSL kernel)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerSwigluFusedToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerSwigluFusionToAppleGPUPass() {
  return std::make_unique<LowerSwigluFusionToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
