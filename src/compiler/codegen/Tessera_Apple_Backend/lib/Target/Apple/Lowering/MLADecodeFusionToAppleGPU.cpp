//===- MLADecodeFusionToAppleGPU.cpp - MLA decode fusion → Apple GPU -----===//
//
// attention_variants_plan, MLA-2.
//
// Lowers the `tessera.mla_decode_fused` op (produced by Stage 2b's
// Schedule IR fusion recognizer at `src/transforms/lib/MLAFusionPass.cpp`)
// into a single Apple GPU runtime call:
//
//   tessera_apple_gpu_mla_decode_f32(x, W_dkv, W_uk, W_uv, Q, O,
//                                     B, S_kv, D_x, D_lat, S_q, D_h)
//
// Constraints:
//   * rank-3 inputs throughout (x is (S_kv, D_x); Q is (B, S_q, D_h); ...)
//   * matching f32 element type
//   * static shapes
//
// The runtime shim today materializes the latent C, expands K/V via the
// two cached weights, and runs the host reference flash-attn — the
// memory win (latent-only cache) is observable but the absorb-K speed
// win waits on the Hopper kernel (Phase G). This pass is the API +
// IR-level unblock so the fusion is visible to the Apple GPU pipeline.
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
#include "llvm/ADT/StringRef.h"

using namespace ::mlir;

namespace tessera {
namespace apple {

namespace {

constexpr llvm::StringLiteral kMLADecodeF32Symbol =
    "tessera_apple_gpu_mla_decode_f32";



struct LowerMLADecodeFusionToAppleGPU : public RewritePattern {
  LowerMLADecodeFusionToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.mla_decode_fused", /*benefit=*/3, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 5)
      return failure();
    Value x = op->getOperand(0);
    Value wDkv = op->getOperand(1);
    Value wUk = op->getOperand(2);
    Value wUv = op->getOperand(3);
    Value q = op->getOperand(4);

    auto xTy = dyn_cast<RankedTensorType>(x.getType());
    auto wDkvTy = dyn_cast<RankedTensorType>(wDkv.getType());
    auto wUkTy = dyn_cast<RankedTensorType>(wUk.getType());
    auto wUvTy = dyn_cast<RankedTensorType>(wUv.getType());
    auto qTy = dyn_cast<RankedTensorType>(q.getType());
    if (!xTy || !wDkvTy || !wUkTy || !wUvTy || !qTy)
      return failure();

    Type elem = xTy.getElementType();
    if (!elem.isF32())
      return rewriter.notifyMatchFailure(
          op, "MLA decode AppleGPU MSL path is f32 only in v1");
    if (wDkvTy.getElementType() != elem || wUkTy.getElementType() != elem ||
        wUvTy.getElementType() != elem || qTy.getElementType() != elem)
      return rewriter.notifyMatchFailure(
          op, "MLA decode AppleGPU MSL path requires matching dtypes");

    if (xTy.getRank() != 2 || wDkvTy.getRank() != 2 || wUkTy.getRank() != 2 ||
        wUvTy.getRank() != 2 || qTy.getRank() != 3)
      return rewriter.notifyMatchFailure(
          op, "MLA decode AppleGPU MSL path: rank contract — x/W*: rank-2, "
              "Q: rank-3");

    if (xTy.isDynamicDim(0) || xTy.isDynamicDim(1) ||
        wDkvTy.isDynamicDim(0) || wDkvTy.isDynamicDim(1) ||
        wUkTy.isDynamicDim(0) || wUkTy.isDynamicDim(1) ||
        wUvTy.isDynamicDim(0) || wUvTy.isDynamicDim(1) ||
        qTy.isDynamicDim(0) || qTy.isDynamicDim(1) || qTy.isDynamicDim(2))
      return rewriter.notifyMatchFailure(
          op, "MLA decode AppleGPU MSL path requires static shapes");

    int64_t S_kv = xTy.getDimSize(0);
    int64_t D_x = xTy.getDimSize(1);
    int64_t D_lat = wDkvTy.getDimSize(1);
    int64_t D_h = wUkTy.getDimSize(1);
    int64_t B = qTy.getDimSize(0);
    int64_t S_q = qTy.getDimSize(1);

    if (wDkvTy.getDimSize(0) != D_x ||
        wUkTy.getDimSize(0) != D_lat ||
        wUvTy.getDimSize(0) != D_lat ||
        wUvTy.getDimSize(1) != D_h ||
        qTy.getDimSize(2) != D_h)
      return rewriter.notifyMatchFailure(
          op, "MLA decode AppleGPU MSL path: weight / Q shape mismatch");

    Location loc = op->getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = op->getContext();

    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();

    auto xMemTy = MemRefType::get({S_kv, D_x}, elem);
    auto wDkvMemTy = MemRefType::get({D_x, D_lat}, elem);
    auto wUkMemTy = MemRefType::get({D_lat, D_h}, elem);
    auto wUvMemTy = MemRefType::get({D_lat, D_h}, elem);
    auto qMemTy = MemRefType::get({B, S_q, D_h}, elem);
    auto oMemTy = MemRefType::get({B, S_q, D_h}, elem);

    rewriter.setInsertionPoint(op);
    Value xPtr = extractPtr(rewriter, loc, x, xMemTy);
    Value wDkvPtr = extractPtr(rewriter, loc, wDkv, wDkvMemTy);
    Value wUkPtr = extractPtr(rewriter, loc, wUk, wUkMemTy);
    Value wUvPtr = extractPtr(rewriter, loc, wUv, wUvMemTy);
    Value qPtr = extractPtr(rewriter, loc, q, qMemTy);
    auto oAlloc = rewriter.create<memref::AllocOp>(loc, oMemTy);
    Value oPtr;
    {
      auto pi =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, oAlloc);
      oPtr = rewriter.create<arith::IndexCastOp>(loc, i64Ty, pi);
    }

    Value Bv = rewriter.create<arith::ConstantIntOp>(loc, B, 32);
    Value Skv = rewriter.create<arith::ConstantIntOp>(loc, S_kv, 32);
    Value Dx = rewriter.create<arith::ConstantIntOp>(loc, D_x, 32);
    Value Dlat = rewriter.create<arith::ConstantIntOp>(loc, D_lat, 32);
    Value Sq = rewriter.create<arith::ConstantIntOp>(loc, S_q, 32);
    Value Dh = rewriter.create<arith::ConstantIntOp>(loc, D_h, 32);

    FunctionType fnTy = FunctionType::get(
        ctx, {i64Ty, i64Ty, i64Ty, i64Ty, i64Ty, i64Ty,
              i32Ty, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty}, {});
    ensureExternalDecl(mod, kMLADecodeF32Symbol, fnTy);

    auto callOp = rewriter.create<func::CallOp>(
        loc, kMLADecodeF32Symbol, TypeRange{},
        ValueRange{xPtr, wDkvPtr, wUkPtr, wUvPtr, qPtr, oPtr,
                   Bv, Skv, Dx, Dlat, Sq, Dh});
    // Decision #19 — emit the fusion descriptor. MLA decode lowers a pre-fused
    // tessera.mla_decode_fused op (the op is the descriptor): source="composite_op".
    callOp->setAttr("tessera.fusion.kernel", rewriter.getStringAttr("mla_decode"));
    callOp->setAttr("tessera.fusion.source", rewriter.getStringAttr("composite_op"));

    auto outTensorTy = RankedTensorType::get({B, S_q, D_h}, elem);
    Value result =
        rewriter.create<bufferization::ToTensorOp>(loc, outTensorTy, oAlloc);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerMLADecodeFusionToAppleGPUPass
    : public PassWrapper<LowerMLADecodeFusionToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LowerMLADecodeFusionToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-mla-decode-fusion-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Lower tessera.mla_decode_fused (rank-3 Q + rank-2 weights, f32) "
           "to an Apple GPU runtime call (host-reference today; absorb-K MSL "
           "kernel is a follow-up)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerMLADecodeFusionToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerMLADecodeFusionToAppleGPUPass() {
  return std::make_unique<LowerMLADecodeFusionToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
