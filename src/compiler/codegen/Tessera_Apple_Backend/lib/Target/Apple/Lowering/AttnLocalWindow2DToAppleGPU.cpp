//===- AttnLocalWindow2DToAppleGPU.cpp — Sub-2 (2026-05-20) ---------------===//
//
// Lower ``tessera.attn_local_window_2d`` (rank-5 f32, window=[rh, rw])
// to a func.call into the Apple GPU runtime shim:
//
//   tessera_apple_gpu_attn_local_window_2d_f32(
//       Q, K, V, O,                 // 4 × i64 raw pointers (fp32 row-major)
//       B, H, Hq, Wq, D,            // 5 × i32 dims
//       rh, rw)                     // 2 × i32 half-window
//
// The runtime symbol is *declared* by this pass and registered as a
// planned kernel in the backend manifest.  The MSL kernel that backs
// the symbol is a separate deliverable (per Sub-2 scope decision — the
// IR-side lowering is what gates "single-device tiled lowering";
// physical kernel codegen is staged alongside the existing Apple GPU
// kernel inventory).
//
// Single-device only.  Distributed callers must go through the
// halo-aware path: HaloMeshIntegrationPass inserts halo.exchange on
// sharded inputs before this lowering runs, and the IR contract is
// independent of the actual transport.
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

constexpr llvm::StringLiteral kAttnLocalWindow2DSymbol =
    "tessera_apple_gpu_attn_local_window_2d_f32";



struct LowerAttnLocalWindow2DToAppleGPU : public RewritePattern {
  LowerAttnLocalWindow2DToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.attn_local_window_2d",
                       /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 3) return failure();
    Value q = op->getOperand(0);
    Value k = op->getOperand(1);
    Value v = op->getOperand(2);

    auto qTy = dyn_cast<RankedTensorType>(q.getType());
    auto kTy = dyn_cast<RankedTensorType>(k.getType());
    auto vTy = dyn_cast<RankedTensorType>(v.getType());
    if (!qTy || !kTy || !vTy) return failure();
    if (qTy.getRank() != 5 || kTy.getRank() != 5 || vTy.getRank() != 5)
      return rewriter.notifyMatchFailure(
          op, "attn_local_window_2d AppleGPU path requires rank-5 (B, H, Hq, Wq, D)");

    Type elem = qTy.getElementType();
    if (!elem.isF32())
      return rewriter.notifyMatchFailure(
          op, "attn_local_window_2d AppleGPU path is f32 only in v1");
    if (kTy.getElementType() != elem || vTy.getElementType() != elem)
      return rewriter.notifyMatchFailure(
          op, "attn_local_window_2d AppleGPU path requires matching dtypes");

    // All five dims must be static so the runtime can route arguments.
    for (int axis = 0; axis < 5; ++axis) {
      if (qTy.isDynamicDim(axis) || kTy.isDynamicDim(axis) ||
          vTy.isDynamicDim(axis))
        return rewriter.notifyMatchFailure(
            op, "attn_local_window_2d AppleGPU path requires static shapes");
    }
    int64_t B  = qTy.getDimSize(0);
    int64_t H  = qTy.getDimSize(1);
    int64_t Hq = qTy.getDimSize(2);
    int64_t Wq = qTy.getDimSize(3);
    int64_t D  = qTy.getDimSize(4);
    // K, V must share the (B, H, Hq, Wq, D) layout for the v1 path.
    if (kTy.getDimSize(0) != B || kTy.getDimSize(1) != H ||
        kTy.getDimSize(2) != Hq || kTy.getDimSize(3) != Wq ||
        kTy.getDimSize(4) != D ||
        vTy.getDimSize(0) != B || vTy.getDimSize(1) != H ||
        vTy.getDimSize(2) != Hq || vTy.getDimSize(3) != Wq ||
        vTy.getDimSize(4) != D)
      return rewriter.notifyMatchFailure(
          op, "attn_local_window_2d AppleGPU path: shape mismatch across Q/K/V");

    // Read window=[rh, rw] from the op attribute.
    auto window = op->getAttrOfType<ArrayAttr>("window");
    if (!window || window.size() != 2)
      return rewriter.notifyMatchFailure(
          op, "attn_local_window_2d AppleGPU path: missing/invalid window attr");
    auto rhAttr = llvm::dyn_cast<IntegerAttr>(window[0]);
    auto rwAttr = llvm::dyn_cast<IntegerAttr>(window[1]);
    if (!rhAttr || !rwAttr)
      return rewriter.notifyMatchFailure(
          op, "attn_local_window_2d AppleGPU path: window entries must be integers");
    int64_t rh = rhAttr.getInt();
    int64_t rw = rwAttr.getInt();
    if (rh < 0 || rw < 0)
      return rewriter.notifyMatchFailure(
          op, "attn_local_window_2d AppleGPU path: window half-widths must be >= 0");
    // Patch size cap: ((2*rh+1) * (2*rw+1)) * D ≤ 1024 fp32 floats per thread.
    // Matches the stack-array convention used by every other Apple GPU
    // fused-attention kernel.
    int64_t patchSize = (2 * rh + 1) * (2 * rw + 1);
    if (patchSize * D > 1024)
      return rewriter.notifyMatchFailure(
          op, "attn_local_window_2d AppleGPU path: patch*D > 1024 (per-thread cap)");

    Location loc = op->getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = op->getContext();

    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();

    auto memTy = MemRefType::get({B, H, Hq, Wq, D}, elem);

    rewriter.setInsertionPoint(op);
    Value qPtr = extractPtr(rewriter, loc, q, memTy);
    Value kPtr = extractPtr(rewriter, loc, k, memTy);
    Value vPtr = extractPtr(rewriter, loc, v, memTy);

    auto oAlloc = rewriter.create<memref::AllocOp>(loc, memTy);
    Value oPtr;
    {
      auto pi = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
          loc, oAlloc);
      oPtr = rewriter.create<arith::IndexCastOp>(loc, i64Ty, pi);
    }

    Value Bv  = rewriter.create<arith::ConstantIntOp>(loc, B,  32);
    Value Hv  = rewriter.create<arith::ConstantIntOp>(loc, H,  32);
    Value Hqv = rewriter.create<arith::ConstantIntOp>(loc, Hq, 32);
    Value Wqv = rewriter.create<arith::ConstantIntOp>(loc, Wq, 32);
    Value Dv  = rewriter.create<arith::ConstantIntOp>(loc, D,  32);
    Value Rhv = rewriter.create<arith::ConstantIntOp>(loc, rh, 32);
    Value Rwv = rewriter.create<arith::ConstantIntOp>(loc, rw, 32);

    FunctionType fnTy = FunctionType::get(
        ctx, {i64Ty, i64Ty, i64Ty, i64Ty,
              i32Ty, i32Ty, i32Ty, i32Ty, i32Ty,
              i32Ty, i32Ty}, {});
    ensureExternalDecl(mod, kAttnLocalWindow2DSymbol, fnTy);

    rewriter.create<func::CallOp>(
        loc, kAttnLocalWindow2DSymbol, TypeRange{},
        ValueRange{qPtr, kPtr, vPtr, oPtr,
                   Bv, Hv, Hqv, Wqv, Dv,
                   Rhv, Rwv});

    auto outTensorTy = RankedTensorType::get({B, H, Hq, Wq, D}, elem);
    Value result =
        rewriter.create<bufferization::ToTensorOp>(loc, outTensorTy, oAlloc);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerAttnLocalWindow2DToAppleGPUPass
    : public PassWrapper<LowerAttnLocalWindow2DToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAttnLocalWindow2DToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-attn-local-window-2d-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Lower tessera.attn_local_window_2d (rank-5, f32, "
           "(2*rh+1)*(2*rw+1)*D ≤ 1024) to an Apple GPU runtime call";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerAttnLocalWindow2DToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerAttnLocalWindow2DToAppleGPUPass() {
  return std::make_unique<LowerAttnLocalWindow2DToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
