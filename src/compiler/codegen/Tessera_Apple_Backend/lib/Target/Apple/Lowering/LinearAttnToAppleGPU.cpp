//===- LinearAttnToAppleGPU.cpp - Lower tessera.linear_attn to MSL ------===//
//
// attention_variants_plan, LA-2 — Apple GPU custom MSL linear-attention
// forward (causal recurrent form).
//
// Replaces tessera.linear_attn ops (rank-4 f32, D_qk * D_v <= 256) with
// calls to the Apple GPU runtime shim:
//
//   tessera_apple_gpu_linear_attn_f32(
//       Q, K, V, O,           // i64 raw pointers (row-major, fp32)
//       B, H, S, D_qk, D_v,   // i32 dims
//       feature_map,          // i32 enum: 0=elu, 1=relu, 2=identity, 3=poly2
//       causal)               // i32 bool (0 / 1)
//
// Decay + custom state inputs are only exercised on the host reference
// path today; the MSL kernel ships causal/no-decay/feature_map ∈ {elu,
// relu, identity, polynomial_2}. Out-of-envelope inputs fall through to
// the host reference so the tessera.linear_attn op stays in IR for
// later passes.
//
// Shape contract:
//   Q: (B, H, S, D_qk)
//   K: (B, H, S, D_qk)
//   V: (B, H, S, D_v)
//   O: (B, H, S, D_v)
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
#include "llvm/ADT/StringRef.h"

using namespace ::mlir;

namespace tessera {
namespace apple {

namespace {

constexpr llvm::StringLiteral kLinearAttnF32Symbol =
    "tessera_apple_gpu_linear_attn_f32";

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

// Map feature_map StringAttr to an i32 enum the runtime understands.
//   0=elu, 1=relu, 2=identity, 3=polynomial_2
static int featureMapToInt(StringRef name) {
  if (name == "elu") return 0;
  if (name == "relu") return 1;
  if (name == "identity") return 2;
  if (name == "polynomial_2") return 3;
  return 0;  // sensible default — runtime still validates
}

struct LowerLinearAttnToAppleGPU : public RewritePattern {
  LowerLinearAttnToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.linear_attn", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 3)
      return failure();
    Value q = op->getOperand(0);
    Value k = op->getOperand(1);
    Value v = op->getOperand(2);

    auto qTy = dyn_cast<RankedTensorType>(q.getType());
    auto kTy = dyn_cast<RankedTensorType>(k.getType());
    auto vTy = dyn_cast<RankedTensorType>(v.getType());
    if (!qTy || !kTy || !vTy)
      return failure();
    if (qTy.getRank() != 4 || kTy.getRank() != 4 || vTy.getRank() != 4)
      return rewriter.notifyMatchFailure(
          op, "linear_attn AppleGPU MSL path is rank-4 only");

    Type elem = qTy.getElementType();
    if (!elem.isF32())
      return rewriter.notifyMatchFailure(
          op, "linear_attn AppleGPU MSL path is f32 only in v1");
    if (kTy.getElementType() != elem || vTy.getElementType() != elem)
      return rewriter.notifyMatchFailure(
          op, "linear_attn AppleGPU MSL path requires matching Q/K/V dtypes");

    if (qTy.isDynamicDim(0) || qTy.isDynamicDim(1) || qTy.isDynamicDim(2) ||
        qTy.isDynamicDim(3) || kTy.isDynamicDim(3) || vTy.isDynamicDim(3))
      return rewriter.notifyMatchFailure(
          op, "linear_attn AppleGPU MSL path requires static shapes");

    int64_t B = qTy.getDimSize(0);
    int64_t H = qTy.getDimSize(1);
    int64_t S = qTy.getDimSize(2);
    int64_t D_qk = qTy.getDimSize(3);
    int64_t D_v = vTy.getDimSize(3);
    if (kTy.getDimSize(0) != B || kTy.getDimSize(1) != H ||
        kTy.getDimSize(2) != S || kTy.getDimSize(3) != D_qk ||
        vTy.getDimSize(0) != B || vTy.getDimSize(1) != H ||
        vTy.getDimSize(2) != S)
      return rewriter.notifyMatchFailure(
          op, "linear_attn AppleGPU MSL path: shape mismatch across Q/K/V");

    // Per-thread state cap: one S[D_qk * D_v] buffer per batch-head.
    // Stack-array layout caps at 256 fp32 floats per thread (1 KB).
    if (D_qk * D_v > 256)
      return rewriter.notifyMatchFailure(
          op, "linear_attn AppleGPU MSL path: D_qk * D_v > 256 (per-thread "
              "state cap exceeded)");

    // v1 ships causal-only; non-causal callers stay on the host reference
    // until the MSL kernel grows a separate non-causal entry point.
    bool causal = true;
    if (auto attr = op->getAttrOfType<BoolAttr>("causal"))
      causal = attr.getValue();
    if (!causal)
      return rewriter.notifyMatchFailure(
          op, "linear_attn AppleGPU MSL path: only causal=true in v1");

    // Decay path falls through to the host reference today.
    if (op->getNumOperands() > 3)
      return rewriter.notifyMatchFailure(
          op, "linear_attn AppleGPU MSL path: decay/state inputs not yet wired");

    // feature_map: read from op attrs; default to elu (matches op default).
    StringRef featureMap = "elu";
    if (auto attr = op->getAttrOfType<StringAttr>("feature_map"))
      featureMap = attr.getValue();

    Location loc = op->getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = op->getContext();

    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();

    auto qMemTy = MemRefType::get({B, H, S, D_qk}, elem);
    auto kMemTy = MemRefType::get({B, H, S, D_qk}, elem);
    auto vMemTy = MemRefType::get({B, H, S, D_v}, elem);
    auto oMemTy = MemRefType::get({B, H, S, D_v}, elem);

    rewriter.setInsertionPoint(op);
    Value qPtr = extractPtr(rewriter, loc, q, qMemTy);
    Value kPtr = extractPtr(rewriter, loc, k, kMemTy);
    Value vPtr = extractPtr(rewriter, loc, v, vMemTy);
    auto oAlloc = rewriter.create<memref::AllocOp>(loc, oMemTy);
    Value oPtr;
    {
      auto pi =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, oAlloc);
      oPtr = rewriter.create<arith::IndexCastOp>(loc, i64Ty, pi);
    }

    Value Bv = rewriter.create<arith::ConstantIntOp>(loc, B, 32);
    Value Hv = rewriter.create<arith::ConstantIntOp>(loc, H, 32);
    Value Sv = rewriter.create<arith::ConstantIntOp>(loc, S, 32);
    Value Dqkv = rewriter.create<arith::ConstantIntOp>(loc, D_qk, 32);
    Value Dvv = rewriter.create<arith::ConstantIntOp>(loc, D_v, 32);
    Value FMv = rewriter.create<arith::ConstantIntOp>(
        loc, featureMapToInt(featureMap), 32);
    Value CausalV = rewriter.create<arith::ConstantIntOp>(loc, causal ? 1 : 0, 32);

    FunctionType fnTy = FunctionType::get(
        ctx, {i64Ty, i64Ty, i64Ty, i64Ty,
              i32Ty, i32Ty, i32Ty, i32Ty, i32Ty,
              i32Ty, i32Ty}, {});
    ensureExternalDecl(mod, kLinearAttnF32Symbol, fnTy);

    rewriter.create<func::CallOp>(
        loc, kLinearAttnF32Symbol, TypeRange{},
        ValueRange{qPtr, kPtr, vPtr, oPtr,
                   Bv, Hv, Sv, Dqkv, Dvv, FMv, CausalV});

    auto outTensorTy = RankedTensorType::get({B, H, S, D_v}, elem);
    Value result =
        rewriter.create<bufferization::ToTensorOp>(loc, outTensorTy, oAlloc);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerLinearAttnToAppleGPUPass
    : public PassWrapper<LowerLinearAttnToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinearAttnToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-linear-attn-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Lower tessera.linear_attn (rank-4, f32, D_qk*D_v <= 256, causal) "
           "to an Apple GPU runtime call (custom MSL kernel)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerLinearAttnToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerLinearAttnToAppleGPUPass() {
  return std::make_unique<LowerLinearAttnToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
