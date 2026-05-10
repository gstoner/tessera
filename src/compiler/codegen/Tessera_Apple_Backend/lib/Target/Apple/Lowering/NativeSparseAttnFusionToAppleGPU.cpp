//===- NativeSparseAttnFusionToAppleGPU.cpp - Lower NSA fusion ---*- C++ -*-===//
//
// attention_variants_plan, NSA-5 — Apple GPU NSA fused-kernel lowering.
//
// Lowers `tessera.native_sparse_attn_fused` (produced by the NSA fusion
// recognizer) to a runtime call:
//
//   tessera_apple_gpu_native_sparse_attn_f32(
//       Q, K, V, gate_logits, O,
//       B, H, S, D,
//       window_size, block_size, top_k, causal)
//
// The runtime today reproduces the three branches + gating on the host
// (host-reference path); a fully fused MSL kernel that does all three
// branches in a single dispatch with simdgroup reductions for top-k is
// a follow-up that doesn't change this ABI.
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

constexpr llvm::StringLiteral kNSASymbol =
    "tessera_apple_gpu_native_sparse_attn_f32";

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

struct LowerNSAFusionToAppleGPU : public RewritePattern {
  LowerNSAFusionToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.native_sparse_attn_fused", /*benefit=*/3,
                       ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 4)
      return failure();
    Value q = op->getOperand(0);
    Value k = op->getOperand(1);
    Value v = op->getOperand(2);
    Value gate = op->getOperand(3);

    auto qTy = dyn_cast<RankedTensorType>(q.getType());
    auto kTy = dyn_cast<RankedTensorType>(k.getType());
    auto vTy = dyn_cast<RankedTensorType>(v.getType());
    auto gTy = dyn_cast<RankedTensorType>(gate.getType());
    if (!qTy || !kTy || !vTy || !gTy)
      return failure();
    Type elem = qTy.getElementType();
    if (!elem.isF32())
      return rewriter.notifyMatchFailure(
          op, "NSA AppleGPU MSL path is f32 only in v1");
    if (qTy.getRank() != 4 || kTy.getRank() != 4 || vTy.getRank() != 4)
      return rewriter.notifyMatchFailure(
          op, "NSA AppleGPU MSL path: rank-4 (B, H, S, D)");

    if (qTy.isDynamicDim(0) || qTy.isDynamicDim(1) || qTy.isDynamicDim(2) ||
        qTy.isDynamicDim(3))
      return rewriter.notifyMatchFailure(
          op, "NSA AppleGPU MSL path requires static shapes");

    int64_t B = qTy.getDimSize(0);
    int64_t H = qTy.getDimSize(1);
    int64_t S = qTy.getDimSize(2);
    int64_t D = qTy.getDimSize(3);
    int64_t window = 0, block = 0, topK = 0;
    if (auto a = op->getAttrOfType<IntegerAttr>("window_size"))
      window = a.getInt();
    if (auto a = op->getAttrOfType<IntegerAttr>("block_size"))
      block = a.getInt();
    if (auto a = op->getAttrOfType<IntegerAttr>("top_k"))
      topK = a.getInt();
    bool causal = true;
    if (auto a = op->getAttrOfType<BoolAttr>("causal"))
      causal = a.getValue();
    if (window <= 0 || block <= 0 || topK <= 0)
      return rewriter.notifyMatchFailure(
          op, "NSA AppleGPU MSL path: window/block/top_k must be > 0");

    Location loc = op->getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = op->getContext();
    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();

    auto qMemTy = MemRefType::get({B, H, S, D}, elem);
    int64_t numBlocks = S / block;
    auto gateMemTy = MemRefType::get({B, H, S, numBlocks}, elem);
    auto oMemTy = MemRefType::get({B, H, S, D}, elem);

    rewriter.setInsertionPoint(op);
    Value qPtr = extractPtr(rewriter, loc, q, qMemTy);
    Value kPtr = extractPtr(rewriter, loc, k, qMemTy);
    Value vPtr = extractPtr(rewriter, loc, v, qMemTy);
    Value gPtr = extractPtr(rewriter, loc, gate, gateMemTy);
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
    Value Dv = rewriter.create<arith::ConstantIntOp>(loc, D, 32);
    Value Wv = rewriter.create<arith::ConstantIntOp>(loc, window, 32);
    Value Bkv = rewriter.create<arith::ConstantIntOp>(loc, block, 32);
    Value Tkv = rewriter.create<arith::ConstantIntOp>(loc, topK, 32);
    Value Cv = rewriter.create<arith::ConstantIntOp>(loc, causal ? 1 : 0, 32);

    FunctionType fnTy = FunctionType::get(
        ctx, {i64Ty, i64Ty, i64Ty, i64Ty, i64Ty,
              i32Ty, i32Ty, i32Ty, i32Ty,
              i32Ty, i32Ty, i32Ty, i32Ty}, {});
    ensureExternalDecl(mod, kNSASymbol, fnTy);

    rewriter.create<func::CallOp>(
        loc, kNSASymbol, TypeRange{},
        ValueRange{qPtr, kPtr, vPtr, gPtr, oPtr,
                   Bv, Hv, Sv, Dv, Wv, Bkv, Tkv, Cv});

    auto outTensorTy = RankedTensorType::get({B, H, S, D}, elem);
    Value result =
        rewriter.create<bufferization::ToTensorOp>(loc, outTensorTy, oAlloc);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerNSAFusionToAppleGPUPass
    : public PassWrapper<LowerNSAFusionToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerNSAFusionToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-native-sparse-attn-fusion-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Lower tessera.native_sparse_attn_fused (rank-4, f32) to "
           "an Apple GPU runtime call";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerNSAFusionToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerNSAFusionToAppleGPUPass() {
  return std::make_unique<LowerNSAFusionToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
