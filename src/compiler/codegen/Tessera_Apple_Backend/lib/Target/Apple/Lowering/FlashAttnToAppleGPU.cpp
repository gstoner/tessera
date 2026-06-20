//===- FlashAttnToAppleGPU.cpp - Lower tessera.flash_attn to MSL kernel --===//
//
// Phase 8.4.1 — Apple GPU custom MSL flash-attention forward.
//
// Replaces tessera.flash_attn ops (rank-3, f32, head_dim <= 256) with calls
// to the Apple-GPU runtime shim:
//
//   tessera_apple_gpu_flash_attn_f32(
//       Q, K, V, O,                  // i64 raw pointers (row-major)
//       B, Sq, Sk, D,                // i32 dims
//       scale,                       // f32
//       causal)                      // i32 bool (0 / 1)
//
// Same plumbing shape as MatmulToAppleGPU and RopeToAppleGPU — bufferize
// each tensor operand, extract aligned pointers as i64, allocate the output
// memref on the host, and emit the func.call. The runtime compiles the
// embedded MSL flash-attention source on first call and caches the resulting
// MTLComputePipelineState by (msl_source, entry_point).
//
// Shape contract:
//   Q: (B, Sq, D)
//   K: (B, Sk, D)
//   V: (B, Sk, D)
//   O: (B, Sq, D)
//
// `scale` defaults to 1.0 / sqrt(D) when the op carries no `scale` attribute,
// matching the numpy reference. `causal` defaults to false.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/LoweringUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cmath>

using namespace ::mlir;

namespace tessera {
namespace apple {

namespace {

constexpr llvm::StringLiteral kFlashAttnF32Symbol =
    "tessera_apple_gpu_flash_attn_f32";
constexpr llvm::StringLiteral kFlashAttnF16Symbol =
    "tessera_apple_gpu_flash_attn_f16";
constexpr llvm::StringLiteral kFlashAttnBF16Symbol =
    "tessera_apple_gpu_flash_attn_bf16";

// Bias-aware MPSGraph attention path: softmax(scale*Q*K^T + bias)*V.
// Same dims as the bias-free symbols plus one extra bias pointer; the
// runtime transposes K internally and adds the (B, Sq, Sk) bias pre-softmax.
constexpr llvm::StringLiteral kFlashAttnBiasF32Symbol =
    "tessera_apple_gpu_flash_attn_bias_f32";
constexpr llvm::StringLiteral kFlashAttnBiasF16Symbol =
    "tessera_apple_gpu_flash_attn_bias_f16";
constexpr llvm::StringLiteral kFlashAttnBiasBF16Symbol =
    "tessera_apple_gpu_flash_attn_bias_bf16";



struct LowerFlashAttnToAppleGPU : public RewritePattern {
  LowerFlashAttnToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.flash_attn", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 3)
      return failure();
    Value q = op->getOperand(0);
    Value k = op->getOperand(1);
    Value v = op->getOperand(2);

    // attn_bias: with AttrSizedOperandSegments the segment layout is
    // [q=1, k_or_cache=1, v=N, attn_bias=0|1]. When the trailing segment is 1
    // the bias is the last operand and we route to the bias-aware MPSGraph path.
    Value bias;
    if (auto seg = op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes")) {
      auto arr = seg.asArrayRef();
      if (arr.size() == 4 && arr[3] == 1)
        bias = op->getOperand(op->getNumOperands() - 1);
    }

    auto qTy = dyn_cast<RankedTensorType>(q.getType());
    auto kTy = dyn_cast<RankedTensorType>(k.getType());
    auto vTy = dyn_cast<RankedTensorType>(v.getType());
    if (!qTy || !kTy || !vTy)
      return failure();
    if (qTy.getRank() != 3 || kTy.getRank() != 3 || vTy.getRank() != 3)
      return rewriter.notifyMatchFailure(
          op, "AppleGPU flash-attn MSL path is rank-3 only in Phase 8.4.1");
    Type qElem = qTy.getElementType();
    if (qElem != kTy.getElementType() || qElem != vTy.getElementType())
      return rewriter.notifyMatchFailure(
          op, "AppleGPU flash-attn MSL path requires matching Q/K/V dtypes");
    // Phase 8.4.4.2 — pick symbol by element type. Q/K/V/O ABI changes
    // pointer interpretation (uint16_t* for f16/bf16) but the func signature
    // stays the same i64×4 + i32×4 + f32 + i32 since pointers are i64.
    const bool hasBias = static_cast<bool>(bias);
    StringRef symbol;
    if (qElem.isF32()) {
      symbol = hasBias ? kFlashAttnBiasF32Symbol : kFlashAttnF32Symbol;
    } else if (qElem.isF16()) {
      symbol = hasBias ? kFlashAttnBiasF16Symbol : kFlashAttnF16Symbol;
    } else if (qElem.isBF16()) {
      symbol = hasBias ? kFlashAttnBiasBF16Symbol : kFlashAttnBF16Symbol;
    } else {
      return rewriter.notifyMatchFailure(
          op, "AppleGPU flash-attn MSL path supports f32, f16, and bf16 in Phase 8.4.4.2");
    }
    // Bias must be a static rank-3 (B, Sq, Sk) tensor matching Q's dtype; the
    // Phase-1 Apple path requires bias batch == B (broadcast (1,Sq,Sk) is a
    // runtime follow-up). Out-of-envelope bias falls back to the reference path.
    if (hasBias) {
      auto bTy = dyn_cast<RankedTensorType>(bias.getType());
      if (!bTy || bTy.getRank() != 3 || !bTy.hasStaticShape() ||
          bTy.getElementType() != qElem)
        return rewriter.notifyMatchFailure(
            op, "AppleGPU bias attn path needs a static rank-3 bias matching Q dtype");
    }
    if (qTy.isDynamicDim(0) || qTy.isDynamicDim(1) || qTy.isDynamicDim(2) ||
        kTy.isDynamicDim(0) || kTy.isDynamicDim(1) || kTy.isDynamicDim(2) ||
        vTy.isDynamicDim(0) || vTy.isDynamicDim(1) || vTy.isDynamicDim(2))
      return rewriter.notifyMatchFailure(
          op, "AppleGPU flash-attn MSL path requires static shapes");

    int64_t B = qTy.getDimSize(0);
    int64_t Sq = qTy.getDimSize(1);
    int64_t Dq = qTy.getDimSize(2);
    int64_t Bk = kTy.getDimSize(0);
    int64_t Sk = kTy.getDimSize(1);
    int64_t Dk = kTy.getDimSize(2);
    int64_t Bv = vTy.getDimSize(0);
    int64_t Sv = vTy.getDimSize(1);
    int64_t Dv = vTy.getDimSize(2);
    if (B != Bk || B != Bv)
      return rewriter.notifyMatchFailure(op, "flash_attn batch mismatch");
    if (Sk != Sv)
      return rewriter.notifyMatchFailure(op, "flash_attn key/value seq mismatch");
    if (Dq != Dk || Dq != Dv)
      return rewriter.notifyMatchFailure(op, "flash_attn head_dim mismatch");
    if (Dq > 256)
      return rewriter.notifyMatchFailure(
          op, "AppleGPU flash-attn MSL kernel limited to head_dim <= 256");

    // attn_bias must be exactly (B, Sq, Sk): this path bufferizes the bias as a
    // (B, Sq, Sk) memref and passes B to the runtime, so a broadcast (1, Sq, Sk)
    // bias (legal per the op verifier) would create an invalid memref / let the
    // runtime read past the single-batch buffer. Reject it so it falls back to
    // the reference path (which numpy-broadcasts the bias correctly).
    if (hasBias) {
      auto bTy = dyn_cast<RankedTensorType>(bias.getType());
      if (!bTy)
        return rewriter.notifyMatchFailure(
            op, "AppleGPU bias attn path needs a ranked attn_bias tensor");
      if (bTy.getDimSize(0) != B || bTy.getDimSize(1) != Sq ||
          bTy.getDimSize(2) != Sk)
        return rewriter.notifyMatchFailure(
            op, "AppleGPU bias attn path needs attn_bias of exact shape "
                "(B, Sq, Sk); broadcast bias falls back to the reference path");
    }

    // Defaults match the numpy reference in tessera.runtime._runtime_flash_attn.
    float scale = 1.0f / std::sqrt(static_cast<float>(Dq));
    if (auto attr = op->getAttrOfType<FloatAttr>("scale"))
      scale = static_cast<float>(attr.getValueAsDouble());
    int32_t causal = 0;
    if (auto attr = op->getAttrOfType<BoolAttr>("causal"))
      causal = attr.getValue() ? 1 : 0;

    Location loc = op->getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = op->getContext();

    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();
    Type f32Ty = rewriter.getF32Type();

    auto qMemTy = MemRefType::get({B, Sq, Dq}, qElem);
    auto kMemTy = MemRefType::get({B, Sk, Dq}, qElem);
    auto vMemTy = MemRefType::get({B, Sk, Dq}, qElem);
    auto oMemTy = MemRefType::get({B, Sq, Dq}, qElem);

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

    Value Bv32  = rewriter.create<arith::ConstantIntOp>(loc, B, 32);
    Value Sqv32 = rewriter.create<arith::ConstantIntOp>(loc, Sq, 32);
    Value Skv32 = rewriter.create<arith::ConstantIntOp>(loc, Sk, 32);
    Value Dv32  = rewriter.create<arith::ConstantIntOp>(loc, Dq, 32);

    auto scaleAttr = rewriter.getF32FloatAttr(scale);
    Value scaleV = rewriter.create<arith::ConstantOp>(loc, f32Ty, scaleAttr);

    Value causalV = rewriter.create<arith::ConstantIntOp>(loc, causal, 32);

    if (hasBias) {
      // softmax(scale*Q*K^T + bias)*V — one extra bias pointer after V.
      auto biasMemTy = MemRefType::get({B, Sq, Sk}, qElem);
      Value biasPtr = extractPtr(rewriter, loc, bias, biasMemTy);
      FunctionType fnTy = FunctionType::get(
          ctx,
          {i64Ty, i64Ty, i64Ty, i64Ty, i64Ty,  // Q, K, V, bias, O ptrs
           i32Ty, i32Ty, i32Ty, i32Ty,         // B, Sq, Sk, D
           f32Ty,                              // scale
           i32Ty},                             // causal
          {});
      ensureExternalDecl(mod, symbol, fnTy);
      rewriter.create<func::CallOp>(
          loc, symbol, TypeRange{},
          ValueRange{qPtr, kPtr, vPtr, biasPtr, oPtr,
                     Bv32, Sqv32, Skv32, Dv32,
                     scaleV, causalV});
    } else {
      FunctionType fnTy = FunctionType::get(
          ctx,
          {i64Ty, i64Ty, i64Ty, i64Ty,        // Q, K, V, O ptrs
           i32Ty, i32Ty, i32Ty, i32Ty,        // B, Sq, Sk, D
           f32Ty,                             // scale
           i32Ty},                            // causal
          {});
      ensureExternalDecl(mod, symbol, fnTy);
      rewriter.create<func::CallOp>(
          loc, symbol, TypeRange{},
          ValueRange{qPtr, kPtr, vPtr, oPtr,
                     Bv32, Sqv32, Skv32, Dv32,
                     scaleV, causalV});
    }

    auto outTensorTy = RankedTensorType::get({B, Sq, Dq}, qElem);
    Value result =
        rewriter.create<bufferization::ToTensorOp>(loc, outTensorTy, oAlloc);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerFlashAttnToAppleGPUPass
    : public PassWrapper<LowerFlashAttnToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerFlashAttnToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-flash_attn-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Lower tessera.flash_attn (rank-3, f32/f16/bf16, head_dim <= 256) "
           "to Apple GPU runtime calls (custom MSL kernel)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerFlashAttnToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerFlashAttnToAppleGPUPass() {
  return std::make_unique<LowerFlashAttnToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
