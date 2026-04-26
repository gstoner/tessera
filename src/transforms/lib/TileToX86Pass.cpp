
// TileToX86Pass.cpp
//
// Replaces tessera.matmul and tessera.fused_epilogue ops (with static bf16/f16
// input tensors) with calls to the tessera_x86_backend C functions:
//
//   tessera_x86_amx_gemm_bf16  (A: bf16, B: bf16 → C: f32)
//   tessera_x86_avx512_gemm_bf16
//   tessera_x86_epilogue_bias_fp32
//   tessera_x86_epilogue_bias_gelu_fp32
//
// Lowering strategy
// ─────────────────
// For each tessera.matmul %A, %B : tensor<MxKxbf16>, tensor<KxNxbf16>
//                                → tensor<MxNxf32>:
//
//   1. bufferization.to_memref %A → memref<MxKxbf16>
//   2. bufferization.to_memref %B → memref<KxNxbf16>
//   3. memref.alloc()             → memref<MxNxf32>
//   4. Declare the external C function in the module (once).
//   5. memref.extract_aligned_pointer_as_index → index
//      arith.index_cast → i64   (raw pointer as integer)
//   6. func.call @tessera_x86_amx_gemm_bf16(aPtr, bPtr, cPtr, M, N, K, beta)
//   7. bufferization.to_tensor %C_buf → tensor<MxNxf32>
//   8. Replace the original result.
//
// For tessera.fused_epilogue (matmul + bias + epilogue):
//   Same GEMM lowering, followed by a call to the appropriate epilogue C fn.
//
// Pass option
//   --prefer-amx  prefer AMX over AVX-512 (default true); set false to always
//                 emit the AVX-512 call.

#include "Tessera/Transforms/Passes.h"
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

using namespace mlir;

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Ensure a private external function declaration exists in the module.
static func::FuncOp ensureExternalDecl(ModuleOp mod, StringRef name,
                                       FunctionType fnTy) {
  if (auto fn = mod.lookupSymbol<func::FuncOp>(name)) return fn;
  OpBuilder b(mod.getBodyRegion());
  b.setInsertionPointToStart(mod.getBody());
  auto fn = b.create<func::FuncOp>(mod.getLoc(), name, fnTy);
  fn.setPrivate();
  return fn;
}

// Emit bufferization.to_memref and extract a raw pointer as i64.
// Returns the i64 value representing the aligned data pointer.
static Value extractPtr(OpBuilder &b, Location loc, Value tensor,
                        MemRefType memTy) {
  auto buf = b.create<bufferization::ToMemrefOp>(loc, memTy, tensor);
  auto ptrIdx =
      b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buf);
  return b.create<arith::IndexCastOp>(loc, b.getI64Type(), ptrIdx);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pattern: LowerMatmulToX86
// ─────────────────────────────────────────────────────────────────────────────

struct LowerMatmulToX86 : public RewritePattern {
  LowerMatmulToX86(MLIRContext *ctx, bool preferAMX)
      : RewritePattern("tessera.matmul", /*benefit=*/2, ctx),
        preferAMX_(preferAMX) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2 || op->getNumResults() != 1)
      return failure();

    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);

    auto lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsTy = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhsTy || !rhsTy || lhsTy.getRank() != 2 || rhsTy.getRank() != 2)
      return failure();

    // Only lower bf16 and f16 GEMMs (the x86 backend supports these).
    Type lhsElem = lhsTy.getElementType();
    if (!lhsElem.isBF16() && !lhsElem.isF16()) return failure();

    // Require static shapes.
    if (lhsTy.isDynamicDim(0) || lhsTy.isDynamicDim(1) ||
        rhsTy.isDynamicDim(0) || rhsTy.isDynamicDim(1))
      return failure();

    int64_t M = lhsTy.getDimSize(0);
    int64_t K = lhsTy.getDimSize(1);
    int64_t N = rhsTy.getDimSize(1);
    if (rhsTy.getDimSize(0) != K) return failure();

    Location loc = op->getLoc();
    ModuleOp mod  = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = op->getContext();

    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();
    Type f32Ty = rewriter.getF32Type();

    // ── Emit pointer extraction ────────────────────────────────────────────
    auto lhsMemTy = MemRefType::get({M, K}, lhsElem);
    auto rhsMemTy = MemRefType::get({K, N}, lhsElem);
    auto outMemTy = MemRefType::get({M, N}, f32Ty);

    Value aPtr = extractPtr(rewriter, loc, lhs, lhsMemTy);
    Value bPtr = extractPtr(rewriter, loc, rhs, rhsMemTy);

    // Allocate output buffer.
    auto cAlloc = rewriter.create<memref::AllocOp>(loc, outMemTy);
    Value cPtr;
    {
      auto ptrIdx =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, cAlloc);
      cPtr = rewriter.create<arith::IndexCastOp>(loc, i64Ty, ptrIdx);
    }

    // ── Scalar dimension and beta constants ───────────────────────────────
    Value Mv = rewriter.create<arith::ConstantIntOp>(loc, M, 32);
    Value Nv = rewriter.create<arith::ConstantIntOp>(loc, N, 32);
    Value Kv = rewriter.create<arith::ConstantIntOp>(loc, K, 32);
    Value betaV = rewriter.create<arith::ConstantFloatOp>(
        loc, APFloat(0.0f), f32Ty);

    // ── Declare and call the x86 kernel ───────────────────────────────────
    StringRef kernelName = preferAMX_ ? "tessera_x86_amx_gemm_bf16"
                                      : "tessera_x86_avx512_gemm_bf16";
    FunctionType kernelFnTy = FunctionType::get(
        ctx, {i64Ty, i64Ty, i64Ty, i32Ty, i32Ty, i32Ty, f32Ty}, {});
    ensureExternalDecl(mod, kernelName, kernelFnTy);

    rewriter.create<func::CallOp>(
        loc, kernelName, TypeRange{},
        ValueRange{aPtr, bPtr, cPtr, Mv, Nv, Kv, betaV});

    // ── Wrap output buffer as tensor and replace op ────────────────────────
    auto outTensorTy = RankedTensorType::get({M, N}, f32Ty);
    Value result = rewriter.create<bufferization::ToTensorOp>(
        loc, outTensorTy, cAlloc);

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  bool preferAMX_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Pattern: LowerFusedEpilogueToX86
// ─────────────────────────────────────────────────────────────────────────────
// tessera.fused_epilogue %lhs, %rhs, %bias {epilogue=Gelu|Relu, has_bias=true}
//   → same GEMM call, then tessera_x86_epilogue_bias_gelu_fp32 / _bias_fp32

struct LowerFusedEpilogueToX86 : public RewritePattern {
  LowerFusedEpilogueToX86(MLIRContext *ctx, bool preferAMX)
      : RewritePattern("tessera.fused_epilogue", /*benefit=*/2, ctx),
        preferAMX_(preferAMX) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 3) return failure();

    Value lhs  = op->getOperand(0);
    Value rhs  = op->getOperand(1);
    Value bias = op->getOperand(2);

    auto lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsTy = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhsTy || !rhsTy || lhsTy.getRank() != 2 || rhsTy.getRank() != 2)
      return failure();
    Type lhsElem = lhsTy.getElementType();
    if (!lhsElem.isBF16() && !lhsElem.isF16()) return failure();
    if (lhsTy.isDynamicDim(0) || lhsTy.isDynamicDim(1) ||
        rhsTy.isDynamicDim(0) || rhsTy.isDynamicDim(1))
      return failure();

    int64_t M = lhsTy.getDimSize(0);
    int64_t K = lhsTy.getDimSize(1);
    int64_t N = rhsTy.getDimSize(1);
    if (rhsTy.getDimSize(0) != K) return failure();

    // Epilogue kind: 2=Gelu, 1=Relu (matches EpilogueKind enum in ODS).
    int epilogueKind = 0;
    if (auto ek = op->getAttrOfType<IntegerAttr>("epilogue"))
      epilogueKind = static_cast<int>(ek.getInt());

    Location loc   = op->getLoc();
    ModuleOp  mod  = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = op->getContext();

    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();
    Type f32Ty = rewriter.getF32Type();

    // GEMM part (same as LowerMatmulToX86).
    auto lhsMemTy = MemRefType::get({M, K}, lhsElem);
    auto rhsMemTy = MemRefType::get({K, N}, lhsElem);
    auto outMemTy = MemRefType::get({M, N}, f32Ty);

    Value aPtr = extractPtr(rewriter, loc, lhs, lhsMemTy);
    Value bPtr = extractPtr(rewriter, loc, rhs, rhsMemTy);
    auto  cAlloc = rewriter.create<memref::AllocOp>(loc, outMemTy);
    Value cPtr;
    {
      auto pi = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
          loc, cAlloc);
      cPtr = rewriter.create<arith::IndexCastOp>(loc, i64Ty, pi);
    }

    Value Mv = rewriter.create<arith::ConstantIntOp>(loc, M, 32);
    Value Nv = rewriter.create<arith::ConstantIntOp>(loc, N, 32);
    Value Kv = rewriter.create<arith::ConstantIntOp>(loc, K, 32);
    Value betaV = rewriter.create<arith::ConstantFloatOp>(
        loc, APFloat(0.0f), f32Ty);

    StringRef gemmName = preferAMX_ ? "tessera_x86_amx_gemm_bf16"
                                    : "tessera_x86_avx512_gemm_bf16";
    FunctionType gemmFnTy = FunctionType::get(
        ctx, {i64Ty, i64Ty, i64Ty, i32Ty, i32Ty, i32Ty, f32Ty}, {});
    ensureExternalDecl(mod, gemmName, gemmFnTy);
    rewriter.create<func::CallOp>(
        loc, gemmName, TypeRange{},
        ValueRange{aPtr, bPtr, cPtr, Mv, Nv, Kv, betaV});

    // Epilogue: bias (always) + optional activation.
    bool hasBias = op->getAttrOfType<BoolAttr>("has_bias") &&
                   op->getAttrOfType<BoolAttr>("has_bias").getValue();
    if (hasBias) {
      // Extract bias pointer.
      auto biasTy = bias.getType().dyn_cast<RankedTensorType>();
      if (biasTy && !biasTy.isDynamicDim(0)) {
        auto biasMemTy = MemRefType::get({N}, f32Ty);
        Value biasPtr = extractPtr(rewriter, loc, bias, biasMemTy);

        StringRef epilogueName;
        if (epilogueKind == 2)      // Gelu
          epilogueName = "tessera_x86_epilogue_bias_gelu_fp32";
        else                        // Relu or None-with-bias
          epilogueName = "tessera_x86_epilogue_bias_fp32";

        FunctionType epFnTy = FunctionType::get(
            ctx, {i64Ty, i64Ty, i32Ty, i32Ty}, {});
        ensureExternalDecl(mod, epilogueName, epFnTy);
        rewriter.create<func::CallOp>(
            loc, epilogueName, TypeRange{},
            ValueRange{cPtr, biasPtr, Mv, Nv});
      }
    }

    auto outTensorTy = RankedTensorType::get({M, N}, f32Ty);
    Value result = rewriter.create<bufferization::ToTensorOp>(
        loc, outTensorTy, cAlloc);

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  bool preferAMX_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Pass
// ─────────────────────────────────────────────────────────────────────────────

struct TileToX86PassImpl
    : public PassWrapper<TileToX86PassImpl, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileToX86PassImpl)

  Option<bool> preferAMXOpt{
      *this, "prefer-amx",
      llvm::cl::desc("Prefer AMX over AVX-512 GEMM kernels"),
      llvm::cl::init(true)};

  StringRef getArgument()    const override { return "tessera-tile-to-x86"; }
  StringRef getDescription() const override {
    return "Lower tiled tessera.matmul to tessera_x86_backend C function calls";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    bool amx = preferAMXOpt;
    patterns.add<LowerMatmulToX86>(&getContext(), amx);
    patterns.add<LowerFusedEpilogueToX86>(&getContext(), amx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                           std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createTileToX86Pass() {
  return std::make_unique<TileToX86PassImpl>();
}
} // namespace tessera
