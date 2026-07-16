
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

#include "Tessera/Common/Lowering.h"
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
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// The bufferize->ptr->func.call C-ABI helpers are shared with the Apple backend
// (Workstream A1) — see Tessera/Common/Lowering.h. `using` keeps the unqualified
// call sites below (extractPtr / ensureExternalDecl) unchanged.
using tessera::common::ensureExternalDecl;
using tessera::common::extractPtr;

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

    auto lhsTy = llvm::dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = llvm::dyn_cast<RankedTensorType>(rhs.getType());
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
        loc, llvm::cast<FloatType>(f32Ty), APFloat(0.0f));

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

    auto lhsTy = llvm::dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = llvm::dyn_cast<RankedTensorType>(rhs.getType());
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
        loc, llvm::cast<FloatType>(f32Ty), APFloat(0.0f));

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
      auto biasTy = llvm::dyn_cast<RankedTensorType>(bias.getType());
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
// kv_cache_coverage_matrix.md (2026-05-10) — KV-cache lowering on x86.
//
// The runtime now ships native contiguous-f32 append/read/prune symbols in
// libtessera_x86_elementwise.so. This legacy opaque-handle rewrite lowers
// `tessera.kv_cache.{create,append,prune,read}` to a single host-shaped
// runtime call:
//
//   tessera_x86_kv_cache_op(kind: i32, args...)
//
// This artifact call remains the bridge for opaque `!tessera.kv_cache` values;
// bufferized f32 execution uses `tessera_x86_kv_cache_{append,read,prune}_f32`
// through the registered `x86_kv_cache_compiled` runtime lane.
//
// `kind` enum (matches the same handle-style discrimination the Apple
// path uses):
//   0 = create, 1 = append, 2 = prune, 3 = read
//
// We emit attribute-only artifact ops here because this pass has no buffer
// pointers for the opaque handle. It must not be used as native-execution
// evidence; the buffer ABI, execution-matrix row, and numerical fixture own
// that proof.
// ─────────────────────────────────────────────────────────────────────────────

struct LowerKVCacheToX86 : public RewritePattern {
  LowerKVCacheToX86(MLIRContext *ctx, StringRef opName)
      : RewritePattern(opName, /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    StringRef name = op->getName().getStringRef();
    int32_t kindInt = 0;
    if (name == "tessera.kv_cache.create")      kindInt = 0;
    else if (name == "tessera.kv_cache.append") kindInt = 1;
    else if (name == "tessera.kv_cache.prune")  kindInt = 2;
    else if (name == "tessera.kv_cache.read")   kindInt = 3;
    else
      return rewriter.notifyMatchFailure(op, "unknown kv_cache op");

    // For ops with consumed results we'd need handle-shaped runtime
    // wiring; in v1 the x86 backend is artifact-only for kv_cache (the
    // Python runtime drives the real `KVCacheHandle` execution path).
    // Skip ops whose results are consumed downstream so we don't break
    // the IR.
    for (Value r : op->getResults()) {
      if (!r.use_empty())
        return rewriter.notifyMatchFailure(
            op, "x86 kv_cache lowering is artifact-only; result is "
                "consumed downstream — keep the op live and let the "
                "Python runtime path drive it");
    }

    ModuleOp mod = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = op->getContext();
    Type i32Ty = rewriter.getI32Type();
    FunctionType fnTy = FunctionType::get(ctx, {i32Ty}, {});
    ensureExternalDecl(mod, "tessera_x86_kv_cache_op", fnTy);

    rewriter.setInsertionPoint(op);
    Value kindV = rewriter.create<arith::ConstantIntOp>(loc, kindInt, 32);
    rewriter.create<func::CallOp>(loc, "tessera_x86_kv_cache_op",
                                   TypeRange{}, ValueRange{kindV});
    rewriter.eraseOp(op);
    return success();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pass
// ─────────────────────────────────────────────────────────────────────────────

struct TileToX86PassImpl
    : public PassWrapper<TileToX86PassImpl, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileToX86PassImpl)

  TileToX86PassImpl() = default;
  TileToX86PassImpl(const TileToX86PassImpl &other)
      : PassWrapper(other) {}

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
    // kv_cache_coverage_matrix.md (2026-05-10) — KV-cache artifact
    // lowering on x86. Each op gets its own pattern instance so the
    // GreedyPatternRewriteDriver can match by string name.
    patterns.add<LowerKVCacheToX86>(&getContext(), "tessera.kv_cache.create");
    patterns.add<LowerKVCacheToX86>(&getContext(), "tessera.kv_cache.append");
    patterns.add<LowerKVCacheToX86>(&getContext(), "tessera.kv_cache.prune");
    patterns.add<LowerKVCacheToX86>(&getContext(), "tessera.kv_cache.read");
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozenPatterns)))
      signalPassFailure();
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createTileToX86Pass() {
  return std::make_unique<TileToX86PassImpl>();
}
} // namespace tessera
