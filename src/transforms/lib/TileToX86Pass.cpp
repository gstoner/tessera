
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
#include "Tessera/Dialect/Tile/TileDialect.h"
#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
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
                    func::FuncDialect, LLVM::LLVMDialect,
                    memref::MemRefDialect>();
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
      return signalPassFailure();

    // X86-E2E-1 launch envelopes already carry raw pointers and dimensions.
    // Adapt them to the stable x86 shared-library ABI here so Python never
    // synthesizes backend Target IR or rediscovers a symbol signature.
    SmallVector<Operation *> launchEnvelopes;
    getOperation().walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tile.matmul_kernel" || name == "tile.softmax_kernel" ||
          name == "tile.reduce_kernel" || name == "tile.attention_kernel" ||
          name == "tile.elementwise_kernel" ||
          name == "tile.argreduce_kernel" || name == "tile.scan_kernel" ||
          name == "tile.norm_kernel" || name == "tile.rope_kernel" ||
          name == "tile.alibi_kernel" || name == "tile.x86_abi_kernel")
        launchEnvelopes.push_back(op);
    });
    ModuleOp module = getOperation();
    for (Operation *op : launchEnvelopes) {
      OpBuilder builder(op);
      Location loc = op->getLoc();
      MLIRContext *ctx = op->getContext();
      Type ptrTy = LLVM::LLVMPointerType::get(ctx);
      Type i64Ty = builder.getI64Type();
      Type i32Ty = builder.getI32Type();
      Type f32Ty = builder.getF32Type();
      StringRef opName = op->getName().getStringRef();
      if (opName == "tile.x86_abi_kernel") {
        auto symbol = op->getAttrOfType<StringAttr>("symbol");
        if (!symbol) {
          op->emitError("x86 stable-ABI carrier requires symbol");
          return signalPassFailure();
        }
        SmallVector<Type> argumentTypes;
        argumentTypes.reserve(op->getNumOperands());
        for (Value operand : op->getOperands())
          argumentTypes.push_back(operand.getType());
        auto returnsStatus = op->getAttrOfType<BoolAttr>("returns_status");
        SmallVector<Type> resultTypes;
        if (returnsStatus && returnsStatus.getValue())
          resultTypes.push_back(i32Ty);
        ensureExternalDecl(module, symbol.getValue(),
                           FunctionType::get(ctx, argumentTypes, resultTypes));
        builder.create<func::CallOp>(loc, symbol.getValue(), resultTypes,
                                     op->getOperands());
        op->erase();
        continue;
      }
      if (opName == "tile.argreduce_kernel" || opName == "tile.scan_kernel") {
        auto kind = op->getAttrOfType<StringAttr>("kind");
        int32_t kindValue = -1;
        StringRef symbol;
        if (opName == "tile.argreduce_kernel") {
          kindValue = llvm::StringSwitch<int32_t>(kind.getValue())
                          .Case("argmax", 0).Case("argmin", 1).Default(-1);
          symbol = "tessera_x86_avx512_argreduce_f32";
        } else {
          kindValue = llvm::StringSwitch<int32_t>(kind.getValue())
                          .Case("sum", 0).Case("product", 1)
                          .Case("max", 2).Case("min", 3).Default(-1);
          symbol = "tessera_x86_avx512_scan_f32";
        }
        if (kindValue < 0) {
          op->emitError("X86-E2E-2 reduction/scan kind has no stable ABI mapping");
          return signalPassFailure();
        }
        ensureExternalDecl(module, symbol,
                           FunctionType::get(ctx, {ptrTy, i64Ty, i64Ty, ptrTy, i32Ty}, {}));
        Value kindConstant = builder.create<arith::ConstantIntOp>(loc, kindValue, 32);
        builder.create<func::CallOp>(
            loc, symbol, TypeRange{},
            ValueRange{op->getOperand(0), op->getOperand(2), op->getOperand(3),
                       op->getOperand(1), kindConstant});
        op->erase();
        continue;
      }
      if (opName == "tile.norm_kernel") {
        auto kind = op->getAttrOfType<StringAttr>("kind");
        StringRef symbol = kind.getValue() == "rmsnorm"
                               ? "tessera_x86_avx512_rmsnorm_f32"
                               : "tessera_x86_avx512_layernorm_f32";
        ensureExternalDecl(module, symbol,
                           FunctionType::get(ctx, {ptrTy, i64Ty, i64Ty, f32Ty, ptrTy}, {}));
        builder.create<func::CallOp>(
            loc, symbol, TypeRange{},
            ValueRange{op->getOperand(0), op->getOperand(2), op->getOperand(3),
                       op->getOperand(4), op->getOperand(1)});
        op->erase();
        continue;
      }
      if (opName == "tile.rope_kernel") {
        StringRef symbol = "tessera_x86_avx512_rope_f32";
        ensureExternalDecl(module, symbol,
                           FunctionType::get(ctx, {ptrTy, ptrTy, i64Ty, i64Ty, ptrTy}, {}));
        builder.create<func::CallOp>(
            loc, symbol, TypeRange{},
            ValueRange{op->getOperand(0), op->getOperand(1), op->getOperand(3),
                       op->getOperand(4), op->getOperand(2)});
        op->erase();
        continue;
      }
      if (opName == "tile.alibi_kernel") {
        StringRef symbol = "tessera_x86_avx512_alibi_f32";
        ensureExternalDecl(module, symbol,
                           FunctionType::get(ctx, {ptrTy, i64Ty, i64Ty, ptrTy}, {}));
        builder.create<func::CallOp>(
            loc, symbol, TypeRange{},
            ValueRange{op->getOperand(0), op->getOperand(2), op->getOperand(3),
                       op->getOperand(1)});
        op->erase();
        continue;
      }
      if (op->getName().getStringRef() == "tile.elementwise_kernel") {
        auto family = op->getAttrOfType<StringAttr>("family");
        auto kind = op->getAttrOfType<StringAttr>("kind");
        auto storage = op->getAttrOfType<StringAttr>("storage");
        auto outputStorage = op->getAttrOfType<StringAttr>("output_storage");
        if (!family || !kind || !storage || !outputStorage) {
          op->emitError("X86-E2E-2 elementwise requires an explicit family/kind/storage/output contract");
          return signalPassFailure();
        }
        int32_t kindValue = -1;
        StringRef symbol;
        if (family.getValue() == "unary") {
          kindValue = llvm::StringSwitch<int32_t>(kind.getValue())
              .Case("sqrt", 0).Case("rsqrt", 1).Case("reciprocal", 2)
              .Case("abs", 3).Case("sign", 5).Case("floor", 6)
              .Case("ceil", 7).Case("trunc", 8).Case("round", 9)
              .Default(-1);
          symbol = "tessera_x86_avx512_unary_f32";
        } else if (family.getValue() == "binary") {
          kindValue = llvm::StringSwitch<int32_t>(kind.getValue())
              .Case("sub", 0).Case("div", 1).Case("maximum", 2)
              .Case("minimum", 3).Case("add", 4).Case("mul", 5)
              .Case("mod", 6).Case("floor_div", 7).Default(-1);
          symbol = "tessera_x86_avx512_binary_f32";
        } else if (family.getValue() == "predicate") {
          kindValue = llvm::StringSwitch<int32_t>(kind.getValue())
              .Case("isnan", 0).Case("isinf", 1).Case("isfinite", 2)
              .Default(-1);
          symbol = "tessera_x86_avx512_predicate_f32";
        } else if (family.getValue() == "compare") {
          kindValue = llvm::StringSwitch<int32_t>(kind.getValue())
              .Case("eq", 0).Case("ne", 1).Case("lt", 2)
              .Case("le", 3).Case("gt", 4).Case("ge", 5).Default(-1);
          symbol = "tessera_x86_avx512_compare_f32";
        } else if (family.getValue() == "logical") {
          kindValue = llvm::StringSwitch<int32_t>(kind.getValue())
              .Case("and", 0).Case("or", 1).Case("xor", 2)
              .Case("not", 3).Default(-1);
          symbol = "tessera_x86_avx512_logical_i8";
        } else if (family.getValue() == "bitwise") {
          kindValue = llvm::StringSwitch<int32_t>(kind.getValue())
              .Case("and", 0).Case("or", 1).Case("xor", 2)
              .Case("not", 3).Case("popcount", 4).Default(-1);
          symbol = "tessera_x86_avx512_bitwise_i32";
        } else if (family.getValue() == "where") {
          kindValue = kind.getValue() == "where" ? 0 : -1;
          symbol = "tessera_x86_avx512_where_f32";
        } else if (family.getValue() == "transcendental") {
          kindValue = llvm::StringSwitch<int32_t>(kind.getValue())
              .Case("exp", 0).Case("log", 1).Case("tanh", 2)
              .Case("sigmoid", 3).Case("silu", 4).Case("gelu", 5)
              .Case("erf", 6).Case("softplus", 7).Case("expm1", 8)
              .Case("log1p", 9).Case("cos", 10).Case("tan", 11)
              .Case("sinh", 12).Case("cosh", 13).Case("asin", 14)
              .Case("acos", 15).Case("atan", 16).Case("erfc", 17)
              .Case("sin", 18).Case("lgamma", 19).Case("digamma", 20)
              .Default(-1);
          symbol = "tessera_x86_avx512_transcendental_f32";
        } else if (family.getValue() == "binary_math") {
          if (kind.getValue() == "pow") {
            kindValue = 0;
            symbol = "tessera_x86_avx512_pow_f32";
          } else if (kind.getValue() == "silu_mul") {
            kindValue = 1;
            symbol = "tessera_x86_avx512_silu_mul_f32";
          }
        }
        if (kindValue < 0 || symbol.empty()) {
          op->emitError("X86-E2E-2 elementwise family/kind has no stable ABI mapping");
          return signalPassFailure();
        }
        Value kindConstant = builder.create<arith::ConstantIntOp>(loc, kindValue, 32);
        if (family.getValue() == "where") {
          ensureExternalDecl(
              module, symbol,
              FunctionType::get(ctx, {ptrTy, ptrTy, ptrTy, i64Ty, ptrTy}, {}));
          builder.create<func::CallOp>(
              loc, symbol, TypeRange{},
              ValueRange{op->getOperand(0), op->getOperand(1),
                         op->getOperand(2), op->getOperand(4),
                         op->getOperand(3)});
          op->erase();
          continue;
        }
        if (family.getValue() == "binary_math") {
          ensureExternalDecl(
              module, symbol,
              FunctionType::get(ctx, {ptrTy, ptrTy, i64Ty, ptrTy}, {}));
          builder.create<func::CallOp>(
              loc, symbol, TypeRange{},
              ValueRange{op->getOperand(0), op->getOperand(1),
                         op->getOperand(3), op->getOperand(2)});
          op->erase();
          continue;
        }
        bool fixedBinaryABI = family.getValue() == "binary" ||
                              family.getValue() == "compare" ||
                              family.getValue() == "logical" ||
                              family.getValue() == "bitwise";
        if (fixedBinaryABI) {
          bool hasB = op->getNumOperands() == 4;
          Value b = hasB ? op->getOperand(1)
                         : builder.create<LLVM::ZeroOp>(loc, ptrTy).getResult();
          unsigned outputIndex = hasB ? 2 : 1;
          unsigned nIndex = hasB ? 3 : 2;
          ensureExternalDecl(
              module, symbol,
              FunctionType::get(ctx, {ptrTy, ptrTy, i64Ty, ptrTy, i32Ty}, {}));
          builder.create<func::CallOp>(
              loc, symbol, TypeRange{},
              ValueRange{op->getOperand(0), b, op->getOperand(nIndex),
                         op->getOperand(outputIndex), kindConstant});
        } else {
          ensureExternalDecl(
              module, symbol,
              FunctionType::get(ctx, {ptrTy, i64Ty, ptrTy, i32Ty}, {}));
          builder.create<func::CallOp>(
              loc, symbol, TypeRange{},
              ValueRange{op->getOperand(0), op->getOperand(2),
                         op->getOperand(1), kindConstant});
        }
        op->erase();
        continue;
      }
      if (op->getName().getStringRef() == "tile.matmul_kernel") {
        auto mma = op->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma");
        auto epilogue =
            op->getAttrOfType<tessera::tile::TileEpilogueAttr>("epilogue");
        auto residual = op->getAttrOfType<BoolAttr>("residual");
        if (op->getNumOperands() != 6 || !mma || !epilogue || epilogue.getBias() ||
            epilogue.getActivation() != "none" ||
            (residual && residual.getValue())) {
          op->emitError("x86 matmul requires a supported plain A/B/acc/output contract and A/B/D/M/N/K");
          return signalPassFailure();
        }
        StringRef aType = mma.getAType(), bType = mma.getBType();
        StringRef accType = mma.getAccType(), outputType = epilogue.getOutputType();
        if (aType == "f32" && bType == "f32" && accType == "f32" && outputType == "f32") {
          StringRef symbol = "tessera_x86_avx512_gemm_f32";
          ensureExternalDecl(module, symbol,
                             FunctionType::get(ctx, {ptrTy, ptrTy, i64Ty, i64Ty,
                                                     i64Ty, ptrTy}, {}));
          builder.create<func::CallOp>(
              loc, symbol, TypeRange{},
              ValueRange{op->getOperand(0), op->getOperand(1),
                         op->getOperand(3), op->getOperand(4),
                         op->getOperand(5), op->getOperand(2)});
        } else if (aType == "f64" && bType == "f64" && accType == "f64" && outputType == "f64") {
          StringRef symbol = "tessera_x86_avx512_gemm_f64";
          ensureExternalDecl(module, symbol,
                             FunctionType::get(ctx, {ptrTy, ptrTy, i64Ty, i64Ty,
                                                     i64Ty, ptrTy}, {}));
          builder.create<func::CallOp>(
              loc, symbol, TypeRange{},
              ValueRange{op->getOperand(0), op->getOperand(1),
                         op->getOperand(3), op->getOperand(4),
                         op->getOperand(5), op->getOperand(2)});
        } else if ((aType == "bf16" && bType == "bf16" && accType == "f32" && outputType == "f32") ||
                   (aType == "u8" && bType == "i8" && accType == "i32" && outputType == "i32")) {
          Value m32 = builder.create<arith::TruncIOp>(loc, i32Ty, op->getOperand(3));
          Value n32 = builder.create<arith::TruncIOp>(loc, i32Ty, op->getOperand(4));
          Value k32 = builder.create<arith::TruncIOp>(loc, i32Ty, op->getOperand(5));
          if (aType == "bf16") {
            StringRef symbol = "tessera_x86_avx512_gemm_bf16";
            Value beta = builder.create<arith::ConstantFloatOp>(
                loc, cast<FloatType>(f32Ty), APFloat(0.0f));
            ensureExternalDecl(module, symbol,
                               FunctionType::get(ctx, {ptrTy, ptrTy, ptrTy, i32Ty,
                                                       i32Ty, i32Ty, f32Ty}, {}));
            builder.create<func::CallOp>(
                loc, symbol, TypeRange{},
                ValueRange{op->getOperand(0), op->getOperand(1), op->getOperand(2),
                           m32, n32, k32, beta});
          } else {
            StringRef symbol = "tessera_x86_avx512_vnni_gemm_u8s8_s32";
            Value beta = builder.create<arith::ConstantIntOp>(loc, 0, 32);
            ensureExternalDecl(module, symbol,
                               FunctionType::get(ctx, {ptrTy, ptrTy, ptrTy, i32Ty,
                                                       i32Ty, i32Ty, i32Ty}, {}));
            builder.create<func::CallOp>(
                loc, symbol, TypeRange{},
                ValueRange{op->getOperand(0), op->getOperand(1), op->getOperand(2),
                           m32, n32, k32, beta});
          }
        } else {
          op->emitError("x86 matmul dtype contract is not supported");
          return signalPassFailure();
        }
        op->erase();
        continue;
      }
      if (op->getName().getStringRef() == "tile.softmax_kernel") {
        auto storage = op->getAttrOfType<StringAttr>("storage");
        auto accum = op->getAttrOfType<StringAttr>("accum");
        auto axis = op->getAttrOfType<IntegerAttr>("axis");
        if (op->getNumOperands() != 4 || !storage || storage.getValue() != "f32" ||
            !accum || accum.getValue() != "f32" || !axis || axis.getInt() != -1) {
          op->emitError("X86-E2E-1 softmax requires f32 storage/accumulation, axis=-1, and X/O/Rows/K");
          return signalPassFailure();
        }
        StringRef symbol = "tessera_x86_avx512_softmax_f32";
        ensureExternalDecl(module, symbol,
                           FunctionType::get(ctx, {ptrTy, i64Ty, i64Ty, ptrTy}, {}));
        builder.create<func::CallOp>(
            loc, symbol, TypeRange{},
            ValueRange{op->getOperand(0), op->getOperand(2),
                       op->getOperand(3), op->getOperand(1)});
        op->erase();
        continue;
      }

      if (op->getName().getStringRef() == "tile.attention_kernel") {
        auto storage = op->getAttrOfType<StringAttr>("storage");
        auto accum = op->getAttrOfType<StringAttr>("accum");
        auto scale = op->getAttrOfType<FloatAttr>("scale");
        auto causal = op->getAttrOfType<BoolAttr>("causal");
        auto bias = op->getAttrOfType<BoolAttr>("bias");
        auto left = op->getAttrOfType<IntegerAttr>("window_left");
        auto right = op->getAttrOfType<IntegerAttr>("window_right");
        auto softcap = op->getAttrOfType<FloatAttr>("softcap");
        auto dropout = op->getAttrOfType<FloatAttr>("dropout_p");
        bool hasBias = bias && bias.getValue();
        unsigned pointerCount = 4 + unsigned(hasBias);
        if (op->getNumOperands() != 11 + unsigned(hasBias) || !storage ||
            storage.getValue() != "f32" || !accum ||
            accum.getValue() != "f32" || !scale || !causal || !bias ||
            !left || !right || left.getInt() != right.getInt() || !softcap ||
            !dropout || dropout.getValueAsDouble() != 0.0) {
          op->emitError("X86-E2E-1 attention requires f32 MHA, symmetric windows, and dropout_p=0");
          return signalPassFailure();
        }
        Value bh = builder.create<arith::MulIOp>(
            loc, op->getOperand(pointerCount), op->getOperand(pointerCount + 1));
        Value scaleValue = builder.create<arith::ConstantFloatOp>(
            loc, cast<FloatType>(f32Ty),
            APFloat(float(scale.getValueAsDouble())));
        Value causalValue = builder.create<arith::ConstantIntOp>(
            loc, causal.getValue() ? 1 : 0, 32);
        bool extended = hasBias || left.getInt() >= 0 ||
                        softcap.getValueAsDouble() > 0.0;
        if (!extended) {
          StringRef symbol = "tessera_x86_flash_attn_f32";
          ensureExternalDecl(
              module, symbol,
              FunctionType::get(ctx, {ptrTy, ptrTy, ptrTy, i64Ty, i64Ty,
                                      i64Ty, i64Ty, i64Ty, f32Ty, i32Ty, ptrTy},
                                {}));
          builder.create<func::CallOp>(
              loc, symbol, TypeRange{},
              ValueRange{op->getOperand(0), op->getOperand(1),
                         op->getOperand(2), bh, op->getOperand(pointerCount + 3),
                         op->getOperand(pointerCount + 4),
                         op->getOperand(pointerCount + 5),
                         op->getOperand(pointerCount + 6), scaleValue,
                         causalValue, op->getOperand(pointerCount - 1)});
        } else {
          StringRef symbol = "tessera_x86_flash_attn_ext_f32";
          ensureExternalDecl(
              module, symbol,
              FunctionType::get(ctx, {ptrTy, ptrTy, ptrTy, ptrTy, i64Ty,
                                      i64Ty, i64Ty, i64Ty, i64Ty, i64Ty,
                                      f32Ty, i32Ty, i64Ty, f32Ty, ptrTy}, {}));
          Value biasPointer = hasBias
              ? op->getOperand(3)
              : builder.create<LLVM::ZeroOp>(loc, ptrTy).getResult();
          Value biasStride = hasBias
              ? builder.create<arith::MulIOp>(
                    loc, op->getOperand(pointerCount + 3),
                    op->getOperand(pointerCount + 4)).getResult()
              : builder.create<arith::ConstantIntOp>(loc, 0, 64).getResult();
          Value window = builder.create<arith::ConstantIntOp>(
              loc, left.getInt() < 0 ? 0 : left.getInt(), 64);
          Value softcapValue = builder.create<arith::ConstantFloatOp>(
              loc, cast<FloatType>(f32Ty),
              APFloat(float(softcap.getValueAsDouble())));
          builder.create<func::CallOp>(
              loc, symbol, TypeRange{},
              ValueRange{op->getOperand(0), op->getOperand(1),
                         op->getOperand(2), biasPointer, biasStride, bh,
                         op->getOperand(pointerCount + 3),
                         op->getOperand(pointerCount + 4),
                         op->getOperand(pointerCount + 5),
                         op->getOperand(pointerCount + 6), scaleValue,
                         causalValue, window, softcapValue,
                         op->getOperand(pointerCount - 1)});
        }
        op->erase();
        continue;
      }

      auto storage = op->getAttrOfType<StringAttr>("storage");
      auto accum = op->getAttrOfType<StringAttr>("accum");
      auto kind = op->getAttrOfType<StringAttr>("kind");
      auto innerIsOne = op->getAttrOfType<BoolAttr>("inner_is_one");
      if (op->getNumOperands() != 5 || !storage || storage.getValue() != "f32" ||
          !accum || accum.getValue() != "f32" || !kind || !innerIsOne ||
          !innerIsOne.getValue()) {
        op->emitError("X86-E2E-1 reduction requires f32 storage/accumulation, last-axis inner_is_one=true, and X/O/Outer/AxisExtent/Inner");
        return signalPassFailure();
      }
      int32_t kindValue;
      if (kind.getValue() == "sum") kindValue = 0;
      else if (kind.getValue() == "max") kindValue = 1;
      else if (kind.getValue() == "mean") kindValue = 2;
      else {
        op->emitError("X86-E2E-1 reduction kind must be sum|max|mean");
        return signalPassFailure();
      }
      StringRef symbol = "tessera_x86_avx512_reduce_f32";
      ensureExternalDecl(module, symbol,
                         FunctionType::get(ctx, {ptrTy, i64Ty, i64Ty, ptrTy, i32Ty}, {}));
      Value kindConstant = builder.create<arith::ConstantIntOp>(loc, kindValue, 32);
      builder.create<func::CallOp>(
          loc, symbol, TypeRange{},
          ValueRange{op->getOperand(0), op->getOperand(2),
                     op->getOperand(3), op->getOperand(1), kindConstant});
      op->erase();
    }
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createTileToX86Pass() {
  return std::make_unique<TileToX86PassImpl>();
}
} // namespace tessera
