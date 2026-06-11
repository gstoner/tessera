//===- MatmulSoftmaxMatmulFusionToAppleGPU.cpp ---------------------------===//
//
// Phase 8.4.5 — Apple GPU 3-op MSL fusion: full attention block as a
// single kernel.
//
// Pattern matches a 3-op SSA chain:
//
//   %m1 = tessera.matmul %A, %B          : (M, K) x (K, N) -> (M, N)
//   %p  = tessera.softmax %m1            : (M, N) -> (M, N)   axis=-1
//   %o  = tessera.matmul %p, %C          : (M, N) x (N, P) -> (M, P)
//
// and replaces the trio with a single func.call into the Apple-GPU runtime
// shim:
//
//   tessera_apple_gpu_matmul_softmax_matmul_{f32,f16,bf16}(A, B, C, O,
//                                                          M, K, N, P)
//
// Constraints:
//   - rank-2 inputs/output throughout
//   - matching element types across A, B, C, output (f32, f16, or bf16)
//   - static shapes
//   - softmax axis must be -1 (default)
//   - N <= 256 AND P <= 256 (per-thread stack arrays in the GPU kernel)
//   - first matmul's result has exactly one use (the softmax)
//   - softmax's result has exactly one use (the second matmul)
//
// The pass runs *before* the 2-op fusion so the 3-op chain wins. If any
// constraint fails, the chain falls back to the 2-op fusion (matmul ->
// softmax) plus a standalone matmul, or all the way to per-op runtime
// calls / artifact-only as appropriate.
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

constexpr llvm::StringLiteral kMSMF32Symbol =
    "tessera_apple_gpu_matmul_softmax_matmul_f32";
constexpr llvm::StringLiteral kMSMF16Symbol =
    "tessera_apple_gpu_matmul_softmax_matmul_f16";
constexpr llvm::StringLiteral kMSMBF16Symbol =
    "tessera_apple_gpu_matmul_softmax_matmul_bf16";



// We rewrite at the second matmul (the chain tail). From there we walk
// upward: matmul.operand0 must come from softmax with single use; the
// softmax's operand0 must come from the first matmul with single use.
// If matched, all three ops are replaced atomically.
struct LowerMatmulSoftmaxMatmulFusionToAppleGPU : public RewritePattern {
  LowerMatmulSoftmaxMatmulFusionToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.matmul", /*benefit=*/3, ctx) {}

  LogicalResult matchAndRewrite(Operation *secondMatmulOp,
                                PatternRewriter &rewriter) const override {
    if (secondMatmulOp->getNumOperands() < 2) return failure();
    Value secondLhs = secondMatmulOp->getOperand(0);  // softmax output
    Value secondRhs = secondMatmulOp->getOperand(1);  // C

    // Decision #19 — consume the compiler's fusion descriptor. When the
    // canonical compile recognized this chain it stamps the tail op with
    // `tessera.fusion.intent = "matmul_softmax_matmul"`; we take that as
    // authoritative (the emitted call is tagged `source = "descriptor"`),
    // rather than re-discovering the fusion purely structurally. Absent the
    // intent, the structural walk below still recognizes the chain and the
    // call is tagged `source = "rediscovered"` (back-compat).
    StringRef intent;
    if (auto a = secondMatmulOp->getAttrOfType<StringAttr>("tessera.fusion.intent"))
      intent = a.getValue();
    bool descriptorDriven = (intent == "matmul_softmax_matmul");

    // Walk: secondLhs must be the result of a tessera.softmax with one use.
    Operation *softmaxOp = secondLhs.getDefiningOp();
    if (!softmaxOp)
      return rewriter.notifyMatchFailure(secondMatmulOp, "fusion3: tail matmul lhs has no defining op");
    if (softmaxOp->getName().getStringRef() != "tessera.softmax") {
      // Decision #21 — a fusion intent that the IR structure contradicts is a
      // real inconsistency; surface it by name instead of silently falling back.
      if (descriptorDriven)
        secondMatmulOp->emitWarning(
            "tessera.fusion.intent = \"matmul_softmax_matmul\" but tail matmul "
            "lhs is not a tessera.softmax — descriptor/IR mismatch; falling back");
      return rewriter.notifyMatchFailure(secondMatmulOp, "fusion3: tail matmul lhs is not a softmax");
    }
    if (!secondLhs.hasOneUse())
      return rewriter.notifyMatchFailure(secondMatmulOp, "fusion3: softmax result has multiple uses");

    // axis: defaults to -1.
    int64_t axis = -1;
    if (auto attr = softmaxOp->getAttrOfType<IntegerAttr>("axis"))
      axis = attr.getInt();

    if (softmaxOp->getNumOperands() < 1) return failure();
    Value softmaxIn = softmaxOp->getOperand(0);

    // Walk: softmax input must be the result of a tessera.matmul with one use.
    Operation *firstMatmulOp = softmaxIn.getDefiningOp();
    if (!firstMatmulOp)
      return rewriter.notifyMatchFailure(secondMatmulOp, "fusion3: softmax input has no defining op");
    if (firstMatmulOp->getName().getStringRef() != "tessera.matmul")
      return rewriter.notifyMatchFailure(secondMatmulOp, "fusion3: softmax input is not from matmul");
    if (!softmaxIn.hasOneUse())
      return rewriter.notifyMatchFailure(secondMatmulOp, "fusion3: first matmul result has multiple uses");

    if (firstMatmulOp->getNumOperands() < 2) return failure();
    Value lhsA = firstMatmulOp->getOperand(0);
    Value lhsB = firstMatmulOp->getOperand(1);

    // Type checks. All four operands must share the same element type.
    auto aTy = dyn_cast<RankedTensorType>(lhsA.getType());
    auto bTy = dyn_cast<RankedTensorType>(lhsB.getType());
    auto cTy = dyn_cast<RankedTensorType>(secondRhs.getType());
    auto smTy = dyn_cast<RankedTensorType>(softmaxIn.getType());
    if (!aTy || !bTy || !cTy || !smTy)
      return rewriter.notifyMatchFailure(secondMatmulOp, "fusion3: non-ranked tensor operand");
    if (aTy.getRank() != 2 || bTy.getRank() != 2 || cTy.getRank() != 2)
      return rewriter.notifyMatchFailure(secondMatmulOp, "fusion3: rank-2 only");

    Type elem = aTy.getElementType();
    if (elem != bTy.getElementType() || elem != cTy.getElementType() ||
        elem != smTy.getElementType())
      return rewriter.notifyMatchFailure(
          secondMatmulOp, "fusion3: matmul/softmax/matmul element types must all match");

    StringRef symbol;
    if (elem.isF32()) {
      symbol = kMSMF32Symbol;
    } else if (elem.isF16()) {
      symbol = kMSMF16Symbol;
    } else if (elem.isBF16()) {
      symbol = kMSMBF16Symbol;
    } else {
      return rewriter.notifyMatchFailure(
          secondMatmulOp, "fusion3: only f32/f16/bf16 in Phase 8.4.5");
    }
    if (axis != -1 && axis != smTy.getRank() - 1)
      return rewriter.notifyMatchFailure(secondMatmulOp, "fusion3: axis must be -1");

    // Static shapes only.
    if (aTy.isDynamicDim(0) || aTy.isDynamicDim(1) ||
        bTy.isDynamicDim(0) || bTy.isDynamicDim(1) ||
        cTy.isDynamicDim(0) || cTy.isDynamicDim(1))
      return rewriter.notifyMatchFailure(secondMatmulOp, "fusion3: requires static shapes");

    int64_t M = aTy.getDimSize(0);
    int64_t K = aTy.getDimSize(1);
    int64_t N = bTy.getDimSize(1);
    int64_t P = cTy.getDimSize(1);
    if (bTy.getDimSize(0) != K)
      return rewriter.notifyMatchFailure(secondMatmulOp, "fusion3: A/B K mismatch");
    if (cTy.getDimSize(0) != N)
      return rewriter.notifyMatchFailure(secondMatmulOp, "fusion3: probs/C N mismatch");
    if (N > 256 || P > 256)
      return rewriter.notifyMatchFailure(
          secondMatmulOp, "fusion3: GPU kernel limited to N <= 256 and P <= 256");

    Location loc = secondMatmulOp->getLoc();
    ModuleOp mod = secondMatmulOp->getParentOfType<ModuleOp>();
    MLIRContext *ctx = secondMatmulOp->getContext();

    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();

    auto aMemTy = MemRefType::get({M, K}, elem);
    auto bMemTy = MemRefType::get({K, N}, elem);
    auto cMemTy = MemRefType::get({N, P}, elem);
    auto oMemTy = MemRefType::get({M, P}, elem);

    rewriter.setInsertionPoint(firstMatmulOp);
    Value aPtr = extractPtr(rewriter, loc, lhsA, aMemTy);
    Value bPtr = extractPtr(rewriter, loc, lhsB, bMemTy);
    Value cPtr = extractPtr(rewriter, loc, secondRhs, cMemTy);
    auto oAlloc = rewriter.create<memref::AllocOp>(loc, oMemTy);
    Value oPtr;
    {
      auto pi =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, oAlloc);
      oPtr = rewriter.create<arith::IndexCastOp>(loc, i64Ty, pi);
    }

    Value Mv = rewriter.create<arith::ConstantIntOp>(loc, M, 32);
    Value Kv = rewriter.create<arith::ConstantIntOp>(loc, K, 32);
    Value Nv = rewriter.create<arith::ConstantIntOp>(loc, N, 32);
    Value Pv = rewriter.create<arith::ConstantIntOp>(loc, P, 32);

    FunctionType fnTy = FunctionType::get(
        ctx, {i64Ty, i64Ty, i64Ty, i64Ty, i32Ty, i32Ty, i32Ty, i32Ty}, {});
    ensureExternalDecl(mod, symbol, fnTy);

    auto callOp = rewriter.create<func::CallOp>(
        loc, symbol, TypeRange{},
        ValueRange{aPtr, bPtr, cPtr, oPtr, Mv, Kv, Nv, Pv});
    // Decision #19 — emit the fusion descriptor into the Target IR so the
    // fusion decision is first-class/auditable (which kernel, and whether the
    // compiler's intent drove it or it was re-discovered structurally).
    callOp->setAttr("tessera.fusion.kernel",
                    rewriter.getStringAttr("matmul_softmax_matmul"));
    callOp->setAttr("tessera.fusion.source",
                    rewriter.getStringAttr(descriptorDriven ? "descriptor"
                                                            : "rediscovered"));

    auto outTensorTy = RankedTensorType::get({M, P}, elem);
    Value result =
        rewriter.create<bufferization::ToTensorOp>(loc, outTensorTy, oAlloc);

    // Replace the chain tail with the fused result, then erase the
    // (now-dead) softmax and first matmul.
    rewriter.replaceOp(secondMatmulOp, result);
    rewriter.eraseOp(softmaxOp);
    rewriter.eraseOp(firstMatmulOp);
    return success();
  }
};

struct LowerMatmulSoftmaxMatmulFusionToAppleGPUPass
    : public PassWrapper<LowerMatmulSoftmaxMatmulFusionToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMatmulSoftmaxMatmulFusionToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-matmul-softmax-matmul-fusion-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Fuse tessera.matmul -> tessera.softmax -> tessera.matmul "
           "(rank-2, f32/f16/bf16, axis=-1, N <= 256, P <= 256) into a single "
           "Apple GPU runtime call (custom MSL kernel) — full attention block";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerMatmulSoftmaxMatmulFusionToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerMatmulSoftmaxMatmulFusionToAppleGPUPass() {
  return std::make_unique<LowerMatmulSoftmaxMatmulFusionToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
