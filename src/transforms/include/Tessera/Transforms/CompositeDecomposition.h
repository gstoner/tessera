//===- CompositeDecomposition.h - Graph composite -> canonical op rewrites -===//
//
// ODS triage WIRE slice 1 (GOV-ODS-CONSUMER-1, 2026-09-27). Two Graph IR ops
// the `@jit` frontend emits have no lowering of their own because they are a
// canonical op in disguise:
//
//   tessera.target_verify(tokens, logits)       -> tessera.softmax(logits){axis = rank-1}
//   tessera.ntk_rope(x, theta){scale = s}       -> tessera.rope(x, theta / s)
//
// `target_verify`'s reference is `softmax(logits, -1)` over S = len(tokens)
// positions; its verifier already pins `tokens` to S and `logits` /
// `target_probs` to S x V f32, so once verified the `tokens` operand carries no
// further semantics and is dropped here (its only content, the S pin, was
// checked before the rewrite -- Decision #32's named reason).
//
// `ntk_rope`'s reference is literally `rope(x, theta / scale)`
// (nn/functional.py). `tessera.div` is tensor x tensor, so the rewrite
// materializes a splat `arith.constant` of theta's statically shaped floating
// type. A `scale` of exactly 1.0 rewrites to `rope(x, theta)` with no division.
// Any other theta (unranked, dynamic, non-float) is refused with
// TESSERA_NTK_ROPE_THETA_UNREWRITABLE rather than left for no consumer to see
// (Decisions #21 / #21a).
//
// ONE pattern source, several routes (Decision #31: one authority, registered
// wherever a consumer of the canonical op lives). The patterns match by op
// name and build by OperationState, so they need only MLIR core + arith and can
// be linked into the Apple backend library, which does not link the Tessera
// dialect or TesseraPasses. Routes (see the ODS triage row for the reasons):
//   * tessera-canonicalize (CanonicalizeTesseraIR.cpp) -> every pipeline that
//     calls addGraphIRPreLoweringPasses: tessera-lower-to-x86, -gpu and the
//     tessera-lower-to-nvidia-sm{90,100,120} pipelines;
//   * tessera-lower-to-apple_gpu-runtime, first, ahead of the softmax / rope
//     runtime lowerings and every fusion that claims a softmax;
//   * tessera-lower-to-apple_{cpu,gpu}-full, in the reasoning prologue;
//   * libtessera_jit's stage-1a pipeline, ahead of tessera-to-linalg (the CPU
//     MLIR JIT lane, where target_verify's numeric check runs).
// ROCm is not a route: its pipelines consume Tile IR / directive carriers and
// have no Graph softmax or rope consumer to feed.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TRANSFORMS_COMPOSITEDECOMPOSITION_H
#define TESSERA_TRANSFORMS_COMPOSITEDECOMPOSITION_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace tessera {
namespace composite {

inline constexpr llvm::StringLiteral kTargetVerify = "tessera.target_verify";
inline constexpr llvm::StringLiteral kNTKRope = "tessera.ntk_rope";

/// Copy every attribute of `from` except `skip` onto `state`. Graph composite
/// ops carry no inherent semantic attribute other than the one the rewrite
/// consumes, so everything else (effects, provenance, layout markers) rides
/// forward rather than vanishing (Decision #32).
inline void forwardAttrs(mlir::Operation *from, mlir::OperationState &state,
                         llvm::StringRef skip = {}) {
  for (mlir::NamedAttribute attr : from->getAttrs())
    if (skip.empty() || attr.getName().strref() != skip)
      state.addAttribute(attr.getName(), attr.getValue());
}

/// target_verify(tokens, logits) -> softmax(logits){axis = rank-1}.
struct DecomposeTargetVerify : public mlir::RewritePattern {
  explicit DecomposeTargetVerify(mlir::MLIRContext *ctx)
      : mlir::RewritePattern(kTargetVerify, /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "target_verify arity");
    auto logitsTy =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
    auto resultTy =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    // The verifier makes both S x V f32; a softmax result must keep the
    // operand's element type, so anything else is not this decomposition.
    if (!logitsTy || !resultTy || logitsTy.getRank() < 1 ||
        logitsTy.getElementType() != resultTy.getElementType() ||
        logitsTy.getRank() != resultTy.getRank())
      return rewriter.notifyMatchFailure(op, "logits/target_probs mismatch");
    mlir::OperationState state(op->getLoc(), "tessera.softmax");
    state.addOperands(op->getOperand(1));
    state.addTypes(resultTy);
    forwardAttrs(op, state);
    state.addAttribute("axis", rewriter.getI64IntegerAttr(logitsTy.getRank() - 1));
    mlir::Operation *softmax = rewriter.create(state);
    rewriter.replaceOp(op, softmax->getResults());
    return mlir::success();
  }
};

/// Why ntk_rope's theta cannot take the scale division, or empty if it can.
inline llvm::StringRef ntkRopeThetaRefusal(mlir::Operation *op) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return "expects (x, theta) -> y";
  auto thetaTy =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
  if (!thetaTy)
    return "theta must be a ranked tensor to materialize the scale splat";
  if (!thetaTy.hasStaticShape())
    return "theta must be statically shaped to materialize the scale splat";
  if (!mlir::isa<mlir::FloatType>(thetaTy.getElementType()))
    return "theta must be a floating tensor to divide by scale";
  return {};
}

inline double ntkRopeScale(mlir::Operation *op) {
  if (auto scale = op->getAttrOfType<mlir::FloatAttr>("scale"))
    return scale.getValueAsDouble();
  return 1.0;  // the ODS DefaultValuedAttr
}

/// ntk_rope(x, theta){scale = s} -> rope(x, div(theta, splat(s))).
struct RewriteNTKRopeToRope : public mlir::RewritePattern {
  explicit RewriteNTKRopeToRope(mlir::MLIRContext *ctx)
      : mlir::RewritePattern(kNTKRope, /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "ntk_rope arity");
    double scale = ntkRopeScale(op);
    if (!(scale > 0.0))
      return rewriter.notifyMatchFailure(op, "scale must be positive");
    mlir::Value theta = op->getOperand(1);
    if (scale != 1.0) {
      if (!ntkRopeThetaRefusal(op).empty())
        return rewriter.notifyMatchFailure(op, ntkRopeThetaRefusal(op));
      auto thetaTy = mlir::cast<mlir::RankedTensorType>(theta.getType());
      auto elemTy = mlir::cast<mlir::FloatType>(thetaTy.getElementType());
      auto splat = mlir::DenseElementsAttr::get(
          thetaTy, rewriter.getFloatAttr(elemTy, scale));
      mlir::Value divisor =
          rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), thetaTy, splat);
      mlir::OperationState div(op->getLoc(), "tessera.div");
      div.addOperands({theta, divisor});
      div.addTypes(thetaTy);
      theta = rewriter.create(div)->getResult(0);
    }
    mlir::OperationState rope(op->getLoc(), "tessera.rope");
    rope.addOperands({op->getOperand(0), theta});
    rope.addTypes(op->getResult(0).getType());
    forwardAttrs(op, rope, /*skip=*/"scale");
    mlir::Operation *ropeOp = rewriter.create(rope);
    rewriter.replaceOp(op, ropeOp->getResults());
    return mlir::success();
  }
};

inline void populateCompositeDecompositionPatterns(
    mlir::RewritePatternSet &patterns) {
  patterns.add<DecomposeTargetVerify, RewriteNTKRopeToRope>(
      patterns.getContext());
}

/// Fail closed on any composite the patterns could not rewrite: after the
/// greedy driver, a surviving target_verify / ntk_rope has no consumer on any
/// route that runs this, so it is an error naming the op (Decision #21), not a
/// silent no-op.
inline mlir::LogicalResult verifyNoResidualComposites(mlir::Operation *root) {
  bool failed = false;
  root->walk([&](mlir::Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (name == kNTKRope) {
      llvm::StringRef why = ntkRopeThetaRefusal(op);
      op->emitError() << "TESSERA_NTK_ROPE_THETA_UNREWRITABLE: "
                      << "tessera.ntk_rope with scale = " << ntkRopeScale(op)
                      << " cannot be rewritten to tessera.rope(x, theta / scale): "
                      << (why.empty() ? llvm::StringRef("no rewrite applied")
                                      : why);
      failed = true;
    } else if (name == kTargetVerify) {
      op->emitError() << "tessera.target_verify could not be decomposed to "
                         "tessera.softmax (logits and target_probs must be "
                         "ranked tensors of one element type)";
      failed = true;
    }
  });
  return mlir::failure(failed);
}

/// Standalone form of the rewrite for routes that do not run
/// tessera-canonicalize (Apple -runtime / -full, the CPU MLIR JIT lane).
struct DecomposeCompositeOpsPass
    : public mlir::PassWrapper<DecomposeCompositeOpsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeCompositeOpsPass)

  llvm::StringRef getArgument() const override {
    return "tessera-decompose-composite-ops";
  }
  llvm::StringRef getDescription() const override {
    return "Rewrite tessera.target_verify to tessera.softmax and "
           "tessera.ntk_rope to tessera.rope(x, theta / scale); fail closed "
           "on a composite that cannot be rewritten";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
  }
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    populateCompositeDecompositionPatterns(patterns);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))) ||
        mlir::failed(verifyNoResidualComposites(getOperation())))
      signalPassFailure();
  }
};

} // namespace composite
} // namespace tessera

#endif // TESSERA_TRANSFORMS_COMPOSITEDECOMPOSITION_H
