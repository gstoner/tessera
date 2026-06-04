#include "Tessera/IR/TesseraOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <algorithm>

using namespace mlir;

#include "TesseraOpsEnums.cpp.inc"

namespace tessera {

LogicalResult MatmulOp::verify() {
  auto lhsType = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(getRhs().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!lhsType || !rhsType || !resultType)
    return success();
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      resultType.getRank() != 2)
    return emitOpError("expects rank-2 lhs, rhs, and result tensors");

  int64_t lhsK = getTransposeA() ? lhsType.getDimSize(0) : lhsType.getDimSize(1);
  int64_t rhsK = getTransposeB() ? rhsType.getDimSize(1) : rhsType.getDimSize(0);
  if (!ShapedType::isDynamic(lhsK) && !ShapedType::isDynamic(rhsK) &&
      lhsK != rhsK)
    return emitOpError("contracting dimensions must match");

  // Result shape must be (M, N): M from lhs's non-contracting dim, N from rhs's
  // non-contracting dim (honoring transposeA/transposeB). Without this, a
  // malformed (4x8)@(8x16)->(5x5) would pass rank+K and could be lowered to an
  // executable value call that silently produces a wrong-shaped output.
  int64_t m = getTransposeA() ? lhsType.getDimSize(1) : lhsType.getDimSize(0);
  int64_t n = getTransposeB() ? rhsType.getDimSize(0) : rhsType.getDimSize(1);
  int64_t rm = resultType.getDimSize(0);
  int64_t rn = resultType.getDimSize(1);
  if (!ShapedType::isDynamic(m) && !ShapedType::isDynamic(rm) && m != rm)
    return emitOpError("result row dimension must equal lhs M (")
           << m << " vs " << rm << ")";
  if (!ShapedType::isDynamic(n) && !ShapedType::isDynamic(rn) && n != rn)
    return emitOpError("result column dimension must equal rhs N (")
           << n << " vs " << rn << ")";
  return success();
}

LogicalResult BatchedGemmOp::verify() {
  // Apple Value Target IR sprint 6: rank-3 C[b] = A[b] @ B[b].
  //   A: B×M×K, B: B×K×N, result B×M×N. Batch and K must agree. No
  //   broadcasting and no transpose in this sprint — both are gated by the
  //   strict rank-3 contract (rank-4+ and rank-2 are rejected here).
  auto lhsType = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(getRhs().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!lhsType || !rhsType || !resultType)
    return success();
  if (lhsType.getRank() != 3 || rhsType.getRank() != 3 ||
      resultType.getRank() != 3)
    return emitOpError("expects rank-3 lhs, rhs, and result tensors "
                       "(no broadcasting; rank-2 / rank-4+ are gated)");

  auto agree = [](int64_t a, int64_t b) {
    return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
  };
  int64_t bl = lhsType.getDimSize(0), br = rhsType.getDimSize(0),
          bo = resultType.getDimSize(0);
  if (!agree(bl, br) || !agree(bl, bo))
    return emitOpError("batch dimensions must match across lhs, rhs, result");
  // No broadcasting: a batch of 1 against a larger batch is rejected by the
  // equality check above (1 != B is a mismatch unless both are 1).
  if (!agree(lhsType.getDimSize(2), rhsType.getDimSize(1)))
    return emitOpError("contracting dimensions must match (lhs K vs rhs K)");
  if (!agree(resultType.getDimSize(1), lhsType.getDimSize(1)))
    return emitOpError("result M must equal lhs M");
  if (!agree(resultType.getDimSize(2), rhsType.getDimSize(2)))
    return emitOpError("result N must equal rhs N");
  return success();
}

namespace {
bool isFloatTensor(RankedTensorType ty) {
  Type elem = ty.getElementType();
  return elem.isF32() || elem.isF16() || elem.isBF16() ||
         isa<FloatType>(elem);
}

LogicalResult verifyReductionResult(Operation *op, RankedTensorType input,
                                    RankedTensorType result,
                                    StringRef reduction) {
  auto agree = [](int64_t a, int64_t b) {
    return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
  };
  if (reduction == "none") {
    if (input.getRank() != result.getRank())
      return op->emitOpError("reduction=\"none\" result must match input rank");
    for (int64_t i = 0, e = input.getRank(); i < e; ++i)
      if (!agree(input.getDimSize(i), result.getDimSize(i)))
        return op->emitOpError("reduction=\"none\" result shape must match inputs");
    return success();
  }
  if (reduction == "mean" || reduction == "sum") {
    if (result.getRank() != 0)
      return op->emitOpError("mean/sum reduction result must be rank-0 tensor");
    return success();
  }
  return op->emitOpError("reduction must be one of \"none\", \"mean\", \"sum\"");
}

LogicalResult verifySameRankedShape(Operation *op, RankedTensorType a,
                                    RankedTensorType b,
                                    StringRef label) {
  auto agree = [](int64_t x, int64_t y) {
    return ShapedType::isDynamic(x) || ShapedType::isDynamic(y) || x == y;
  };
  if (a.getRank() != b.getRank())
    return op->emitOpError() << label << " ranks must match";
  for (int64_t i = 0, e = a.getRank(); i < e; ++i)
    if (!agree(a.getDimSize(i), b.getDimSize(i)))
      return op->emitOpError() << label << " shapes must match";
  return success();
}

// Helper: static dim equality (treats dynamic as "matches anything").
bool dimsAgree(int64_t a, int64_t b) {
  return mlir::ShapedType::isDynamic(a) || mlir::ShapedType::isDynamic(b) ||
         a == b;
}

StringRef reductionOrMean(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>("reduction"))
    return attr.getValue();
  return "mean";
}

double f64AttrOr(Operation *op, StringRef name, double fallback) {
  if (auto attr = op->getAttrOfType<FloatAttr>(name))
    return attr.getValueAsDouble();
  return fallback;
}
} // namespace

LogicalResult RLNormalizeGroupAdvantagesOp::verify() {
  auto rewards = dyn_cast<RankedTensorType>(getRewards().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!rewards || !result)
    return success();
  if (!isFloatTensor(rewards) || rewards.getElementType() != result.getElementType())
    return emitOpError("expects floating rewards/result with matching dtype");
  if (failed(verifySameRankedShape(getOperation(), rewards, result,
                                   "normalize_group_advantages")))
    return failure();
  int64_t axis = getGroupAxis();
  if (axis < 0 || axis >= rewards.getRank())
    return emitOpError("group_axis must be within the rewards rank");
  if (f64AttrOr(getOperation(), "eps", 1.0e-8) <= 0.0)
    return emitOpError("eps must be positive");
  return success();
}

LogicalResult RLPPOPolicyLossOp::verify() {
  auto next = dyn_cast<RankedTensorType>(getLogpNew().getType());
  auto old = dyn_cast<RankedTensorType>(getLogpOld().getType());
  auto adv = dyn_cast<RankedTensorType>(getAdvantages().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!next || !old || !adv || !result)
    return success();
  if (!isFloatTensor(next) || next.getElementType() != old.getElementType() ||
      next.getElementType() != adv.getElementType() ||
      next.getElementType() != result.getElementType())
    return emitOpError("expects floating operands/result with matching dtype");
  if (failed(verifySameRankedShape(getOperation(), next, old,
                                   "ppo_policy_loss log-prob")) ||
      failed(verifySameRankedShape(getOperation(), next, adv,
                                   "ppo_policy_loss advantage")))
    return failure();
  auto verifyOptional = [&](Value v, StringRef label) -> LogicalResult {
    if (!v)
      return success();
    auto ty = dyn_cast<RankedTensorType>(v.getType());
    if (!ty)
      return success();
    if (ty.getElementType() != next.getElementType())
      return emitOpError() << label << " dtype must match log-prob dtype";
    std::string context = "ppo_policy_loss ";
    context += label.str();
    return verifySameRankedShape(getOperation(), next, ty, context);
  };
  if (failed(verifyOptional(getMask(), "mask")) ||
      failed(verifyOptional(getRefLogp(), "ref_logp")) ||
      failed(verifyOptional(getEntropy(), "entropy")))
    return failure();
  if (f64AttrOr(getOperation(), "clip_epsilon", 0.2) <= 0.0)
    return emitOpError("clip_epsilon must be positive");
  if (f64AttrOr(getOperation(), "kl_coef", 0.0) < 0.0)
    return emitOpError("kl_coef must be non-negative");
  if (f64AttrOr(getOperation(), "entropy_coef", 0.0) < 0.0)
    return emitOpError("entropy_coef must be non-negative");
  if (f64AttrOr(getOperation(), "kl_coef", 0.0) != 0.0 && !getRefLogp())
    return emitOpError("kl_coef requires ref_logp operand");
  if (f64AttrOr(getOperation(), "entropy_coef", 0.0) != 0.0 && !getEntropy())
    return emitOpError("entropy_coef requires entropy operand");
  return verifyReductionResult(getOperation(), next, result,
                               reductionOrMean(getOperation()));
}

LogicalResult RLGRPOPolicyLossOp::verify() {
  auto next = dyn_cast<RankedTensorType>(getLogpNew().getType());
  auto old = dyn_cast<RankedTensorType>(getLogpOld().getType());
  auto rewards = dyn_cast<RankedTensorType>(getRewards().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!next || !old || !rewards || !result)
    return success();
  if (!isFloatTensor(next) || next.getElementType() != old.getElementType() ||
      next.getElementType() != rewards.getElementType() ||
      next.getElementType() != result.getElementType())
    return emitOpError("expects floating operands/result with matching dtype");
  if (failed(verifySameRankedShape(getOperation(), next, old,
                                   "grpo_policy_loss log-prob")) ||
      failed(verifySameRankedShape(getOperation(), next, rewards,
                                   "grpo_policy_loss rewards")))
    return failure();
  int64_t axis = getGroupAxis();
  if (axis < 0 || axis >= next.getRank())
    return emitOpError("group_axis must be within the operand rank");
  if (f64AttrOr(getOperation(), "clip_epsilon", 0.2) <= 0.0)
    return emitOpError("clip_epsilon must be positive");
  if (f64AttrOr(getOperation(), "kl_coef", 0.0) < 0.0)
    return emitOpError("kl_coef must be non-negative");
  return verifyReductionResult(getOperation(), next, result,
                               reductionOrMean(getOperation()));
}

LogicalResult RLCISPOPolicyLossOp::verify() {
  auto next = dyn_cast<RankedTensorType>(getLogpNew().getType());
  auto old = dyn_cast<RankedTensorType>(getLogpOld().getType());
  auto rewards = dyn_cast<RankedTensorType>(getRewards().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!next || !old || !rewards || !result)
    return success();
  if (!isFloatTensor(next) || next.getElementType() != old.getElementType() ||
      next.getElementType() != rewards.getElementType() ||
      next.getElementType() != result.getElementType())
    return emitOpError("expects floating operands/result with matching dtype");
  if (failed(verifySameRankedShape(getOperation(), next, old,
                                   "cispo_policy_loss log-prob")) ||
      failed(verifySameRankedShape(getOperation(), next, rewards,
                                   "cispo_policy_loss rewards")))
    return failure();
  int64_t axis = getGroupAxis();
  if (axis < 0 || axis >= next.getRank())
    return emitOpError("group_axis must be within the operand rank");
  if (f64AttrOr(getOperation(), "epsilon_high", 5.0) <= 0.0)
    return emitOpError("epsilon_high must be positive");
  if (f64AttrOr(getOperation(), "kl_coef", 0.0) < 0.0)
    return emitOpError("kl_coef must be non-negative");
  return verifyReductionResult(getOperation(), next, result,
                               reductionOrMean(getOperation()));
}

LogicalResult EBMEnergyQuadraticOp::verify() {
  auto x = dyn_cast<RankedTensorType>(getX().getType());
  auto y = dyn_cast<RankedTensorType>(getY().getType());
  auto energies = dyn_cast<RankedTensorType>(getEnergies().getType());
  if (!x || !y || !energies)
    return success();
  if (!isFloatTensor(x) || x.getElementType() != y.getElementType() ||
      x.getElementType() != energies.getElementType())
    return emitOpError("expects floating x/y/energies with matching dtype");
  if (x.getRank() != 2 || y.getRank() != 2)
    return emitOpError("expects rank-2 x and y tensors");
  if (energies.getRank() != 1)
    return emitOpError("energies result must be rank-1");
  if (failed(verifySameRankedShape(getOperation(), x, y,
                                   "ebm.energy_quadratic x/y")))
    return failure();
  if (!dimsAgree(x.getDimSize(0), energies.getDimSize(0)))
    return emitOpError("energies length must equal batch dimension");
  return success();
}

LogicalResult EBMLangevinStepOp::verify() {
  auto y = dyn_cast<RankedTensorType>(getY().getType());
  auto grad = dyn_cast<RankedTensorType>(getGrad().getType());
  auto noise = dyn_cast<RankedTensorType>(getNoise().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!y || !grad || !noise || !result)
    return success();
  if (!isFloatTensor(y) || y.getElementType() != grad.getElementType() ||
      y.getElementType() != noise.getElementType() ||
      y.getElementType() != result.getElementType())
    return emitOpError("expects floating y/grad/noise/result with matching dtype");
  if (failed(verifySameRankedShape(getOperation(), y, grad,
                                   "ebm.langevin_step grad")) ||
      failed(verifySameRankedShape(getOperation(), y, noise,
                                   "ebm.langevin_step noise")) ||
      failed(verifySameRankedShape(getOperation(), y, result,
                                   "ebm.langevin_step result")))
    return failure();
  if (f64AttrOr(getOperation(), "eta", 0.0) <= 0.0)
    return emitOpError("eta must be positive");
  if (f64AttrOr(getOperation(), "noise_scale", 0.0) < 0.0)
    return emitOpError("noise_scale must be non-negative");
  return success();
}

LogicalResult CholeskyOp::verify() {
  auto aType = dyn_cast<RankedTensorType>(getA().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!aType || !resultType)
    return success();
  // L-series pilot: rank-2 square SPD input; result matches input shape.
  // (Batched rank-3 is a follow-on once the rank-2 spine is proven.)
  if (aType.getRank() != 2 || resultType.getRank() != 2)
    return emitOpError("expects rank-2 input and result tensors");
  int64_t m = aType.getDimSize(0);
  int64_t n = aType.getDimSize(1);
  if (!ShapedType::isDynamic(m) && !ShapedType::isDynamic(n) && m != n)
    return emitOpError("input matrix must be square");
  int64_t rm = resultType.getDimSize(0);
  int64_t rn = resultType.getDimSize(1);
  if (!ShapedType::isDynamic(m) && !ShapedType::isDynamic(rm) && m != rm)
    return emitOpError("result must have the same shape as the input");
  if (!ShapedType::isDynamic(n) && !ShapedType::isDynamic(rn) && n != rn)
    return emitOpError("result must have the same shape as the input");
  return success();
}

// L-series linalg family verifiers (2026-06-02). Pilot-level: rank + the core
// shape relations, all dynamic-dim-aware (skip checks on dynamic dims).

LogicalResult TriSolveOp::verify() {
  auto a = dyn_cast<RankedTensorType>(getA().getType());
  auto b = dyn_cast<RankedTensorType>(getB().getType());
  auto x = dyn_cast<RankedTensorType>(getResult().getType());
  if (!a || !b || !x)
    return success();
  if (a.getRank() != 2 || b.getRank() != 2 || x.getRank() != 2)
    return emitOpError("expects rank-2 A, B, and result tensors");
  if (!dimsAgree(a.getDimSize(0), a.getDimSize(1)))
    return emitOpError("A must be square");
  if (!dimsAgree(a.getDimSize(0), b.getDimSize(0)))
    return emitOpError("A rows must match B rows (op(A) X = B)");
  if (!dimsAgree(b.getDimSize(0), x.getDimSize(0)) ||
      !dimsAgree(b.getDimSize(1), x.getDimSize(1)))
    return emitOpError("result must have the same shape as B");
  return success();
}

LogicalResult CholeskySolveOp::verify() {
  auto a = dyn_cast<RankedTensorType>(getA().getType());
  auto b = dyn_cast<RankedTensorType>(getB().getType());
  auto x = dyn_cast<RankedTensorType>(getResult().getType());
  if (!a || !b || !x)
    return success();
  if (a.getRank() != 2 || b.getRank() != 2 || x.getRank() != 2)
    return emitOpError("expects rank-2 A, B, and result tensors");
  if (!dimsAgree(a.getDimSize(0), a.getDimSize(1)))
    return emitOpError("A must be square (SPD)");
  if (!dimsAgree(a.getDimSize(0), b.getDimSize(0)))
    return emitOpError("A rows must match B rows (A X = B)");
  if (!dimsAgree(b.getDimSize(0), x.getDimSize(0)) ||
      !dimsAgree(b.getDimSize(1), x.getDimSize(1)))
    return emitOpError("result must have the same shape as B");
  return success();
}

LogicalResult LUOp::verify() {
  auto a = dyn_cast<RankedTensorType>(getA().getType());
  auto lu = dyn_cast<RankedTensorType>(getLu().getType());
  auto piv = dyn_cast<RankedTensorType>(getPivots().getType());
  if (!a || !lu || !piv)
    return success();
  if (a.getRank() != 2 || lu.getRank() != 2)
    return emitOpError("expects rank-2 A and packed-LU tensors");
  if (!dimsAgree(a.getDimSize(0), a.getDimSize(1)))
    return emitOpError("A must be square");
  if (!dimsAgree(a.getDimSize(0), lu.getDimSize(0)) ||
      !dimsAgree(a.getDimSize(1), lu.getDimSize(1)))
    return emitOpError("packed LU must have the same shape as A");
  if (piv.getRank() != 1)
    return emitOpError("pivots must be a rank-1 tensor");
  if (!dimsAgree(a.getDimSize(0), piv.getDimSize(0)))
    return emitOpError("pivots length must equal the matrix order");
  return success();
}

LogicalResult QROp::verify() {
  auto a = dyn_cast<RankedTensorType>(getA().getType());
  auto q = dyn_cast<RankedTensorType>(getQ().getType());
  auto r = dyn_cast<RankedTensorType>(getR().getType());
  if (!a || !q || !r)
    return success();
  if (a.getRank() != 2 || q.getRank() != 2 || r.getRank() != 2)
    return emitOpError("expects rank-2 A, Q, and R tensors");
  // Reduced QR (A is M×N, M≥N): Q is M×N with orthonormal columns, R is N×N
  // upper-triangular.  Enforce the full shape contract: Q rows = A rows; R is
  // square; R order = A columns; Q columns = R rows = A columns (review R4).
  int64_t aRows = a.getDimSize(0), aCols = a.getDimSize(1);
  int64_t qRows = q.getDimSize(0), qCols = q.getDimSize(1);
  int64_t rRows = r.getDimSize(0), rCols = r.getDimSize(1);
  if (!dimsAgree(aRows, qRows))
    return emitOpError("Q rows must match A rows");
  if (!dimsAgree(rRows, rCols))
    return emitOpError("R must be square");
  if (!dimsAgree(rCols, aCols))
    return emitOpError("R order must equal the number of columns of A");
  if (!dimsAgree(qCols, aCols))
    return emitOpError("Q columns must equal the number of columns of A "
                       "(reduced QR)");
  if (!dimsAgree(qCols, rRows))
    return emitOpError("Q columns must match R rows");
  return success();
}

LogicalResult SVDOp::verify() {
  auto a = dyn_cast<RankedTensorType>(getA().getType());
  auto u = dyn_cast<RankedTensorType>(getU().getType());
  auto s = dyn_cast<RankedTensorType>(getS().getType());
  auto v = dyn_cast<RankedTensorType>(getV().getType());
  if (!a || !u || !s || !v)
    return success();
  if (a.getRank() != 2 || u.getRank() != 2 || v.getRank() != 2)
    return emitOpError("expects rank-2 A, U, and V tensors");
  if (s.getRank() != 1)
    return emitOpError("singular values S must be a rank-1 tensor");
  // A = U diag(S) V.  U rows = A rows and V columns = A columns always hold
  // (review R4).
  int64_t aRows = a.getDimSize(0), aCols = a.getDimSize(1);
  int64_t uCols = u.getDimSize(1);
  int64_t vRows = v.getDimSize(0), vCols = v.getDimSize(1);
  int64_t sLen = s.getDimSize(0);
  if (!dimsAgree(aRows, u.getDimSize(0)))
    return emitOpError("U rows must match A rows");
  if (!dimsAgree(vCols, aCols))
    return emitOpError("V columns must match A columns");
  // The number of singular values is min(M, N) in BOTH reduced and full SVD
  // (review RV-P2a): checkable whenever M and N are static.
  if (!ShapedType::isDynamic(aRows) && !ShapedType::isDynamic(aCols) &&
      !ShapedType::isDynamic(sLen)) {
    int64_t k = std::min(aRows, aCols);
    if (sLen != k)
      return emitOpError("number of singular values S must equal min(M, N)");
  }
  if (!getFullMatrices()) {
    // Reduced SVD: U is M×K, S is K, V is K×N with K = min(M, N).  The inner
    // dimensions must agree (U cols = |S| = V rows); A = U·diag(S)·V then
    // type-checks as (M×K)(K×K)(K×N).
    if (!dimsAgree(uCols, sLen))
      return emitOpError("U columns must equal the number of singular values "
                         "(reduced SVD)");
    if (!dimsAgree(vRows, sLen))
      return emitOpError("V rows must equal the number of singular values "
                         "(reduced SVD)");
  } else {
    // Full SVD: U is the M×M left-orthogonal basis and V is the N×N
    // right-orthogonal basis (both square); S still has min(M, N) values but
    // that requires both dims static to check, so enforce the squareness
    // contract that always holds (review R4 follow-on, AV0).
    if (!dimsAgree(uCols, aRows))
      return emitOpError("U must be square (M×M) for full_matrices SVD");
    if (!dimsAgree(vRows, aCols))
      return emitOpError("V must be square (N×N) for full_matrices SVD");
  }
  return success();
}

LogicalResult Conv2DNHWCOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto filterType = dyn_cast<RankedTensorType>(getFilter().getType());
  if (!inputType || !filterType)
    return success();
  if (inputType.getRank() != 4 || filterType.getRank() != 4)
    return emitOpError("expects NHWC input and HWCF filter rank-4 tensors");
  return success();
}

// Sprint V3 (2026-05-22) — target-aware head_dim limits.
//
// Per-SM max head_dim for the fused flash-attention kernel.  Numbers
// come from `docs/apple_gpu_kernel_inventory.md` (Apple GPU MSL
// kernel ships head_dim ≤ 256) + Sprint G-2 NVIDIA kernel inventory.
// SM_80 / SM_86 / SM_89: 128 (legacy FA-2/3 limit).
// SM_90 / SM_100 / SM_120 + Apple Hopper-class: 256.
// CPU / Apple CPU: no SM tag ⇒ no limit applied here (CPU path runs
// the numpy reference, which has no fragment-size constraint).
static int64_t maxHeadDimForTargetSm(StringRef sm) {
  return llvm::StringSwitch<int64_t>(sm)
      .Case("sm_70", 128)
      .Case("sm_75", 128)
      .Case("sm_80", 128)
      .Case("sm_86", 128)
      .Case("sm_89", 128)
      .Case("sm_90", 256)
      .Case("sm_90a", 256)
      .Case("sm_100", 256)
      .Case("sm_100a", 256)
      .Case("sm_120", 256)
      .Case("sm_120a", 256)
      .Default(-1);  // unknown / no limit applied
}

LogicalResult FlashAttnOp::verify() {
  if (getHeadDim() <= 0)
    return emitOpError("head_dim must be positive");
  if (auto dropout = getDropoutP()) {
    double p = dropout->convertToDouble();
    if (p < 0.0 || p >= 1.0)
      return emitOpError("dropout_p must satisfy 0 <= p < 1");
  }
  // Sprint V3 (2026-05-22): target-aware head_dim ceiling.  When the
  // parent function carries ``tessera.target_sm = "sm_XX"`` (set by
  // ``DistributionLoweringPass`` from ``GPUTargetProfile`` or by the
  // string-target dispatcher), enforce the per-SM head_dim limit.
  // Functions without the attribute (CPU path) skip this check.
  Operation* parent = (*this)->getParentOp();
  while (parent && !parent->hasAttr("tessera.target_sm"))
    parent = parent->getParentOp();
  if (parent) {
    if (auto attr = dyn_cast<StringAttr>(parent->getAttr("tessera.target_sm"))) {
      int64_t limit = maxHeadDimForTargetSm(attr.getValue());
      if (limit > 0 && getHeadDim() > limit)
        return emitOpError("head_dim=")
               << getHeadDim() << " exceeds the SM "
               << attr.getValue()
               << " flash-attention kernel limit of " << limit
               << " (Sprint G-2 NVIDIA kernel inventory)";
    }
  }
  return success();
}

// Sprint V6a (2026-05-22) — ReshapeOp: element-count-preserving.
// Input and output may have different rank, but their static dims
// must multiply to the same number and the element type must match.
// Dynamic dims are unconstrained (the runtime witness checks at
// dispatch time).
LogicalResult ReshapeOp::verify() {
  auto inTy = dyn_cast<RankedTensorType>(getX().getType());
  auto outTy = dyn_cast<RankedTensorType>(getY().getType());
  if (!inTy || !outTy) return success();
  if (inTy.getElementType() != outTy.getElementType())
    return emitOpError("reshape must preserve element type");
  // Static-dim product equality: only check when both shapes are
  // fully static.  Otherwise the runtime witness covers it.
  if (!inTy.hasStaticShape() || !outTy.hasStaticShape()) return success();
  int64_t inProd = 1;
  for (int64_t i = 0, e = inTy.getRank(); i < e; ++i)
    inProd *= inTy.getDimSize(i);
  int64_t outProd = 1;
  for (int64_t i = 0, e = outTy.getRank(); i < e; ++i)
    outProd *= outTy.getDimSize(i);
  if (inProd != outProd)
    return emitOpError("reshape must preserve element count: input has ")
           << inProd << " elements but output has " << outProd;
  return success();
}

// Sprint V1 (2026-05-22) — TransposeOp: rank-preserving permutation.
LogicalResult TransposeOp::verify() {
  auto inTy = dyn_cast<RankedTensorType>(getX().getType());
  auto outTy = dyn_cast<RankedTensorType>(getY().getType());
  if (!inTy || !outTy) return success();
  if (inTy.getRank() != outTy.getRank())
    return emitOpError("transpose must preserve rank: ")
           << inTy.getRank() << " -> " << outTy.getRank();
  if (inTy.getElementType() != outTy.getElementType())
    return emitOpError("transpose must preserve element type");
  // Multiset of static dim sizes must agree; dynamic dims contribute
  // nothing to the check (they could legitimately be any size).
  SmallVector<int64_t> inDims, outDims;
  for (int64_t i = 0, e = inTy.getRank(); i < e; ++i) {
    if (!ShapedType::isDynamic(inTy.getDimSize(i)))
      inDims.push_back(inTy.getDimSize(i));
    if (!ShapedType::isDynamic(outTy.getDimSize(i)))
      outDims.push_back(outTy.getDimSize(i));
  }
  if (inDims.size() == outDims.size()) {
    SmallVector<int64_t> a(inDims.begin(), inDims.end());
    SmallVector<int64_t> b(outDims.begin(), outDims.end());
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    if (a != b)
      return emitOpError(
          "output static dims must be a permutation of input static dims");
  }
  return success();
}

// Sprint V1 (2026-05-22) — LayerNormOp: shape-preserving + eps > 0.
LogicalResult LayerNormOp::verify() {
  auto inTy = dyn_cast<RankedTensorType>(getX().getType());
  auto outTy = dyn_cast<RankedTensorType>(getY().getType());
  if (inTy && outTy) {
    if (inTy.getRank() != outTy.getRank())
      return emitOpError("layer_norm must preserve rank");
    if (inTy.getElementType() != outTy.getElementType())
      return emitOpError("layer_norm must preserve element type");
    for (int64_t i = 0, e = inTy.getRank(); i < e; ++i) {
      int64_t in = inTy.getDimSize(i);
      int64_t out = outTy.getDimSize(i);
      if (!ShapedType::isDynamic(in) && !ShapedType::isDynamic(out)
          && in != out)
        return emitOpError("layer_norm must preserve dim ") << i;
    }
  }
  if (auto eps = getEps()) {
    double v = eps->convertToDouble();
    if (!(v > 0.0))
      return emitOpError("eps must be positive for stable rsqrt; got ") << v;
  }
  return success();
}

// Sprint V1 (2026-05-22) — MoeDispatchOp: token-count match on dim 0.
LogicalResult MoeDispatchOp::verify() {
  auto xTy = dyn_cast<RankedTensorType>(getX().getType());
  auto routeTy = dyn_cast<RankedTensorType>(getRoute().getType());
  if (!xTy || !routeTy) return success();
  if (xTy.getRank() < 1 || routeTy.getRank() < 1)
    return emitOpError(
        "moe_dispatch requires rank >= 1 token and route tensors");
  int64_t xN = xTy.getDimSize(0);
  int64_t rN = routeTy.getDimSize(0);
  if (!ShapedType::isDynamic(xN) && !ShapedType::isDynamic(rN)
      && xN != rN)
    return emitOpError("token count mismatch: x[0]=")
           << xN << " route[0]=" << rN;
  return success();
}

LogicalResult FusedEpilogueOp::verify() {
  if (!getHasBias()) {
    auto biasType = dyn_cast<ShapedType>(getBias().getType());
    if (biasType && biasType.hasStaticShape() && biasType.getNumElements() != 0)
      return emitOpError("bias operand must be empty when has_bias is false");
  }
  return success();
}

LogicalResult AttnLocalWindow2DOp::verify() {
  // Q, K, V must all be ranked rank-5 tensors of the same element type.
  auto qTy = dyn_cast<RankedTensorType>(getQ().getType());
  auto kTy = dyn_cast<RankedTensorType>(getK().getType());
  auto vTy = dyn_cast<RankedTensorType>(getV().getType());
  if (!qTy || !kTy || !vTy)
    return emitOpError("Q, K, V must be ranked tensors");
  if (qTy.getRank() != 5 || kTy.getRank() != 5 || vTy.getRank() != 5)
    return emitOpError(
        "Q, K, V must all be rank-5 (B, H, Hq, Wq, D) tensors");
  if (qTy.getElementType() != kTy.getElementType()
      || qTy.getElementType() != vTy.getElementType())
    return emitOpError("Q, K, V element types must match");

  // window = [rh, rw], non-negative.
  auto window = getWindow();
  if (window.size() != 2)
    return emitOpError("window must be [rh, rw] (length 2)");
  for (auto a : window) {
    auto i = dyn_cast<IntegerAttr>(a);
    if (!i || i.getInt() < 0)
      return emitOpError("window half-widths must be non-negative integers");
  }

  // K/V spatial axes (Hk, Wk) must match.
  for (int axis : {2, 3}) {
    if (kTy.isDynamicDim(axis) || vTy.isDynamicDim(axis)) continue;
    if (kTy.getDimSize(axis) != vTy.getDimSize(axis))
      return emitOpError("K and V must agree on spatial axis ") << axis;
  }
  // For self-attention v1, Q and K share spatial layout.  When either is
  // dynamic we leave the check to runtime.
  for (int axis : {2, 3}) {
    if (qTy.isDynamicDim(axis) || kTy.isDynamicDim(axis)) continue;
    if (qTy.getDimSize(axis) != kTy.getDimSize(axis))
      return emitOpError(
          "Q and K must share spatial layout on axis ") << axis;
  }
  return success();
}

// Sprint V4b (2026-05-22) — CastOp: shape-preserving element-type
// conversion.  Input and output must have the same rank + agreeing
// static dim sizes; only the element type may differ.
LogicalResult CastOp::verify() {
  auto inTy = dyn_cast<RankedTensorType>(getX().getType());
  auto outTy = dyn_cast<RankedTensorType>(getY().getType());
  if (!inTy || !outTy) return success();
  if (inTy.getRank() != outTy.getRank())
    return emitOpError("cast must preserve rank: ")
           << inTy.getRank() << " -> " << outTy.getRank();
  for (int64_t i = 0, e = inTy.getRank(); i < e; ++i) {
    int64_t in = inTy.getDimSize(i);
    int64_t out = outTy.getDimSize(i);
    if (!ShapedType::isDynamic(in) && !ShapedType::isDynamic(out)
        && in != out)
      return emitOpError("cast must preserve dim ") << i
             << ": " << in << " vs " << out;
  }
  return success();
}

// Sprint V4b (2026-05-22) — SoftmaxOp: shape-preserving normalization
// over an explicit axis.  When ``axis`` is set, normalize to
// canonical (non-negative) form and require ``-rank <= axis < rank``.
LogicalResult SoftmaxOp::verify() {
  auto inTy = dyn_cast<RankedTensorType>(getX().getType());
  auto outTy = dyn_cast<RankedTensorType>(getY().getType());
  if (inTy && outTy) {
    if (inTy.getRank() != outTy.getRank())
      return emitOpError("softmax must preserve rank");
    if (inTy.getElementType() != outTy.getElementType())
      return emitOpError("softmax must preserve element type");
    for (int64_t i = 0, e = inTy.getRank(); i < e; ++i) {
      int64_t in = inTy.getDimSize(i);
      int64_t out = outTy.getDimSize(i);
      if (!ShapedType::isDynamic(in) && !ShapedType::isDynamic(out)
          && in != out)
        return emitOpError("softmax must preserve dim ") << i;
    }
    if (auto axisOpt = getAxis()) {
      int64_t axis = *axisOpt;
      int64_t rank = inTy.getRank();
      if (axis < -rank || axis >= rank)
        return emitOpError("axis out of range: got ")
               << axis << " for rank-" << rank << " input "
               << "(expected -" << rank << " <= axis < " << rank << ")";
    }
  }
  return success();
}

// Sprint V4b (2026-05-22) — RopeOp: shape-preserving position
// embedding.  Input and output ranks + static dim sizes must agree.
LogicalResult RopeOp::verify() {
  auto inTy = dyn_cast<RankedTensorType>(getX().getType());
  auto outTy = dyn_cast<RankedTensorType>(getY().getType());
  if (!inTy || !outTy) return success();
  if (inTy.getRank() != outTy.getRank())
    return emitOpError("rope must preserve rank: ")
           << inTy.getRank() << " -> " << outTy.getRank();
  if (inTy.getElementType() != outTy.getElementType())
    return emitOpError("rope must preserve element type");
  for (int64_t i = 0, e = inTy.getRank(); i < e; ++i) {
    int64_t in = inTy.getDimSize(i);
    int64_t out = outTy.getDimSize(i);
    if (!ShapedType::isDynamic(in) && !ShapedType::isDynamic(out)
        && in != out)
      return emitOpError("rope must preserve dim ") << i
             << ": " << in << " vs " << out;
  }
  return success();
}

// Sprint V4b (2026-05-22) — DropoutOp: probability bounds + shape
// preservation.  ``p`` must satisfy ``0.0 <= p < 1.0`` (p=1 would
// zero every element; we reject as ill-defined rather than silently
// emit a zero tensor).
LogicalResult DropoutOp::verify() {
  if (auto p = getP()) {
    double v = p->convertToDouble();
    if (!(v >= 0.0) || !(v < 1.0))
      return emitOpError(
          "dropout probability must satisfy 0.0 <= p < 1.0; got ") << v;
  }
  auto inTy = dyn_cast<RankedTensorType>(getX().getType());
  auto outTy = dyn_cast<RankedTensorType>(getY().getType());
  if (inTy && outTy) {
    if (inTy.getRank() != outTy.getRank())
      return emitOpError("dropout must preserve rank");
    if (inTy.getElementType() != outTy.getElementType())
      return emitOpError("dropout must preserve element type");
    for (int64_t i = 0, e = inTy.getRank(); i < e; ++i) {
      int64_t in = inTy.getDimSize(i);
      int64_t out = outTy.getDimSize(i);
      if (!ShapedType::isDynamic(in) && !ShapedType::isDynamic(out)
          && in != out)
        return emitOpError("dropout must preserve dim ") << i;
    }
  }
  return success();
}

LogicalResult KVCacheCreateOp::verify() { return success(); }
LogicalResult RingCreateOp::verify() { return success(); }
LogicalResult ArchParameterOp::verify() { return success(); }
LogicalResult ArchGumbelSoftmaxOp::verify() { return success(); }
LogicalResult ArchHardConcreteOp::verify() { return success(); }
LogicalResult ArchSTEOneHotOp::verify() { return success(); }
LogicalResult ArchWeightedSumOp::verify() { return success(); }
LogicalResult ArchSwitchOp::verify() { return success(); }
LogicalResult ArchMixedOp::verify() { return success(); }

} // namespace tessera

#define GET_OP_CLASSES
#include "TesseraOps.cpp.inc"
