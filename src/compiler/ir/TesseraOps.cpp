#include "Tessera/IR/TesseraOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <algorithm>
#include <optional>

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

int64_t i64AttrOr(Operation *op, StringRef name, int64_t fallback) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return fallback;
}

bool optionalTensorPresent(Value value) { return static_cast<bool>(value); }

LogicalResult verifyPositiveAttr(Operation *op, StringRef name) {
  if (f64AttrOr(op, name, 0.0) <= 0.0)
    return op->emitOpError() << name << " must be positive";
  return success();
}

LogicalResult verifyNonNegativeOptionalAttr(Operation *op, StringRef name) {
  if (op->hasAttr(name) && f64AttrOr(op, name, 0.0) < 0.0)
    return op->emitOpError() << name << " must be non-negative";
  return success();
}

LogicalResult verifyFloatSameDtype(Operation *op,
                                   ArrayRef<RankedTensorType> tensors,
                                   StringRef label) {
  RankedTensorType first;
  for (RankedTensorType ty : tensors) {
    if (!ty)
      continue;
    if (!first)
      first = ty;
    if (!isFloatTensor(ty))
      return op->emitOpError() << label << " expects floating tensors";
    if (ty.getElementType() != first.getElementType())
      return op->emitOpError() << label << " dtypes must match";
  }
  return success();
}

LogicalResult verifyOptionalTensorShape(Operation *op, Value maybe,
                                        RankedTensorType reference,
                                        StringRef label) {
  if (!maybe)
    return success();
  auto ty = dyn_cast<RankedTensorType>(maybe.getType());
  if (!ty)
    return success();
  if (ty.getElementType() != reference.getElementType())
    return op->emitOpError() << label << " dtype must match state dtype";
  return verifySameRankedShape(op, reference, ty, label);
}

LogicalResult verifyPositiveI64(Operation *op, StringRef name, int64_t value) {
  if (value <= 0)
    return op->emitOpError() << name << " must be positive";
  return success();
}

LogicalResult verifyPositiveOptionalI64(Operation *op, StringRef name,
                                        std::optional<int64_t> value) {
  if (value && *value <= 0)
    return op->emitOpError() << name << " must be positive when set";
  return success();
}

LogicalResult verifyAttentionQKV(Operation *op, Value q, Value k, Value v,
                                 Value o, StringRef label) {
  auto qTy = dyn_cast<RankedTensorType>(q.getType());
  auto kTy = dyn_cast<RankedTensorType>(k.getType());
  auto vTy = dyn_cast<RankedTensorType>(v.getType());
  auto oTy = dyn_cast<RankedTensorType>(o.getType());
  if (!qTy || !kTy || !vTy || !oTy)
    return success();
  if (qTy.getRank() < 2 || kTy.getRank() < 2 || vTy.getRank() < 2 ||
      oTy.getRank() < 2)
    return op->emitOpError() << label << " expects rank >= 2 tensors";
  if (failed(verifyFloatSameDtype(op, {qTy, kTy, vTy, oTy}, label)))
    return failure();
  int64_t qD = qTy.getDimSize(qTy.getRank() - 1);
  int64_t kD = kTy.getDimSize(kTy.getRank() - 1);
  if (!dimsAgree(qD, kD))
    return op->emitOpError() << label
                             << " requires Q/K head dimensions to match";
  int64_t vTokens = vTy.getDimSize(vTy.getRank() - 2);
  int64_t kTokens = kTy.getDimSize(kTy.getRank() - 2);
  if (!dimsAgree(kTokens, vTokens))
    return op->emitOpError() << label
                             << " requires K/V sequence dimensions to match";
  int64_t qTokens = qTy.getDimSize(qTy.getRank() - 2);
  int64_t oTokens = oTy.getDimSize(oTy.getRank() - 2);
  if (!dimsAgree(qTokens, oTokens))
    return op->emitOpError() << label
                             << " output sequence must match Q sequence";
  int64_t vD = vTy.getDimSize(vTy.getRank() - 1);
  int64_t oD = oTy.getDimSize(oTy.getRank() - 1);
  if (!dimsAgree(vD, oD))
    return op->emitOpError() << label
                             << " output feature dimension must match V";
  return success();
}

LogicalResult verifyMatmulLike(Operation *op, Value lhs, Value rhs, Value out,
                               StringRef label) {
  auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
  auto outTy = dyn_cast<RankedTensorType>(out.getType());
  if (!lhsTy || !rhsTy || !outTy)
    return success();
  if (lhsTy.getRank() < 2 || rhsTy.getRank() < 2 || outTy.getRank() < 2)
    return op->emitOpError() << label << " expects rank >= 2 tensors";
  if (failed(verifyFloatSameDtype(op, {lhsTy, rhsTy, outTy}, label)))
    return failure();
  int64_t lhsK = lhsTy.getDimSize(lhsTy.getRank() - 1);
  int64_t rhsK = rhsTy.getDimSize(rhsTy.getRank() - 2);
  if (!dimsAgree(lhsK, rhsK))
    return op->emitOpError() << label
                             << " contracting dimensions must match";
  int64_t lhsM = lhsTy.getDimSize(lhsTy.getRank() - 2);
  int64_t outM = outTy.getDimSize(outTy.getRank() - 2);
  if (!dimsAgree(lhsM, outM))
    return op->emitOpError() << label << " output M must match lhs";
  int64_t rhsN = rhsTy.getDimSize(rhsTy.getRank() - 1);
  int64_t outN = outTy.getDimSize(outTy.getRank() - 1);
  if (!dimsAgree(rhsN, outN))
    return op->emitOpError() << label << " output N must match rhs";
  return success();
}

LogicalResult verifyOptionalOperandsMatchQ(Operation *op, RankedTensorType qTy,
                                           unsigned firstOperand,
                                           StringRef label) {
  for (unsigned i = firstOperand, e = op->getNumOperands(); i < e; ++i) {
    auto ty = dyn_cast<RankedTensorType>(op->getOperand(i).getType());
    if (!ty)
      continue;
    if (!isFloatTensor(ty))
      return op->emitOpError() << label
                               << " optional tensor operands must be floating";
    if (ty.getElementType() != qTy.getElementType())
      return op->emitOpError() << label
                               << " optional tensor dtype must match Q";
  }
  return success();
}

LogicalResult verifyAllowedStringAttr(Operation *op, StringRef name,
                                      ArrayRef<StringRef> allowed,
                                      StringRef fallback = "") {
  StringRef value = fallback;
  if (auto attr = op->getAttrOfType<StringAttr>(name))
    value = attr.getValue();
  if (value.empty())
    return success();
  for (StringRef a : allowed)
    if (value == a)
      return success();
  return op->emitOpError() << name << " has unsupported value '" << value
                           << "'";
}

LogicalResult verifyAttentionFeatureMap(Operation *op, StringRef featureMap) {
  return verifyAllowedStringAttr(
      op, "feature_map", {"elu", "relu", "identity", "softplus"},
      featureMap);
}

LogicalResult verifyTemperatureNoiseCoupling(Operation *op, bool hasNoise,
                                             bool hasRNGState = false) {
  double temperature = f64AttrOr(op, "temperature", 0.0);
  if (temperature < 0.0)
    return op->emitOpError("temperature must be non-negative");
  if (temperature > 0.0 && !hasNoise && !hasRNGState)
    return op->emitOpError(
        "temperature > 0 requires a noise operand or RNG state");
  return success();
}

LogicalResult verifyEBMAffineStep(Operation *op, RankedTensorType state,
                                  RankedTensorType grad, Value noise,
                                  RankedTensorType result,
                                  StringRef label) {
  if (!state || !grad || !result)
    return success();
  if (failed(verifyFloatSameDtype(op, {state, grad, result}, label)))
    return failure();
  if (failed(verifySameRankedShape(op, state, grad, label)) ||
      failed(verifySameRankedShape(op, state, result, label)) ||
      failed(verifyOptionalTensorShape(op, noise, state, label)))
    return failure();
  if (failed(verifyPositiveAttr(op, "eta")) ||
      failed(verifyNonNegativeOptionalAttr(op, "noise_scale")))
    return failure();
  if (failed(verifyTemperatureNoiseCoupling(op, optionalTensorPresent(noise))))
    return failure();
  if (f64AttrOr(op, "noise_scale", 0.0) > 0.0 && !noise)
    return op->emitOpError("noise_scale > 0 requires a noise operand");
  return success();
}

int64_t bladeCountFor(int64_t p, int64_t q) {
  int64_t n = p + q;
  if (n < 0 || n > 30)
    return -1;
  return int64_t{1} << n;
}

LogicalResult verifyCliffordMetadata(Operation *op, int64_t p, int64_t q) {
  if (p < 0 || q < 0)
    return op->emitOpError("Clifford signature p/q must be non-negative");
  if (bladeCountFor(p, q) <= 0)
    return op->emitOpError("Clifford algebra rank p+q is too large");

  auto readI64Array = [](Attribute attr, SmallVectorImpl<int64_t> &out) {
    if (auto dense = dyn_cast_or_null<DenseI64ArrayAttr>(attr)) {
      out.append(dense.asArrayRef().begin(), dense.asArrayRef().end());
      return true;
    }
    if (auto array = dyn_cast_or_null<ArrayAttr>(attr)) {
      for (Attribute element : array) {
        auto intAttr = dyn_cast<IntegerAttr>(element);
        if (!intAttr)
          return false;
        out.push_back(intAttr.getInt());
      }
      return true;
    }
    return false;
  };

  if (Attribute sigAttr = op->getAttr("signature")) {
    SmallVector<int64_t> signature;
    if (!readI64Array(sigAttr, signature))
      return op->emitOpError("signature must be an i64 array");
    if (static_cast<int64_t>(signature.size()) != p + q)
      return op->emitOpError("signature length must equal p+q");
    for (int64_t v : signature)
      if (v != -1 && v != 1)
        return op->emitOpError("signature entries must be +1 or -1");
  }
  return verifyAllowedStringAttr(
      op, "coefficient_layout",
      {StringRef("blade_last"), StringRef("coeff_last"),
       StringRef("blade_major")},
      StringRef("blade_last"));
}

LogicalResult verifyCliffordTensor(Operation *op, RankedTensorType ty,
                                   int64_t p, int64_t q, StringRef label) {
  if (!ty)
    return success();
  if (!isFloatTensor(ty))
    return op->emitOpError() << label << " must be a floating tensor";
  if (ty.getRank() < 1)
    return op->emitOpError() << label << " must include a coefficient axis";
  int64_t blades = bladeCountFor(p, q);
  int64_t coeffs = ty.getDimSize(ty.getRank() - 1);
  if (!ShapedType::isDynamic(coeffs) && blades > 0 && coeffs != blades)
    return op->emitOpError()
           << label << " coefficient axis must equal 2^(p+q)";
  return success();
}

LogicalResult verifyCliffordSameShapeAndDtype(Operation *op,
                                             ArrayRef<RankedTensorType> tensors,
                                             StringRef label) {
  RankedTensorType first;
  for (RankedTensorType ty : tensors) {
    if (!ty)
      continue;
    if (!first) {
      first = ty;
      continue;
    }
    if (ty.getElementType() != first.getElementType())
      return op->emitOpError() << label << " dtypes must match";
    if (failed(verifySameRankedShape(op, first, ty, label)))
      return failure();
  }
  return success();
}

LogicalResult verifyGradeMask(Operation *op, int64_t p, int64_t q,
                              StringRef attrName) {
  Attribute maskAttr = op->getAttr(attrName);
  if (!maskAttr)
    return success();
  SmallVector<int64_t> mask;
  if (auto dense = dyn_cast<DenseI64ArrayAttr>(maskAttr)) {
    mask.append(dense.asArrayRef().begin(), dense.asArrayRef().end());
  } else if (auto array = dyn_cast<ArrayAttr>(maskAttr)) {
    for (Attribute element : array) {
      auto intAttr = dyn_cast<IntegerAttr>(element);
      if (!intAttr)
        return op->emitOpError() << attrName << " must be an i64 array";
      mask.push_back(intAttr.getInt());
    }
  } else {
    return op->emitOpError() << attrName << " must be an i64 array";
  }
  if (mask.empty())
    return op->emitOpError() << attrName << " must not be empty";
  int64_t maxGrade = p + q;
  for (int64_t grade : mask)
    if (grade < 0 || grade > maxGrade)
      return op->emitOpError()
             << attrName << " entries must be in [0, p+q]";
  return success();
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
  if (failed(verifyFloatSameDtype(getOperation(), {x, y, energies},
                                  "ebm.energy_quadratic")))
    return failure();
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
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  return verifyEBMAffineStep(getOperation(), y, grad, getNoise(), result,
                             "ebm.langevin_step");
}

LogicalResult EBMInnerStepOp::verify() {
  auto y = dyn_cast<RankedTensorType>(getY().getType());
  auto grad = dyn_cast<RankedTensorType>(getGrad().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  return verifyEBMAffineStep(getOperation(), y, grad, getNoise(), result,
                             "ebm.inner_step");
}

LogicalResult EBMRefinementOp::verify() {
  auto y = dyn_cast<RankedTensorType>(getY().getType());
  auto grad = dyn_cast<RankedTensorType>(getGrad().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (i64AttrOr(getOperation(), "steps", 0) <= 0)
    return emitOpError("steps must be positive");
  return verifyEBMAffineStep(getOperation(), y, grad, getNoise(), result,
                             "ebm.refinement");
}

LogicalResult EBMLangevinStepPhiloxOp::verify() {
  auto y = dyn_cast<RankedTensorType>(getY().getType());
  auto grad = dyn_cast<RankedTensorType>(getGrad().getType());
  auto seed = dyn_cast<RankedTensorType>(getSeed().getType());
  auto counter = dyn_cast<RankedTensorType>(getCounter().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!y || !grad || !result)
    return success();
  if (failed(verifyFloatSameDtype(getOperation(), {y, grad, result},
                                  "ebm.langevin_step_philox")) ||
      failed(verifySameRankedShape(getOperation(), y, grad,
                                   "ebm.langevin_step_philox grad")) ||
      failed(verifySameRankedShape(getOperation(), y, result,
                                   "ebm.langevin_step_philox result")) ||
      failed(verifyPositiveAttr(getOperation(), "eta")) ||
      failed(verifyPositiveAttr(getOperation(), "temperature")) ||
      failed(verifyNonNegativeOptionalAttr(getOperation(), "noise_scale")))
    return failure();
  if (!seed || !counter)
    return success();
  if (!seed.getElementType().isInteger(64) || !counter.getElementType().isInteger(64))
    return emitOpError("Philox seed/counter tensors must be i64");
  if (seed.getRank() != 1 || counter.getRank() != 1)
    return emitOpError("Philox seed/counter tensors must be rank-1");
  if (!dimsAgree(seed.getDimSize(0), 1))
    return emitOpError("Philox seed tensor must have length 1");
  if (!dimsAgree(counter.getDimSize(0), 4))
    return emitOpError("Philox counter tensor must have length 4");
  return success();
}

LogicalResult EBMDecodeInitOp::verify() {
  auto x = dyn_cast<RankedTensorType>(getX().getType());
  auto candidates = dyn_cast<RankedTensorType>(getCandidates().getType());
  if (!x || !candidates)
    return success();
  if (failed(verifyFloatSameDtype(getOperation(), {x, candidates},
                                  "ebm.decode_init")))
    return failure();
  if (candidates.getRank() != x.getRank() + 1)
    return emitOpError("candidates rank must be input rank + 1");
  if (!dimsAgree(candidates.getDimSize(0), x.getDimSize(0)))
    return emitOpError("candidate batch dimension must match input batch");
  for (int64_t i = 1, e = x.getRank(); i < e; ++i)
    if (!dimsAgree(candidates.getDimSize(i + 1), x.getDimSize(i)))
      return emitOpError("candidate trailing dimensions must match input");
  int64_t k = i64AttrOr(getOperation(), "steps", 0);
  if (k <= 0)
    return emitOpError("steps must be positive");
  if (!dimsAgree(candidates.getDimSize(1), k))
    return emitOpError("candidate axis must equal steps");
  if (failed(verifyAllowedStringAttr(getOperation(), "strategy",
                                     {StringRef("noise"), StringRef("copy"),
                                      StringRef("base_model")})))
    return failure();
  StringRef strategy = getOperation()->getAttrOfType<StringAttr>("strategy")
                           .getValue();
  if (strategy == "noise" && !getNoise() && !getOperation()->hasAttr("seed"))
    return emitOpError("strategy=\"noise\" requires noise operand or seed attr");
  if (strategy == "base_model" && !getBase())
    return emitOpError("strategy=\"base_model\" requires base operand");
  if (failed(verifyOptionalTensorShape(getOperation(), getBase(), x,
                                       "ebm.decode_init base")) ||
      failed(verifyOptionalTensorShape(getOperation(), getNoise(), candidates,
                                       "ebm.decode_init noise")))
    return failure();
  return success();
}

LogicalResult EBMSelfVerifyOp::verify() {
  auto energies = dyn_cast<RankedTensorType>(getEnergies().getType());
  auto candidates = dyn_cast<RankedTensorType>(getCandidates().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!energies || !candidates || !result)
    return success();
  if (failed(verifyFloatSameDtype(getOperation(), {energies, candidates, result},
                                  "ebm.self_verify")))
    return failure();
  if (energies.getRank() != 2)
    return emitOpError("energies must be rank-2 [B,K]");
  if (candidates.getRank() < 2)
    return emitOpError("candidates must include batch and candidate axes");
  if (!dimsAgree(energies.getDimSize(0), candidates.getDimSize(0)) ||
      !dimsAgree(energies.getDimSize(1), candidates.getDimSize(1)))
    return emitOpError("energies and candidates B/K axes must match");
  if (result.getRank() != candidates.getRank() - 1)
    return emitOpError("result rank must be candidates rank - 1");
  if (!dimsAgree(result.getDimSize(0), candidates.getDimSize(0)))
    return emitOpError("result batch dimension must match candidates");
  for (int64_t i = 2, e = candidates.getRank(); i < e; ++i)
    if (!dimsAgree(result.getDimSize(i - 1), candidates.getDimSize(i)))
      return emitOpError("result trailing dimensions must match candidate values");
  if (failed(verifyAllowedStringAttr(getOperation(), "reduction",
                                     {StringRef("hard_argmin"),
                                      StringRef("softmin")},
                                     StringRef("hard_argmin"))))
    return failure();
  double temperature = f64AttrOr(getOperation(), "temperature", 0.0);
  if (temperature < 0.0)
    return emitOpError("temperature must be non-negative");
  if (auto r = getOperation()->getAttrOfType<StringAttr>("reduction");
      r && r.getValue() == "softmin" && temperature <= 0.0)
    return emitOpError("reduction=\"softmin\" requires temperature > 0");
  return success();
}

LogicalResult EBMPartitionExactOp::verify() {
  auto energies = dyn_cast<RankedTensorType>(getEnergies().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!energies || !result)
    return success();
  if (failed(verifyFloatSameDtype(getOperation(), {energies, result},
                                  "ebm.partition_exact")))
    return failure();
  if (result.getRank() != 0 && result.getRank() != energies.getRank() - 1)
    return emitOpError(
        "result must be scalar or reduce exactly one candidate axis");
  if (result.getRank() == energies.getRank() - 1)
    for (int64_t i = 0, e = result.getRank(); i < e; ++i)
      if (!dimsAgree(result.getDimSize(i), energies.getDimSize(i)))
        return emitOpError(
            "partition result dimensions must match non-candidate axes");
  if (f64AttrOr(getOperation(), "temperature", 1.0) <= 0.0)
    return emitOpError("temperature must be positive");
  return verifyAllowedStringAttr(getOperation(), "reduction",
                                 {StringRef("logsumexp"), StringRef("sum"),
                                  StringRef("mean")},
                                 StringRef("logsumexp"));
}

LogicalResult EBMBivectorLangevinStepOp::verify() {
  auto state = dyn_cast<RankedTensorType>(getState().getType());
  auto grad = dyn_cast<RankedTensorType>(getGrad().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (i64AttrOr(getOperation(), "grade", 2) != 2)
    return emitOpError("bivector Langevin requires grade = 2");
  if (failed(verifyAllowedStringAttr(getOperation(), "manifold",
                                     {StringRef("bivector"),
                                      StringRef("so_n")},
                                     StringRef("bivector"))) ||
      failed(verifyAllowedStringAttr(getOperation(), "projection",
                                     {StringRef("grade"),
                                      StringRef("bivector")},
                                     StringRef("grade"))) ||
      failed(verifyAllowedStringAttr(getOperation(), "metric",
                                     {StringRef("euclidean"),
                                      StringRef("killing")},
                                     StringRef("euclidean"))))
    return failure();
  return verifyEBMAffineStep(getOperation(), state, grad, getNoise(), result,
                             "ebm.bivector_langevin_step");
}

LogicalResult EBMSphereLangevinStepOp::verify() {
  auto state = dyn_cast<RankedTensorType>(getState().getType());
  auto grad = dyn_cast<RankedTensorType>(getGrad().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (getOperation()->hasAttr("normalized_state") &&
      !getOperation()->getAttrOfType<BoolAttr>("normalized_state").getValue())
    return emitOpError("sphere Langevin requires normalized_state = true");
  if (failed(verifyAllowedStringAttr(getOperation(), "manifold",
                                     {StringRef("sphere"),
                                      StringRef("unit_sphere")},
                                     StringRef("sphere"))) ||
      failed(verifyAllowedStringAttr(getOperation(), "projection",
                                     {StringRef("tangent"),
                                      StringRef("normalize_retract")},
                                     StringRef("tangent"))) ||
      failed(verifyAllowedStringAttr(getOperation(), "metric",
                                     {StringRef("riemannian"),
                                      StringRef("euclidean")},
                                     StringRef("riemannian"))))
    return failure();
  return verifyEBMAffineStep(getOperation(), state, grad, getNoise(), result,
                             "ebm.sphere_langevin_step");
}

LogicalResult CliffordGeometricProductOp::verify() {
  int64_t p = getP(), q = getQ();
  auto lhs = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhs = dyn_cast<RankedTensorType>(getRhs().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (failed(verifyCliffordMetadata(getOperation(), p, q)) ||
      failed(verifyCliffordTensor(getOperation(), lhs, p, q, "lhs")) ||
      failed(verifyCliffordTensor(getOperation(), rhs, p, q, "rhs")) ||
      failed(verifyCliffordTensor(getOperation(), result, p, q, "result")) ||
      failed(verifyCliffordSameShapeAndDtype(getOperation(),
                                             {lhs, rhs, result},
                                             "clifford.geometric_product")))
    return failure();
  return success();
}

LogicalResult CliffordOuterProductOp::verify() {
  int64_t p = getP(), q = getQ();
  auto lhs = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhs = dyn_cast<RankedTensorType>(getRhs().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (failed(verifyCliffordMetadata(getOperation(), p, q)) ||
      failed(verifyCliffordTensor(getOperation(), lhs, p, q, "lhs")) ||
      failed(verifyCliffordTensor(getOperation(), rhs, p, q, "rhs")) ||
      failed(verifyCliffordTensor(getOperation(), result, p, q, "result")) ||
      failed(verifyCliffordSameShapeAndDtype(getOperation(),
                                             {lhs, rhs, result},
                                             "clifford.outer_product")))
    return failure();
  return success();
}

LogicalResult CliffordInnerProductOp::verify() {
  int64_t p = getP(), q = getQ();
  auto lhs = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhs = dyn_cast<RankedTensorType>(getRhs().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (failed(verifyCliffordMetadata(getOperation(), p, q)) ||
      failed(verifyCliffordTensor(getOperation(), lhs, p, q, "lhs")) ||
      failed(verifyCliffordTensor(getOperation(), rhs, p, q, "rhs")) ||
      failed(verifyCliffordTensor(getOperation(), result, p, q, "result")) ||
      failed(verifyCliffordSameShapeAndDtype(getOperation(),
                                             {lhs, rhs, result},
                                             "clifford.inner_product")))
    return failure();
  return success();
}

LogicalResult CliffordReverseOp::verify() {
  int64_t p = getP(), q = getQ();
  auto input = dyn_cast<RankedTensorType>(getInput().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (failed(verifyCliffordMetadata(getOperation(), p, q)) ||
      failed(verifyCliffordTensor(getOperation(), input, p, q, "input")) ||
      failed(verifyCliffordTensor(getOperation(), result, p, q, "result")) ||
      failed(verifyCliffordSameShapeAndDtype(getOperation(),
                                             {input, result},
                                             "clifford.reverse")))
    return failure();
  return success();
}

LogicalResult CliffordGradeProjectOp::verify() {
  int64_t p = getP(), q = getQ();
  auto input = dyn_cast<RankedTensorType>(getInput().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (failed(verifyCliffordMetadata(getOperation(), p, q)) ||
      failed(verifyGradeMask(getOperation(), p, q, "grade_mask")) ||
      failed(verifyCliffordTensor(getOperation(), input, p, q, "input")) ||
      failed(verifyCliffordTensor(getOperation(), result, p, q, "result")) ||
      failed(verifyCliffordSameShapeAndDtype(getOperation(),
                                             {input, result},
                                             "clifford.grade_project")))
    return failure();
  return success();
}

LogicalResult CliffordNormOp::verify() {
  int64_t p = getP(), q = getQ();
  auto input = dyn_cast<RankedTensorType>(getInput().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (failed(verifyCliffordMetadata(getOperation(), p, q)) ||
      failed(verifyCliffordTensor(getOperation(), input, p, q, "input")))
    return failure();
  if (!input || !result)
    return success();
  if (!isFloatTensor(result) || result.getElementType() != input.getElementType())
    return emitOpError("result must be a floating tensor with input dtype");
  if (result.getRank() == input.getRank()) {
    if (failed(verifyCliffordTensor(getOperation(), result, p, q, "result")) ||
        failed(verifySameRankedShape(getOperation(), input, result,
                                     "clifford.norm")))
      return failure();
    return success();
  }
  if (result.getRank() != input.getRank() - 1)
    return emitOpError(
        "result must either preserve coefficients or drop the coefficient axis");
  for (int64_t i = 0, e = result.getRank(); i < e; ++i)
    if (!dimsAgree(result.getDimSize(i), input.getDimSize(i)))
      return emitOpError("norm result batch dimensions must match input");
  return success();
}

LogicalResult CliffordRotorSandwichOp::verify() {
  int64_t p = getP(), q = getQ();
  auto rotor = dyn_cast<RankedTensorType>(getRotor().getType());
  auto value = dyn_cast<RankedTensorType>(getValue().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (failed(verifyCliffordMetadata(getOperation(), p, q)) ||
      failed(verifyGradeMask(getOperation(), p, q, "grade_mask")) ||
      failed(verifyCliffordTensor(getOperation(), rotor, p, q, "rotor")) ||
      failed(verifyCliffordTensor(getOperation(), value, p, q, "value")) ||
      failed(verifyCliffordTensor(getOperation(), result, p, q, "result")) ||
      failed(verifyCliffordSameShapeAndDtype(getOperation(),
                                             {rotor, value, result},
                                             "clifford.rotor_sandwich")))
    return failure();
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
      uint64_t headDim = getHeadDim();
      if (limit > 0 && headDim > static_cast<uint64_t>(limit))
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

LogicalResult MoeCombineOp::verify() {
  auto partialsTy = dyn_cast<RankedTensorType>(getPartials().getType());
  auto routeTy = dyn_cast<RankedTensorType>(getInverseRoute().getType());
  auto outTy = dyn_cast<RankedTensorType>(getX().getType());
  if (failed(verifyAllowedStringAttr(this->getOperation(), "reduce",
                                     {"sum", "mean"}, "sum")))
    return failure();
  if (!partialsTy || !routeTy || !outTy)
    return success();
  if (partialsTy.getRank() < 1 || routeTy.getRank() < 1 || outTy.getRank() < 1)
    return emitOpError(
        "moe_combine requires rank >= 1 partial, route, and output tensors");
  if (partialsTy.getElementType() != outTy.getElementType())
    return emitOpError("partials/output dtypes must match");
  int64_t routeN = routeTy.getDimSize(0);
  int64_t outN = outTy.getDimSize(0);
  if (!dimsAgree(routeN, outN))
    return emitOpError("inverse_route token count must match output dim 0");
  return success();
}

LogicalResult MultiHeadAttentionOp::verify() {
  if (failed(verifyPositiveI64(this->getOperation(), "num_heads",
                               getNumHeads())))
    return failure();
  return verifyAttentionQKV(this->getOperation(), getQ(), getK(), getV(),
                            getO(), "multi_head_attention");
}

LogicalResult GQAAttentionOp::verify() {
  int64_t qHeads = getNumQueryHeads();
  int64_t kvHeads = getNumKvHeads();
  if (failed(verifyPositiveI64(this->getOperation(), "num_query_heads",
                               qHeads)) ||
      failed(verifyPositiveI64(this->getOperation(), "num_kv_heads",
                               kvHeads)))
    return failure();
  if (qHeads % kvHeads != 0)
    return emitOpError(
        "num_query_heads must be divisible by num_kv_heads");
  return verifyAttentionQKV(this->getOperation(), getQ(), getK(), getV(),
                            getO(), "gqa_attention");
}

LogicalResult MQAAttentionOp::verify() {
  return verifyAttentionQKV(this->getOperation(), getQ(), getK(), getV(),
                            getO(), "mqa_attention");
}

LogicalResult MLADecodeOp::verify() {
  auto qTy = dyn_cast<RankedTensorType>(getQ().getType());
  auto kTy = dyn_cast<RankedTensorType>(getKLatent().getType());
  auto vTy = dyn_cast<RankedTensorType>(getVLatent().getType());
  auto oTy = dyn_cast<RankedTensorType>(getO().getType());
  if (!qTy || !kTy || !vTy || !oTy)
    return success();
  if (failed(verifyFloatSameDtype(this->getOperation(), {qTy, kTy, vTy, oTy},
                                  "mla_decode")))
    return failure();
  for (Value weight : getWeights()) {
    auto wTy = dyn_cast<RankedTensorType>(weight.getType());
    if (!wTy)
      continue;
    if (!isFloatTensor(wTy))
      return emitOpError("MLA weights must be floating tensors");
    if (wTy.getElementType() != qTy.getElementType())
      return emitOpError("MLA weight dtype must match q dtype");
  }
  return success();
}

LogicalResult LinearAttnOp::verify() {
  if (failed(verifyAttentionFeatureMap(this->getOperation(), getFeatureMap())))
    return failure();
  if (failed(verifyPositiveOptionalI64(this->getOperation(), "chunk_size",
                                       getChunkSize())))
    return failure();
  return verifyAttentionQKV(this->getOperation(), getQ(), getK(), getV(),
                            getO(), "linear_attn");
}

LogicalResult LinearAttnStateOp::verify() {
  if (failed(verifyAttentionFeatureMap(this->getOperation(), getFeatureMap())))
    return failure();
  if (failed(verifyPositiveOptionalI64(this->getOperation(), "chunk_size",
                                       getChunkSize())))
    return failure();
  auto qTy = dyn_cast<RankedTensorType>(getQ().getType());
  auto kTy = dyn_cast<RankedTensorType>(getK().getType());
  auto vTy = dyn_cast<RankedTensorType>(getV().getType());
  auto stateTy = dyn_cast<RankedTensorType>(getState().getType());
  if (!qTy || !kTy || !vTy || !stateTy)
    return success();
  return verifyFloatSameDtype(this->getOperation(), {qTy, kTy, vTy, stateTy},
                              "linear_attn_state");
}

LogicalResult PowerAttnOp::verify() {
  if (failed(verifyPositiveI64(this->getOperation(), "state", getState())) ||
      failed(verifyPositiveI64(this->getOperation(), "deg", getDeg())) ||
      failed(verifyPositiveOptionalI64(this->getOperation(), "window",
                                       getWindow())))
    return failure();
  return verifyAttentionQKV(this->getOperation(), getQ(), getK(), getV(),
                            getO(), "power_attn");
}

LogicalResult LightningAttentionOp::verify() {
  if (failed(verifyAllowedStringAttr(this->getOperation(), "state_dtype",
                                     {"fp32", "fp16", "bf16"}, "fp32")))
    return failure();
  if (failed(verifyAttentionQKV(this->getOperation(), getQ(), getK(), getV(),
                                getO(), "lightning_attention")))
    return failure();
  auto qTy = dyn_cast<RankedTensorType>(getQ().getType());
  if (!qTy)
    return success();
  if (failed(verifyOptionalTensorShape(this->getOperation(), getState(), qTy,
                                       "state")) ||
      failed(verifyOptionalTensorShape(this->getOperation(), getDecay(), qTy,
                                       "decay")))
    return failure();
  return success();
}

LogicalResult GatedAttentionOp::verify() {
  if (failed(verifyAllowedStringAttr(this->getOperation(), "gate_activation",
                                     {"sigmoid", "silu", "linear"},
                                     "sigmoid")))
    return failure();
  if (failed(verifyAttentionQKV(this->getOperation(), getQ(), getK(), getV(),
                                getO(), "gated_attention")))
    return failure();
  auto qTy = dyn_cast<RankedTensorType>(getQ().getType());
  auto gateTy = dyn_cast<RankedTensorType>(getGate().getType());
  if (!qTy || !gateTy)
    return success();
  if (!isFloatTensor(gateTy))
    return emitOpError("gate must be a floating tensor");
  int64_t qTokens = qTy.getDimSize(qTy.getRank() - 2);
  int64_t gateTokens = gateTy.getDimSize(gateTy.getRank() - 2);
  if (!dimsAgree(qTokens, gateTokens))
    return emitOpError("gate sequence dimension must match Q");
  return success();
}

LogicalResult RetentionOp::verify() {
  if (failed(verifyPositiveI64(this->getOperation(), "deg", getDeg())) ||
      failed(verifyPositiveI64(this->getOperation(), "chunk", getChunk())) ||
      failed(verifyPositiveOptionalI64(this->getOperation(), "switch_over",
                                       getSwitchOver())))
    return failure();
  if (failed(verifyAttentionQKV(this->getOperation(), getQ(), getK(), getV(),
                                getO(), "retention")))
    return failure();
  auto qTy = dyn_cast<RankedTensorType>(getQ().getType());
  auto stateTy = dyn_cast<RankedTensorType>(getState().getType());
  auto sumTy = dyn_cast<RankedTensorType>(getSumOfKeys().getType());
  if (failed(verifyFloatSameDtype(this->getOperation(), {qTy, stateTy, sumTy},
                                  "retention state")))
    return failure();
  return success();
}

LogicalResult GatedDeltaNetOp::verify() {
  if (failed(verifyAllowedStringAttr(this->getOperation(), "state_dtype",
                                     {"fp32", "fp16", "bf16"}, "fp32")))
    return failure();
  if (failed(verifyAttentionQKV(this->getOperation(), getQ(), getK(), getV(),
                                getO(), "gated_deltanet")))
    return failure();
  auto qTy = dyn_cast<RankedTensorType>(getQ().getType());
  if (!qTy)
    return success();
  return verifyOptionalOperandsMatchQ(this->getOperation(), qTy, 3,
                                      "gated_deltanet");
}

LogicalResult KimiDeltaAttentionOp::verify() {
  if (failed(verifyAllowedStringAttr(this->getOperation(), "state_dtype",
                                     {"fp32", "fp16", "bf16"}, "fp32")))
    return failure();
  if (failed(verifyAttentionQKV(this->getOperation(), getQ(), getK(), getV(),
                                getO(), "kimi_delta_attention")))
    return failure();
  auto qTy = dyn_cast<RankedTensorType>(getQ().getType());
  if (!qTy)
    return success();
  return verifyOptionalOperandsMatchQ(this->getOperation(), qTy, 3,
                                      "kimi_delta_attention");
}

LogicalResult ModifiedDeltaAttentionOp::verify() {
  if (failed(verifyAllowedStringAttr(this->getOperation(), "state_dtype",
                                     {"fp32", "fp16", "bf16"}, "fp32")))
    return failure();
  if (failed(verifyAttentionQKV(this->getOperation(), getQ(), getK(), getV(),
                                getO(), "modified_delta_attention")))
    return failure();
  auto qTy = dyn_cast<RankedTensorType>(getQ().getType());
  if (!qTy)
    return success();
  return verifyOptionalOperandsMatchQ(this->getOperation(), qTy, 3,
                                      "modified_delta_attention");
}

LogicalResult HybridAttentionOp::verify() {
  // ``pattern`` on hybrid_attention is a *free-form* model-specific hybrid
  // variant (e.g. "kimi_kda_mla", "ling_1_7_mla_lightning"), not a closed
  // category enum: the reasoning passes pass it through verbatim as
  // ``tessera.reasoning.variant`` (see AttentionFamilyPasses::hybridVariant).
  // Only ``state_dtype`` is a closed set.
  if (failed(verifyAllowedStringAttr(this->getOperation(), "state_dtype",
                                     {"fp32", "fp16", "bf16"}, "fp32")))
    return failure();
  if (getLayerIndex() < 0)
    return emitOpError("layer_index must be non-negative");
  return verifyAttentionQKV(this->getOperation(), getQ(), getK(), getV(),
                            getO(), "hybrid_attention");
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

LogicalResult MLADecodeFusedOp::verify() {
  auto xTy = dyn_cast<RankedTensorType>(getOperation()->getOperand(0).getType());
  auto wDkvTy = dyn_cast<RankedTensorType>(getOperation()->getOperand(1).getType());
  auto wUkTy = dyn_cast<RankedTensorType>(getOperation()->getOperand(2).getType());
  auto wUvTy = dyn_cast<RankedTensorType>(getOperation()->getOperand(3).getType());
  auto qTy = dyn_cast<RankedTensorType>(getOperation()->getOperand(4).getType());
  auto oTy = dyn_cast<RankedTensorType>(getOperation()->getResult(0).getType());
  if (!xTy || !wDkvTy || !wUkTy || !wUvTy || !qTy || !oTy)
    return success();
  if (failed(verifyFloatSameDtype(this->getOperation(),
                                  {xTy, wDkvTy, wUkTy, wUvTy, qTy, oTy},
                                  "mla_decode_fused")))
    return failure();
  if (xTy.getRank() < 2 || qTy.getRank() < 2 || oTy.getRank() < 2)
    return emitOpError("mla_decode_fused expects rank >= 2 x/q/o tensors");
  return success();
}

LogicalResult LatentKVCompressOp::verify() {
  return verifyMatmulLike(this->getOperation(),
                          getOperation()->getOperand(0),
                          getOperation()->getOperand(1),
                          getOperation()->getResult(0),
                          "latent_kv_compress");
}

LogicalResult LatentKVExpandKOp::verify() {
  return verifyMatmulLike(this->getOperation(),
                          getOperation()->getOperand(0),
                          getOperation()->getOperand(1),
                          getOperation()->getResult(0),
                          "latent_kv_expand_k");
}

LogicalResult LatentKVExpandVOp::verify() {
  return verifyMatmulLike(this->getOperation(),
                          getOperation()->getOperand(0),
                          getOperation()->getOperand(1),
                          getOperation()->getResult(0),
                          "latent_kv_expand_v");
}

LogicalResult AttnSlidingWindowOp::verify() {
  if (failed(verifyPositiveI64(this->getOperation(), "window_size",
                               getWindowSize())))
    return failure();
  return verifyAttentionQKV(this->getOperation(), getQ(), getK(), getV(),
                            getO(), "attn_sliding_window");
}

LogicalResult AttnCompressedBlocksOp::verify() {
  return verifyAttentionQKV(this->getOperation(),
                            getOperation()->getOperand(0),
                            getOperation()->getOperand(1),
                            getOperation()->getOperand(2),
                            getOperation()->getResult(0),
                            "attn_compressed_blocks");
}

LogicalResult AttnTopKBlocksOp::verify() {
  if (failed(verifyPositiveI64(this->getOperation(), "top_k", getTopK())) ||
      failed(verifyPositiveI64(this->getOperation(), "block_size",
                               getBlockSize())))
    return failure();
  return verifyAttentionQKV(this->getOperation(), getQ(), getK(), getV(),
                            getO(), "attn_top_k_blocks");
}

LogicalResult NativeSparseAttnFusedOp::verify() {
  if (failed(verifyPositiveI64(this->getOperation(), "window_size",
                               getWindowSize())) ||
      failed(verifyPositiveI64(this->getOperation(), "block_size",
                               getBlockSize())) ||
      failed(verifyPositiveI64(this->getOperation(), "top_k", getTopK())))
    return failure();
  if (failed(verifyAttentionQKV(this->getOperation(), getQ(), getK(), getV(),
                                getO(), "native_sparse_attn_fused")))
    return failure();
  auto qTy = dyn_cast<RankedTensorType>(getQ().getType());
  auto gateTy = dyn_cast<RankedTensorType>(getGateLogits().getType());
  if (!qTy || !gateTy)
    return success();
  if (!isFloatTensor(gateTy))
    return emitOpError("gate_logits must be a floating tensor");
  if (gateTy.getElementType() != qTy.getElementType())
    return emitOpError("gate_logits dtype must match Q");
  int64_t qTokens = qTy.getDimSize(qTy.getRank() - 2);
  int64_t gateTokens = gateTy.getDimSize(gateTy.getRank() - 2);
  if (!dimsAgree(qTokens, gateTokens))
    return emitOpError("gate_logits sequence dimension must match Q");
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

LogicalResult DeepSeekSparseAttentionOp::verify() {
  if (failed(verifyPositiveI64(this->getOperation(), "window_size",
                               getWindowSize())) ||
      failed(verifyPositiveI64(this->getOperation(), "block_size",
                               getBlockSize())) ||
      failed(verifyPositiveI64(this->getOperation(), "top_k", getTopK())))
    return failure();
  if (failed(verifyAttentionQKV(this->getOperation(), getQ(), getK(), getV(),
                                getO(), "deepseek_sparse_attention")))
    return failure();
  if (getOperation()->getNumOperands() > 3) {
    auto gateTy =
        dyn_cast<RankedTensorType>(getOperation()->getOperand(3).getType());
    if (gateTy && !isFloatTensor(gateTy))
      return emitOpError("gate_logits must be a floating tensor");
  }
  return success();
}

LogicalResult KVCacheAppendOp::verify() {
  auto kTy = dyn_cast<RankedTensorType>(getK().getType());
  auto vTy = dyn_cast<RankedTensorType>(getV().getType());
  if (!kTy || !vTy)
    return success();
  if (failed(verifyFloatSameDtype(this->getOperation(), {kTy, vTy},
                                  "kv_cache.append")))
    return failure();
  if (kTy.getRank() < 2 || vTy.getRank() < 2)
    return emitOpError("kv_cache.append expects rank >= 2 key/value tensors");
  int64_t kTokens = kTy.getDimSize(kTy.getRank() - 2);
  int64_t vTokens = vTy.getDimSize(vTy.getRank() - 2);
  if (!dimsAgree(kTokens, vTokens))
    return emitOpError("key/value token dimensions must match");
  return success();
}

LogicalResult KVCachePruneOp::verify() {
  if (getWindow() <= 0)
    return emitOpError("window must be positive");
  return success();
}

// Sprint V8 (2026-06-07) — norm/softmax family shape+dtype contracts.
// These ops are pointwise-over-the-normalized-axis: they preserve rank,
// per-axis static dims, and element type (mirrors SoftmaxOp / LayerNormOp).
static LogicalResult verifyShapeDtypePreserving(Operation *op, Value in,
                                                Value out, StringRef name) {
  auto inTy = dyn_cast<RankedTensorType>(in.getType());
  auto outTy = dyn_cast<RankedTensorType>(out.getType());
  if (!inTy || !outTy)
    return success();
  if (inTy.getRank() != outTy.getRank())
    return op->emitOpError() << name << " must preserve rank";
  if (inTy.getElementType() != outTy.getElementType())
    return op->emitOpError() << name << " must preserve element type";
  for (int64_t i = 0, e = inTy.getRank(); i < e; ++i) {
    int64_t a = inTy.getDimSize(i), b = outTy.getDimSize(i);
    if (!ShapedType::isDynamic(a) && !ShapedType::isDynamic(b) && a != b)
      return op->emitOpError() << name << " must preserve dim " << i;
  }
  return success();
}

LogicalResult RmsNormOp::verify() {
  if (failed(verifyShapeDtypePreserving(getOperation(), getX(), getY(),
                                        "rmsnorm")))
    return failure();
  if (auto eps = getEps()) {
    double v = eps->convertToDouble();
    if (!(v > 0.0))
      return emitOpError("eps must be positive for stable rsqrt; got ") << v;
  }
  return success();
}

LogicalResult RMSNormSafeOp::verify() {
  if (failed(verifyShapeDtypePreserving(getOperation(), getX(), getY(),
                                        "rmsnorm_safe")))
    return failure();
  if (auto eps = getEps()) {
    double v = eps->convertToDouble();
    if (!(v > 0.0))
      return emitOpError("eps must be positive for stable rsqrt; got ") << v;
  }
  return success();
}

LogicalResult SoftmaxSafeOp::verify() {
  return verifyShapeDtypePreserving(getOperation(), getX(), getY(),
                                    "softmax_safe");
}

LogicalResult LogSoftmaxOp::verify() {
  if (failed(verifyShapeDtypePreserving(getOperation(), getX(), getY(),
                                        "log_softmax")))
    return failure();
  if (auto axisOpt = getAxis()) {
    if (auto inTy = dyn_cast<RankedTensorType>(getX().getType())) {
      int64_t axis = *axisOpt, rank = inTy.getRank();
      if (axis < -rank || axis >= rank)
        return emitOpError("axis out of range: got ")
               << axis << " for rank-" << rank << " input "
               << "(expected -" << rank << " <= axis < " << rank << ")";
    }
  }
  return success();
}

// Sprint V9 (2026-06-07) — close the trivial-stub verifiers with real
// resource/scalar contracts.
LogicalResult KVCacheCreateOp::verify() {
  if (failed(verifyPositiveI64(getOperation(), "max_seq", getMaxSeq())) ||
      failed(verifyPositiveI64(getOperation(), "head_dim", getHeadDim())))
    return failure();
  if (auto ps = getPageSize())
    return verifyPositiveI64(getOperation(), "page_size", *ps);
  return success();
}
LogicalResult RingCreateOp::verify() {
  return verifyPositiveI64(getOperation(), "capacity", getCapacity());
}

LogicalResult ArchParameterOp::verify() {
  return verifyPositiveI64(getOperation(), "size", getSize());
}
LogicalResult ArchGumbelSoftmaxOp::verify() {
  return verifyPositiveAttr(getOperation(), "temperature");
}
LogicalResult ArchHardConcreteOp::verify() {
  return verifyPositiveAttr(getOperation(), "temperature");
}
// STE one-hot: an opaque ArchParam → ArchGate with no scalar/shape contract to
// check beyond ODS typing — left as a structural success.
LogicalResult ArchSTEOneHotOp::verify() { return success(); }
// WeightedSum / Switch: a non-empty candidate set, each candidate shape-matching
// the (single) mixed result.
static LogicalResult verifyArchCandidates(Operation *op,
                                          OperandRange candidates, Value result) {
  if (candidates.empty())
    return op->emitOpError("requires at least one candidate");
  auto outTy = dyn_cast<RankedTensorType>(result.getType());
  if (!outTy)
    return success();
  for (Value c : candidates)
    if (auto cty = dyn_cast<RankedTensorType>(c.getType()))
      if (failed(verifySameRankedShape(op, cty, outTy, "candidate")))
        return failure();
  return success();
}
LogicalResult ArchWeightedSumOp::verify() {
  return verifyArchCandidates(getOperation(), getCandidates(), getResult());
}
LogicalResult ArchSwitchOp::verify() {
  return verifyArchCandidates(getOperation(), getCandidates(), getResult());
}
LogicalResult ArchMixedOp::verify() {
  if (getCandidates().empty())
    return emitOpError("candidates array must be non-empty");
  return success();
}

// ── Phase-G control flow — payload-ABI + carry/flag-index contracts. ──────────
// The serialized run_graph op-list arrays (opcodes/in0/in1/iattr/fattr) must be
// mutually length-consistent (one entry per body op) with out_id set, else the
// runtime reads past the op-list. IR-only ops (no opcodes) skip the check.
static LogicalResult verifyControlPayload(Operation *op, StringRef prefix) {
  auto i32 = [&](const Twine &suffix) -> std::optional<ArrayRef<int32_t>> {
    if (auto a = op->getAttrOfType<DenseI32ArrayAttr>((prefix + suffix).str()))
      return a.asArrayRef();
    return std::nullopt;
  };
  auto opcodes = i32("_opcodes");
  if (!opcodes)
    return success();
  int64_t n = static_cast<int64_t>(opcodes->size());
  for (const char *suf : {"_in0", "_in1", "_iattr"}) {
    if (auto arr = i32(suf))
      if (static_cast<int64_t>(arr->size()) != n)
        return op->emitOpError()
               << prefix << suf << " length (" << arr->size()
               << ") must match " << prefix << "_opcodes length (" << n << ")";
  }
  if (auto fattr = op->getAttrOfType<DenseF32ArrayAttr>((prefix + "_fattr").str()))
    if (static_cast<int64_t>(fattr.size()) != n)
      return op->emitOpError() << prefix << "_fattr length must match "
                               << prefix << "_opcodes length (" << n << ")";
  if (!op->getAttrOfType<IntegerAttr>((prefix + "_out_id").str()))
    return op->emitOpError()
           << prefix << "_out_id required when " << prefix
           << "_opcodes is present";
  return success();
}

LogicalResult ControlForOp::verify() {
  if (getStep() == 0)
    return emitOpError("step must be non-zero");
  int64_t n = static_cast<int64_t>(getIterArgs().size());
  // Executable-payload form: `iter_args` are [carry + loop-invariant consts],
  // `carry_arg_index` selects the one loop-carried operand (the rest are
  // invariant captures, not carried). The op yields exactly one result whose
  // type matches the carried operand — mirroring ControlWhileOp. This is what
  // the front-end emits whenever the loop closes over consts (a weight matrix,
  // etc.); requiring one-result-per-operand here would reject every loop with
  // a single const capture.
  if (auto opt = getCarryArgIndex()) {
    int64_t idx = static_cast<int64_t>(*opt);
    if (idx < 0 || idx >= n)
      return emitOpError("carry_arg_index out of range: ") << idx;
    if (getResults().size() != 1)
      return emitOpError("loop with carry_arg_index carries one value "
                         "(#results=")
             << getResults().size() << ", expected 1)";
    if (n > 0 && getResults()[0].getType() != getIterArgs()[idx].getType())
      return emitOpError("for result type must match the carried iter_arg type");
    return verifyControlPayload(getOperation(), "body");
  }
  // Legacy IR-only form (no carry_arg_index): every iter_arg is loop-carried.
  if (getResults().size() != getIterArgs().size())
    return emitOpError("loop must carry each iter_arg to a result (#results=")
           << getResults().size() << ", #iter_args=" << getIterArgs().size()
           << ")";
  for (auto [in, out] : llvm::zip(getIterArgs(), getResults()))
    if (in.getType() != out.getType())
      return emitOpError("loop-carried type mismatch between iter_arg and result");
  return verifyControlPayload(getOperation(), "body");
}

LogicalResult ControlIfOp::verify() {
  int64_t n = static_cast<int64_t>(getIterArgs().size());
  int64_t flag = getFlagArgIndex();
  if (flag < 0 || flag >= n)
    return emitOpError("flag_arg_index out of range: ") << flag;
  bool hasThen = getOperation()->getAttrOfType<DenseI32ArrayAttr>("then_opcodes")
                 != nullptr;
  bool hasElse = getOperation()->getAttrOfType<DenseI32ArrayAttr>("else_opcodes")
                 != nullptr;
  if (hasThen != hasElse)
    return emitOpError("then/else payloads must both be present or both absent");
  if (failed(verifyControlPayload(getOperation(), "then")) ||
      failed(verifyControlPayload(getOperation(), "else")))
    return failure();
  return success();
}

LogicalResult ControlWhileOp::verify() {
  if (getMaxIters() <= 0)
    return emitOpError("max_iters must be positive");
  int64_t n = static_cast<int64_t>(getIterArgs().size());
  int64_t idx = getCarryArgIndex();
  if (idx < 0 || idx >= n)
    return emitOpError("carry_arg_index out of range: ") << idx;
  if (!getResults().empty() &&
      getResults()[0].getType() != getIterArgs()[idx].getType())
    return emitOpError("while result type must match the carried iter_arg type");
  if (failed(verifyControlPayload(getOperation(), "body")) ||
      failed(verifyControlPayload(getOperation(), "cond")))
    return failure();
  return success();
}

// ── MoR (mixture-of-recursions) ──────────────────────────────────────────────
LogicalResult MorRouterOp::verify() {
  return verifyPositiveI64(getOperation(), "max_depth", getMaxDepth());
}
LogicalResult MorPartitionOp::verify() {
  if (getStep() < 0)
    return emitOpError("step must be non-negative");
  return success();
}
LogicalResult MorScatterOp::verify() {
  auto fullTy = dyn_cast<RankedTensorType>(getFull().getType());
  auto outTy = dyn_cast<RankedTensorType>(getOut().getType());
  if (fullTy && outTy)
    return verifyShapeDtypePreserving(getOperation(), getFull(), getOut(),
                                      "mor_scatter");
  return success();
}

// ── Quantize / dequantize — shape-preserving (dtype changes), format set. ─────
static LogicalResult verifyQuantFormat(Operation *op, StringRef name,
                                       ArrayRef<StringRef> allowed) {
  return verifyAllowedStringAttr(op, name, allowed);
}
static LogicalResult verifyQuantShape(Operation *op, Value a, Value b,
                                      StringRef label) {
  auto aTy = dyn_cast<RankedTensorType>(a.getType());
  auto bTy = dyn_cast<RankedTensorType>(b.getType());
  if (aTy && bTy)
    return verifySameRankedShape(op, aTy, bTy, label);
  return success();
}
LogicalResult QuantizeFP8Op::verify() {
  if (failed(verifyQuantFormat(getOperation(), "format", {"e4m3", "e5m2"})))
    return failure();
  return verifyQuantShape(getOperation(), getX(), getXQ(), "quantize_fp8");
}
LogicalResult DequantizeFP8Op::verify() {
  if (failed(verifyQuantFormat(getOperation(), "format", {"e4m3", "e5m2"})))
    return failure();
  return verifyQuantShape(getOperation(), getXQ(), getX(), "dequantize_fp8");
}
LogicalResult QuantizeFP4Op::verify() {
  if (failed(verifyQuantFormat(getOperation(), "format", {"e2m1", "nvfp4"})))
    return failure();
  return verifyQuantShape(getOperation(), getX(), getXQ(), "quantize_fp4");
}
LogicalResult DequantizeFP4Op::verify() {
  if (failed(verifyQuantFormat(getOperation(), "format", {"e2m1", "nvfp4"})))
    return failure();
  return verifyQuantShape(getOperation(), getXQ(), getX(), "dequantize_fp4");
}

// ── Spectral (FFT family) — axis-in-range against the input rank. ─────────────
static LogicalResult verifySpectralAxis(Operation *op, Value x,
                                        std::optional<int64_t> axis,
                                        StringRef name) {
  if (!axis)
    return success();
  auto xTy = dyn_cast<RankedTensorType>(x.getType());
  if (!xTy)
    return success();
  int64_t rank = xTy.getRank(), a = *axis;
  if (a < -rank || a >= rank)
    return op->emitOpError() << name << " axis out of range: got " << a
                             << " for rank-" << rank << " input";
  return success();
}
LogicalResult FFTOp::verify() {
  return verifySpectralAxis(getOperation(), getX(), getAxis(), "fft");
}
LogicalResult IFFTOp::verify() {
  return verifySpectralAxis(getOperation(), getX(), getAxis(), "ifft");
}
LogicalResult RFFTOp::verify() {
  return verifySpectralAxis(getOperation(), getX(), getAxis(), "rfft");
}
LogicalResult IRFFTOp::verify() {
  return verifySpectralAxis(getOperation(), getX(), getAxis(), "irfft");
}
LogicalResult DCTOp::verify() {
  return verifySpectralAxis(getOperation(), getX(), getAxis(), "dct");
}

} // namespace tessera

#define GET_OP_CLASSES
#include "TesseraOps.cpp.inc"
