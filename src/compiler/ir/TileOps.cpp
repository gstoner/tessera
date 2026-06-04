//===- TileOps.cpp - Tessera Tile IR op verifiers -------------*- C++ -*-===//
//
// Sprint 9: rank/shape verifiers for the contraction tile ops. These mirror the
// Graph IR MatmulOp / BatchedGemmOp verifiers — the value lane only ever
// produces in-envelope tile ops, and this re-checks at the Tile layer so a
// malformed tile op can never reach the Target lowering.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Dialect/Tile/TileDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace tessera {
namespace tile {

namespace {
// Read an optional discardable bool attr (transposeA/transposeB), default false.
bool boolAttr(Operation *op, llvm::StringRef name) {
  if (auto a = op->getAttrOfType<BoolAttr>(name))
    return a.getValue();
  return false;
}

bool sameStaticShape(RankedTensorType a, RankedTensorType b) {
  if (a.getRank() != b.getRank())
    return false;
  if (!a.hasStaticShape() || !b.hasStaticShape())
    return false;
  for (int64_t i = 0, e = a.getRank(); i < e; ++i)
    if (a.getDimSize(i) != b.getDimSize(i))
      return false;
  return true;
}

LogicalResult requireBoolAttr(Operation *op, llvm::StringRef name,
                              bool expected) {
  if (boolAttr(op, name) != expected)
    return op->emitOpError()
           << name << " must be " << (expected ? "true" : "false");
  return success();
}

int64_t i64AttrOr(Operation *op, llvm::StringRef name, int64_t fallback) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return fallback;
}

int64_t bladeCountFor(int64_t p, int64_t q) {
  int64_t n = p + q;
  if (n < 0 || n > 30)
    return -1;
  return int64_t{1} << n;
}

bool isStaticF32RankedTensor(Type ty, RankedTensorType &out) {
  out = dyn_cast<RankedTensorType>(ty);
  return out && out.hasStaticShape() && out.getElementType().isF32();
}

LogicalResult verifyCliffordAttrs(Operation *op) {
  int64_t p = i64AttrOr(op, "p", -1);
  int64_t q = i64AttrOr(op, "q", -1);
  if (p != 3 || q != 0)
    return op->emitOpError("Clifford Tile value seam currently requires p=3, q=0");
  if (bladeCountFor(p, q) != 8)
    return op->emitOpError("Clifford Tile value seam expects cl30 coefficient axis");

  if (Attribute sigAttr = op->getAttr("signature")) {
    SmallVector<int64_t> signature;
    if (auto dense = dyn_cast<DenseI64ArrayAttr>(sigAttr)) {
      signature.append(dense.asArrayRef().begin(), dense.asArrayRef().end());
    } else if (auto array = dyn_cast<ArrayAttr>(sigAttr)) {
      for (Attribute element : array) {
        auto intAttr = dyn_cast<IntegerAttr>(element);
        if (!intAttr)
          return op->emitOpError("signature must be an i64 array");
        signature.push_back(intAttr.getInt());
      }
    } else {
      return op->emitOpError("signature must be an i64 array");
    }
    if (signature.size() != 3)
      return op->emitOpError("signature length must equal p+q");
    for (int64_t v : signature)
      if (v != 1 && v != -1)
        return op->emitOpError("signature entries must be +1 or -1");
  }
  return success();
}

LogicalResult verifyCliffordTensor(Operation *op, RankedTensorType ty,
                                   llvm::StringRef label) {
  if (!ty)
    return success();
  if (ty.getRank() < 1)
    return op->emitOpError() << label << " must include a coefficient axis";
  if (ty.getDimSize(ty.getRank() - 1) != 8)
    return op->emitOpError()
           << label << " coefficient axis must equal 8 for cl30";
  return success();
}

LogicalResult verifyGradeMask(Operation *op) {
  Attribute maskAttr = op->getAttr("grade_mask");
  if (!maskAttr)
    return success();
  SmallVector<int64_t> mask;
  if (auto dense = dyn_cast<DenseI64ArrayAttr>(maskAttr)) {
    mask.append(dense.asArrayRef().begin(), dense.asArrayRef().end());
  } else if (auto array = dyn_cast<ArrayAttr>(maskAttr)) {
    for (Attribute element : array) {
      auto intAttr = dyn_cast<IntegerAttr>(element);
      if (!intAttr)
        return op->emitOpError("grade_mask must be an i64 array");
      mask.push_back(intAttr.getInt());
    }
  } else {
    return op->emitOpError("grade_mask must be an i64 array");
  }
  if (mask.empty())
    return op->emitOpError("grade_mask must not be empty");
  for (int64_t grade : mask)
    if (grade < 0 || grade > 3)
      return op->emitOpError("grade_mask entries must be in [0, p+q]");
  return success();
}

LogicalResult verifyCliffordBinary(Operation *op, ValueRange inputs,
                                   ValueRange outputs,
                                   llvm::StringRef label) {
  if (inputs.size() != 2 || outputs.size() != 1)
    return op->emitOpError("expects exactly 2 inputs and 1 result");
  if (failed(verifyCliffordAttrs(op)) || failed(verifyGradeMask(op)))
    return failure();
  RankedTensorType lhs, rhs, res;
  if (!isStaticF32RankedTensor(inputs[0].getType(), lhs) ||
      !isStaticF32RankedTensor(inputs[1].getType(), rhs) ||
      !isStaticF32RankedTensor(outputs[0].getType(), res))
    return op->emitOpError()
           << label << " expects static fp32 tensors and result";
  if (failed(verifyCliffordTensor(op, lhs, "lhs")) ||
      failed(verifyCliffordTensor(op, rhs, "rhs")) ||
      failed(verifyCliffordTensor(op, res, "result")))
    return failure();
  if (!sameStaticShape(lhs, rhs) || !sameStaticShape(lhs, res))
    return op->emitOpError() << label << " shapes must match exactly";
  return success();
}

LogicalResult verifyCliffordUnary(Operation *op, ValueRange inputs,
                                  ValueRange outputs,
                                  llvm::StringRef label,
                                  bool allowNormDrop = false) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return op->emitOpError("expects exactly 1 input and 1 result");
  if (failed(verifyCliffordAttrs(op)) || failed(verifyGradeMask(op)))
    return failure();
  RankedTensorType input, res;
  if (!isStaticF32RankedTensor(inputs[0].getType(), input) ||
      !isStaticF32RankedTensor(outputs[0].getType(), res))
    return op->emitOpError()
           << label << " expects static fp32 tensors and result";
  if (failed(verifyCliffordTensor(op, input, "input")))
    return failure();
  if (allowNormDrop && res.getRank() == input.getRank() - 1) {
    for (int64_t i = 0, e = res.getRank(); i < e; ++i)
      if (res.getDimSize(i) != input.getDimSize(i))
        return op->emitOpError("norm result batch dimensions must match input");
    return success();
  }
  if (failed(verifyCliffordTensor(op, res, "result")))
    return failure();
  if (!sameStaticShape(input, res))
    return op->emitOpError() << label << " shapes must match exactly";
  return success();
}

// Shared rank-2 matmul contract (honors transposeA/transposeB discardable
// attrs). lhs/rhs/result must be rank-2; K, M, N consistent.
LogicalResult verifyRank2Matmul(Operation *op, ValueRange inputs,
                                ValueRange outputs) {
  if (inputs.size() != 2 || outputs.size() != 1)
    return op->emitOpError("expects exactly 2 inputs and 1 result");
  auto lhs = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto rhs = dyn_cast<RankedTensorType>(inputs[1].getType());
  auto res = dyn_cast<RankedTensorType>(outputs[0].getType());
  if (!lhs || !rhs || !res)
    return success(); // unranked — nothing to check
  if (lhs.getRank() != 2 || rhs.getRank() != 2 || res.getRank() != 2)
    return op->emitOpError("expects rank-2 lhs, rhs, and result tensors");
  bool ta = boolAttr(op, "transposeA"), tb = boolAttr(op, "transposeB");
  int64_t lk = ta ? lhs.getDimSize(0) : lhs.getDimSize(1);
  int64_t rk = tb ? rhs.getDimSize(1) : rhs.getDimSize(0);
  auto dyn = [](int64_t d) { return ShapedType::isDynamic(d); };
  if (!dyn(lk) && !dyn(rk) && lk != rk)
    return op->emitOpError("contracting dimensions must match");
  int64_t m = ta ? lhs.getDimSize(1) : lhs.getDimSize(0);
  int64_t n = tb ? rhs.getDimSize(0) : rhs.getDimSize(1);
  if (!dyn(m) && !dyn(res.getDimSize(0)) && m != res.getDimSize(0))
    return op->emitOpError("result row dimension must equal lhs M");
  if (!dyn(n) && !dyn(res.getDimSize(1)) && n != res.getDimSize(1))
    return op->emitOpError("result column dimension must equal rhs N");
  return success();
}
} // namespace

LogicalResult MatmulOp::verify() {
  return verifyRank2Matmul(getOperation(), getInputs(), getOutputs());
}

LogicalResult GemmOp::verify() {
  return verifyRank2Matmul(getOperation(), getInputs(), getOutputs());
}

LogicalResult BatchedGemmOp::verify() {
  auto inputs = getInputs();
  auto outputs = getOutputs();
  if (inputs.size() != 2 || outputs.size() != 1)
    return emitOpError("expects exactly 2 inputs and 1 result");
  auto lhs = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto rhs = dyn_cast<RankedTensorType>(inputs[1].getType());
  auto res = dyn_cast<RankedTensorType>(outputs[0].getType());
  if (!lhs || !rhs || !res)
    return success();
  if (lhs.getRank() != 3 || rhs.getRank() != 3 || res.getRank() != 3)
    return emitOpError("expects rank-3 lhs, rhs, and result tensors");
  auto dyn = [](int64_t d) { return ShapedType::isDynamic(d); };
  auto agree = [&](int64_t a, int64_t b) { return dyn(a) || dyn(b) || a == b; };
  if (!agree(lhs.getDimSize(0), rhs.getDimSize(0)) ||
      !agree(lhs.getDimSize(0), res.getDimSize(0)))
    return emitOpError("batch dimensions must match (no broadcasting)");
  if (!agree(lhs.getDimSize(2), rhs.getDimSize(1)))
    return emitOpError("contracting dimensions must match");
  if (!agree(res.getDimSize(1), lhs.getDimSize(1)))
    return emitOpError("result M must equal lhs M");
  if (!agree(res.getDimSize(2), rhs.getDimSize(2)))
    return emitOpError("result N must equal rhs N");
  return success();
}

LogicalResult PPOPolicyLossOp::verify() {
  auto inputs = getInputs();
  auto outputs = getOutputs();
  if (inputs.size() < 3 || inputs.size() > 6 || outputs.size() != 1)
    return emitOpError("expects 3 to 6 inputs and 1 result");
  auto next = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto old = dyn_cast<RankedTensorType>(inputs[1].getType());
  auto adv = dyn_cast<RankedTensorType>(inputs[2].getType());
  auto res = dyn_cast<RankedTensorType>(outputs[0].getType());
  if (!next || !old || !adv || !res)
    return success();
  if (!next.hasStaticShape() || !old.hasStaticShape() ||
      !adv.hasStaticShape() || !res.hasStaticShape())
    return emitOpError("expects static fp32 operands and rank-0 fp32 result");
  if (!next.getElementType().isF32() || !old.getElementType().isF32() ||
      !adv.getElementType().isF32() || !res.getElementType().isF32())
    return emitOpError("expects fp32 operands and fp32 result");
  if (!sameStaticShape(next, old) || !sameStaticShape(next, adv))
    return emitOpError("input shapes must match exactly");
  for (auto input : inputs.drop_front(3)) {
    auto side = dyn_cast<RankedTensorType>(input.getType());
    if (!side)
      continue;
    if (!side.hasStaticShape() || !side.getElementType().isF32() ||
        !sameStaticShape(next, side))
      return emitOpError(
          "optional mask/ref_logp/entropy inputs must be static fp32 tensors "
          "with the same shape as logp_new");
  }
  if (res.getRank() != 0)
    return emitOpError("mean-reduction result must be rank-0 tensor");
  if (auto r = getOperation()->getAttrOfType<StringAttr>("reduction");
      r && r.getValue() != "mean")
    return emitOpError("only reduction=\"mean\" is executable in Tile PPO");
  if (auto c = getOperation()->getAttrOfType<FloatAttr>("clip_epsilon");
      c && c.getValueAsDouble() <= 0.0)
    return emitOpError("clip_epsilon must be positive");
  if (auto k = getOperation()->getAttrOfType<FloatAttr>("kl_coef");
      k && k.getValueAsDouble() < 0.0)
    return emitOpError("kl_coef must be non-negative for Tile PPO value mode");
  if (auto e = getOperation()->getAttrOfType<FloatAttr>("entropy_coef");
      e && e.getValueAsDouble() < 0.0)
    return emitOpError("entropy_coef must be non-negative for Tile PPO value mode");
  return success();
}

LogicalResult EBMEnergyQuadraticOp::verify() {
  auto inputs = getInputs();
  auto outputs = getOutputs();
  if (inputs.size() != 2 || outputs.size() != 1)
    return emitOpError("expects exactly 2 inputs and 1 result");
  auto x = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto y = dyn_cast<RankedTensorType>(inputs[1].getType());
  auto energies = dyn_cast<RankedTensorType>(outputs[0].getType());
  if (!x || !y || !energies)
    return success();
  if (!x.hasStaticShape() || !y.hasStaticShape() ||
      !energies.hasStaticShape())
    return emitOpError("expects static fp32 rank-2 inputs and rank-1 result");
  if (!x.getElementType().isF32() || !y.getElementType().isF32() ||
      !energies.getElementType().isF32())
    return emitOpError("expects fp32 inputs and fp32 result");
  if (x.getRank() != 2 || y.getRank() != 2)
    return emitOpError("expects rank-2 x and y tensors");
  if (energies.getRank() != 1)
    return emitOpError("energies result must be rank-1");
  if (!sameStaticShape(x, y))
    return emitOpError("x and y shapes must match exactly");
  if (energies.getDimSize(0) != x.getDimSize(0))
    return emitOpError("energies length must equal batch dimension");
  return success();
}

LogicalResult EBMLangevinStepOp::verify() {
  auto inputs = getInputs();
  auto outputs = getOutputs();
  if (inputs.size() != 3 || outputs.size() != 1)
    return emitOpError("expects exactly 3 inputs and 1 result");
  if (failed(requireBoolAttr(getOperation(), "has_noise", true)))
    return failure();
  auto y = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto grad = dyn_cast<RankedTensorType>(inputs[1].getType());
  auto noise = dyn_cast<RankedTensorType>(inputs[2].getType());
  auto out = dyn_cast<RankedTensorType>(outputs[0].getType());
  if (!y || !grad || !noise || !out)
    return success();
  if (!y.hasStaticShape() || !grad.hasStaticShape() ||
      !noise.hasStaticShape() || !out.hasStaticShape())
    return emitOpError("expects static fp32 operands and result");
  if (!y.getElementType().isF32() || !grad.getElementType().isF32() ||
      !noise.getElementType().isF32() || !out.getElementType().isF32())
    return emitOpError("expects fp32 operands and fp32 result");
  if (!sameStaticShape(y, grad) || !sameStaticShape(y, noise) ||
      !sameStaticShape(y, out))
    return emitOpError("y/grad/noise/result shapes must match exactly");
  if (auto eta = getOperation()->getAttrOfType<FloatAttr>("eta");
      !eta || eta.getValueAsDouble() <= 0.0)
    return emitOpError("eta must be present and positive");
  if (auto scale = getOperation()->getAttrOfType<FloatAttr>("noise_scale");
      scale && scale.getValueAsDouble() < 0.0)
    return emitOpError("noise_scale must be non-negative");
  return success();
}

LogicalResult CliffordGeometricProductOp::verify() {
  return verifyCliffordBinary(getOperation(), getInputs(), getOutputs(),
                              "clifford_geometric_product");
}

LogicalResult CliffordOuterProductOp::verify() {
  return verifyCliffordBinary(getOperation(), getInputs(), getOutputs(),
                              "clifford_outer_product");
}

LogicalResult CliffordInnerProductOp::verify() {
  return verifyCliffordBinary(getOperation(), getInputs(), getOutputs(),
                              "clifford_inner_product");
}

LogicalResult CliffordReverseOp::verify() {
  return verifyCliffordUnary(getOperation(), getInputs(), getOutputs(),
                             "clifford_reverse");
}

LogicalResult CliffordGradeProjectOp::verify() {
  return verifyCliffordUnary(getOperation(), getInputs(), getOutputs(),
                             "clifford_grade_project");
}

LogicalResult CliffordNormOp::verify() {
  return verifyCliffordUnary(getOperation(), getInputs(), getOutputs(),
                             "clifford_norm", /*allowNormDrop=*/true);
}

LogicalResult CliffordRotorSandwichOp::verify() {
  return verifyCliffordBinary(getOperation(), getInputs(), getOutputs(),
                              "clifford_rotor_sandwich");
}

} // namespace tile
} // namespace tessera
