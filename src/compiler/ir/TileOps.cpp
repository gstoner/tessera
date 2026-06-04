//===- TileOps.cpp - Tessera Tile IR op verifiers -------------*- C++ -*-===//
//
// Sprint 9: rank/shape verifiers for the contraction tile ops. These mirror the
// Graph IR MatmulOp / BatchedGemmOp verifiers — the value lane only ever
// produces in-envelope tile ops, and this re-checks at the Tile layer so a
// malformed tile op can never reach the Target lowering.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Dialect/Tile/TileDialect.h"

#include "mlir/IR/BuiltinTypes.h"

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

} // namespace tile
} // namespace tessera
