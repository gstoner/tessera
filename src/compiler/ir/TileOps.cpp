//===- TileOps.cpp - Tessera Tile IR op verifiers -------------*- C++ -*-===//
//
// Sprint 9: rank/shape verifiers for the contraction tile ops. These mirror the
// Graph IR MatmulOp / BatchedGemmOp verifiers — the value lane only ever
// produces in-envelope tile ops, and this re-checks at the Tile layer so a
// malformed tile op can never reach the Target lowering.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Dialect/Tile/TileDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

TileMmaDescAttr mmaDescAttr(Operation *op) {
  return op->getAttrOfType<TileMmaDescAttr>("mma");
}

LogicalResult requireFragmentProducer(Operation *op, Value value,
                                      llvm::StringRef expectedRole,
                                      TileMmaDescAttr expectedDesc) {
  if (!isa<FragmentType>(value.getType()))
    return op->emitOpError() << "expects !tile.fragment operands";
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return op->emitOpError() << "fragment operand must have a Tile producer";
  auto role = producer->getAttrOfType<StringAttr>("role");
  auto desc = mmaDescAttr(producer);
  if (!role || role.getValue() != expectedRole)
    return op->emitOpError() << "expects a fragment with role \"" << expectedRole
                             << "\"";
  if (!desc || desc != expectedDesc)
    return op->emitOpError() << "fragment descriptor must match tile.mma";
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

LogicalResult verifyTileControl(Operation *op, ValueRange inputs,
                                ValueRange outputs) {
  auto verifyStatic = [&](ValueRange values) -> LogicalResult {
    for (Value value : values) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || !type.hasStaticShape())
        return op->emitOpError(
            "NVIDIA control-flow Tile contract requires static ranked tensors");
    }
    return success();
  };
  if (failed(verifyStatic(inputs)) || failed(verifyStatic(outputs)))
    return failure();
  StringRef name = op->getName().getStringRef();
  if (!op->getAttrOfType<StringAttr>("source"))
    return op->emitOpError("requires a source Graph op attribute");

  if (name == "tile.control_for") {
    auto start = op->getAttrOfType<IntegerAttr>("start");
    auto stop = op->getAttrOfType<IntegerAttr>("stop");
    auto step = op->getAttrOfType<IntegerAttr>("step");
    if (!start || !stop || !step || step.getInt() == 0)
      return op->emitOpError("requires start/stop and a non-zero step");
    int64_t carry = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("carry_arg_index"))
      carry = attr.getInt();
    if (carry < 0 || carry >= static_cast<int64_t>(inputs.size()) ||
        outputs.size() != 1 || outputs[0].getType() != inputs[carry].getType())
      return op->emitOpError("requires one shape-stable carried result");
  } else if (name == "tile.control_if") {
    auto flag = op->getAttrOfType<IntegerAttr>("flag_arg_index");
    if (!flag || flag.getInt() < 0 ||
        flag.getInt() >= static_cast<int64_t>(inputs.size()) || outputs.empty())
      return op->emitOpError("requires an in-range flag and branch result");
  } else if (name == "tile.control_while") {
    auto maxIters = op->getAttrOfType<IntegerAttr>("max_iters");
    auto carry = op->getAttrOfType<IntegerAttr>("carry_arg_index");
    if (!maxIters || maxIters.getInt() <= 0 || !carry || carry.getInt() < 0 ||
        carry.getInt() >= static_cast<int64_t>(inputs.size()) ||
        outputs.size() != 1 ||
        outputs[0].getType() != inputs[carry.getInt()].getType())
      return op->emitOpError(
          "requires max_iters>0 and one shape-stable carried result");
  } else if (name == "tile.control_scan") {
    auto trip = op->getAttrOfType<IntegerAttr>("trip");
    if (!trip || trip.getInt() <= 0 || inputs.size() < 2 ||
        outputs.size() != 2 || outputs[0].getType() != inputs[0].getType())
      return op->emitOpError(
          "requires trip>0, init/xs, and shape-stable carry/ys results");
    auto xs = cast<RankedTensorType>(inputs[1].getType());
    auto ys = cast<RankedTensorType>(outputs[1].getType());
    if (xs.getRank() < 1 || ys.getRank() < 1 ||
        xs.getDimSize(0) != trip.getInt() ||
        ys.getDimSize(0) != trip.getInt())
      return op->emitOpError("xs/ys leading dimension must equal trip");
  }
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

LogicalResult ControlForOp::verify() {
  return verifyTileControl(getOperation(), getInputs(), getOutputs());
}

LogicalResult ControlIfOp::verify() {
  return verifyTileControl(getOperation(), getInputs(), getOutputs());
}

LogicalResult ControlWhileOp::verify() {
  return verifyTileControl(getOperation(), getInputs(), getOutputs());
}

LogicalResult ControlScanOp::verify() {
  return verifyTileControl(getOperation(), getInputs(), getOutputs());
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

LogicalResult EBMRefinementOp::verify() {
  auto inputs = getInputs();
  auto outputs = getOutputs();
  if (inputs.size() != 2 || outputs.size() != 1)
    return emitOpError("expects exactly 2 inputs and 1 result");
  auto y = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto grad = dyn_cast<RankedTensorType>(inputs[1].getType());
  auto out = dyn_cast<RankedTensorType>(outputs[0].getType());
  if (!y || !grad || !out)
    return success();
  if (!y.hasStaticShape() || !grad.hasStaticShape() || !out.hasStaticShape())
    return emitOpError("expects static fp32 operands and result");
  if (!y.getElementType().isF32() || !grad.getElementType().isF32() ||
      !out.getElementType().isF32())
    return emitOpError("expects fp32 operands and fp32 result");
  if (!sameStaticShape(y, grad) || !sameStaticShape(y, out))
    return emitOpError("y/grad/result shapes must match exactly");
  if (auto eta = getOperation()->getAttrOfType<FloatAttr>("eta");
      !eta || eta.getValueAsDouble() <= 0.0)
    return emitOpError("eta must be present and positive");
  if (auto steps = getOperation()->getAttrOfType<IntegerAttr>("steps");
      !steps || steps.getInt() <= 0)
    return emitOpError("steps must be present and positive");
  return success();
}

LogicalResult EBMPartitionExactOp::verify() {
  auto inputs = getInputs();
  auto outputs = getOutputs();
  if (inputs.size() != 1 || outputs.size() != 1)
    return emitOpError("expects exactly 1 input and 1 result");
  auto energies = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto out = dyn_cast<RankedTensorType>(outputs[0].getType());
  if (!energies || !out)
    return success();
  if (!energies.hasStaticShape() || !out.hasStaticShape())
    return emitOpError("expects static fp32 energies and scalar result");
  if (!energies.getElementType().isF32() || !out.getElementType().isF32())
    return emitOpError("expects fp32 energies and fp32 result");
  if (out.getRank() != 0)
    return emitOpError("partition result must be scalar for value execution");
  if (auto temperature = getOperation()->getAttrOfType<FloatAttr>("temperature");
      temperature && temperature.getValueAsDouble() <= 0.0)
    return emitOpError("temperature must be positive");
  if (auto reduction = getOperation()->getAttrOfType<StringAttr>("reduction");
      reduction && reduction.getValue() != "logsumexp")
    return emitOpError("only reduction=\"logsumexp\" is executable");
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

LogicalResult ViewOp::verify() {
  if (getInputs().empty())
    return emitOpError("expects a source and optional index operands");
  if (!getOperation()->getAttrOfType<TileLayoutAttr>("tile.layout"))
    return emitOpError("requires a #tile.layout attribute");
  if (auto memory = getOperation()->getAttrOfType<TileMemoryLayoutAttr>("tile.memory")) {
    if (getInputs().size() < 3)
      return emitOpError("pointer-backed tile.view requires base pointer plus row/column origins");
    (void)memory;
  }
  return success();
}

LogicalResult FragmentPackOp::verify() {
  if (getInputs().size() != 1 || !isa<TileValueType>(getInputs().front().getType()))
    return emitOpError("expects exactly one !tile.tile input");
  auto role = getOperation()->getAttrOfType<StringAttr>("role");
  if (!role || (role.getValue() != "a" && role.getValue() != "b" &&
                role.getValue() != "acc"))
    return emitOpError("requires role = a, b, or acc");
  if (!mmaDescAttr(getOperation()))
    return emitOpError("requires a #tile.mma_desc mma attribute");
  return success();
}

LogicalResult FragmentZeroOp::verify() {
  auto role = getOperation()->getAttrOfType<StringAttr>("role");
  if (!role || role.getValue() != "acc")
    return emitOpError("requires role = acc");
  if (!mmaDescAttr(getOperation()))
    return emitOpError("requires a #tile.mma_desc mma attribute");
  return success();
}

LogicalResult MMAOp::verify() {
  // Preserve the legacy permissive form during migration. Only the typed
  // fragment form is eligible for physical cooperative-matrix lowering.
  bool hasFragment = llvm::any_of(getInputs(), [](Value v) {
    return isa<FragmentType>(v.getType());
  });
  if (!hasFragment)
    return success();
  if (getInputs().size() != 3 || getOutputs().size() != 1 ||
      !isa<FragmentType>(getOutputs().front().getType()))
    return emitOpError(
        "typed fragment form expects A, B, accumulator -> !tile.fragment");
  auto desc = mmaDescAttr(getOperation());
  if (!desc)
    return emitOpError("typed fragment form requires a #tile.mma_desc mma attribute");
  if (failed(requireFragmentProducer(getOperation(), getInputs()[0], "a", desc)) ||
      failed(requireFragmentProducer(getOperation(), getInputs()[1], "b", desc)))
    return failure();
  Value accumulator = getInputs()[2];
  if (!isa<FragmentType>(accumulator.getType()))
    return emitOpError("accumulator must be a !tile.fragment");
  Operation *accProducer = accumulator.getDefiningOp();
  if (!accProducer || !mmaDescAttr(accProducer) || mmaDescAttr(accProducer) != desc)
    return emitOpError("accumulator descriptor must match tile.mma");
  auto role = accProducer->getAttrOfType<StringAttr>("role");
  if (role && role.getValue() != "acc")
    return emitOpError("accumulator fragment must have role acc");
  return success();
}

LogicalResult FragmentUnpackOp::verify() {
  if (getInputs().size() != 1 || !isa<FragmentType>(getInputs().front().getType()))
    return emitOpError("expects exactly one !tile.fragment input");
  Operation *producer = getInputs().front().getDefiningOp();
  if (!producer || !mmaDescAttr(producer))
    return emitOpError("fragment must be produced by a descriptor-carrying Tile op");
  auto role = producer->getAttrOfType<StringAttr>("role");
  if (role && role.getValue() != "acc")
    return emitOpError("only an accumulator fragment may be unpacked");
  auto desc = mmaDescAttr(getOperation());
  if (desc && desc != mmaDescAttr(producer))
    return emitOpError("mma attribute must match the input fragment descriptor");
  if (!getOperation()->getAttrOfType<TileLayoutAttr>("tile.layout"))
    return emitOpError("requires a #tile.layout attribute");
  return success();
}

LogicalResult StoreOp::verify() {
  if (getInputs().size() != 4 ||
      !isa<TileValueType>(getInputs().front().getType()))
    return emitOpError(
        "pointer-backed form expects tile, base, row origin, column origin");
  if (!getOperation()->getAttrOfType<TileLayoutAttr>("tile.layout"))
    return emitOpError("requires a #tile.layout attribute");
  if (!getOperation()->getAttrOfType<TileMemoryLayoutAttr>("tile.memory"))
    return emitOpError("requires a #tile.memory_layout attribute");
  return success();
}

LogicalResult MatmulKernelOp::verify() {
  auto desc = getOperation()->getAttrOfType<TileMmaDescAttr>("mma");
  auto epilogue = getOperation()->getAttrOfType<TileEpilogueAttr>("epilogue");
  if (!desc)
    return emitOpError("requires a #tile.mma_desc mma attribute");
  if (!epilogue)
    return emitOpError("requires a #tile.epilogue epilogue attribute");
  unsigned expected = epilogue.getBias() ? 7 : 6;
  if (getInputs().size() != expected)
    return emitOpError() << "expects A, B, "
                         << (epilogue.getBias() ? "bias, " : "")
                         << "D, M, N, K operands";
  unsigned dimStart = expected - 3;
  for (Value dim : getInputs().drop_front(dimStart))
    if (!dim.getType().isInteger(64))
      return emitOpError("M, N, and K must be i64");
  for (Value pointer : getInputs().take_front(dimStart))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError("A, B, optional bias, and D must be !llvm.ptr");
  int64_t warps = 1;
  if (auto attr = getOperation()->getAttrOfType<IntegerAttr>("warps"))
    warps = attr.getInt();
  if (warps != 1 && warps != 4)
    return emitOpError("warps must be 1 or 4");
  StringRef staging = "global";
  if (auto attr = getOperation()->getAttrOfType<StringAttr>("staging"))
    staging = attr.getValue();
  if (staging != "global" && staging != "shared")
    return emitOpError("staging must be global or shared");
  if (staging == "shared" && warps != 4)
    return emitOpError("shared staging currently requires four warps");
  return success();
}

} // namespace tile
} // namespace tessera
