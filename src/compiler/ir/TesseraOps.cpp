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
