#include "Tessera/IR/TesseraOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringSwitch.h"

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

LogicalResult Conv2DNHWCOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto filterType = dyn_cast<RankedTensorType>(getFilter().getType());
  if (!inputType || !filterType)
    return success();
  if (inputType.getRank() != 4 || filterType.getRank() != 4)
    return emitOpError("expects NHWC input and HWCF filter rank-4 tensors");
  return success();
}

LogicalResult FlashAttnOp::verify() {
  if (getHeadDim() <= 0)
    return emitOpError("head_dim must be positive");
  if (auto dropout = getDropoutP()) {
    double p = dropout->convertToDouble();
    if (p < 0.0 || p >= 1.0)
      return emitOpError("dropout_p must satisfy 0 <= p < 1");
  }
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

LogicalResult DropoutOp::verify() { return success(); }
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
