#include "Tessera/IR/TesseraOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace tessera {

#include "TesseraOpsEnums.cpp.inc"

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
    double p = dropout->getValueAsDouble();
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

#define GET_OP_CLASSES
#include "TesseraOps.cpp.inc"

} // namespace tessera
