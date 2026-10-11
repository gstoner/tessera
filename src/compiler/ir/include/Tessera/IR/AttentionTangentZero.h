#ifndef TESSERA_IR_ATTENTION_TANGENT_ZERO_H
#define TESSERA_IR_ATTENTION_TANGENT_ZERO_H
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
namespace tessera {
// Symbolic inactive directions retain the exact source tensor extent.
// Native attention export consumes this proven zero region without a
// host-filled tensor or an allocation at the capacity shape.
inline mlir::Value buildAttentionDynamicZero(mlir::OpBuilder &b,
                                             mlir::Location loc,
                                             mlir::Value source) {
  using namespace mlir;
  auto type = dyn_cast<RankedTensorType>(source.getType());
  if (!type || type.hasStaticShape() || type.getEncoding() ||
      !type.getElementType().isF32()) return {};
  SmallVector<Value> extents;
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    if (type.isDynamicDim(axis))
      extents.push_back(tensor::DimOp::create(b, loc, source, axis));
  return tensor::GenerateOp::create(b, loc, type, extents,
      [](OpBuilder &body, Location location, ValueRange) {
        Value zero = arith::ConstantOp::create(body, location,
                                               body.getF32FloatAttr(0));
        tensor::YieldOp::create(body, location, zero);
      }).getResult();
}
inline bool isAttentionZeroLike(mlir::Value value, mlir::Value source) {
  using namespace mlir;
  if (!value || value.getType() != source.getType()) return false;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto dense = dyn_cast<DenseFPElementsAttr>(constant.getValue());
    return dense && dense.isSplat() && dense.getSplatValue<APFloat>().isZero();
  }
  auto generated = value.getDefiningOp<tensor::GenerateOp>();
  auto type = dyn_cast<RankedTensorType>(source.getType());
  if (!generated || !type || type.hasStaticShape() ||
      !generated.getBody().hasOneBlock()) return false;
  auto &block = generated.getBody().front();
  if (block.getOperations().size() != 2) return false;
  auto zero = dyn_cast<arith::ConstantOp>(block.front());
  auto yield = dyn_cast<tensor::YieldOp>(block.back());
  auto number = zero ? dyn_cast<FloatAttr>(zero.getValue()) : FloatAttr();
  if (!number || !number.getType().isF32() || !number.getValue().isZero() ||
      !yield || yield.getValue() != zero.getResult()) return false;
  unsigned position = 0;
  for (int64_t axis = 0; axis < type.getRank(); ++axis) {
    if (!type.isDynamicDim(axis)) continue;
    if (position >= generated.getDynamicExtents().size()) return false;
    auto dim = generated.getDynamicExtents()[position++].getDefiningOp<tensor::DimOp>();
    if (!dim || dim.getSource() != source || dim.getConstantIndex() != axis)
      return false;
  }
  return position == generated.getDynamicExtents().size();
}
} // namespace tessera
#endif
