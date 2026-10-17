#pragma once
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <optional>
#include <utility>

namespace tessera::slice_contract {
struct StaticBounds {
  llvm::SmallVector<int64_t> offsets;
  llvm::SmallVector<int64_t> sizes;
};
// Public slice and dynamic_slice both mean x[start:start + size], with unit
// stride. The end coordinate must not wrap when two signed i64 attributes add.
inline std::pair<int64_t, int64_t> normalizeAxis(int64_t extent, int64_t start,
                                                int64_t size) {
  const llvm::APInt n(65, uint64_t(extent));
  auto clip = [&](llvm::APInt value) -> int64_t {
    if (value.isNegative()) value += n;
    if (value.isNegative()) return 0;
    if (value.sgt(n)) return extent;
    return value.getSExtValue();
  };
  const llvm::APInt first(65, uint64_t(start), true);
  const llvm::APInt length(65, uint64_t(size), true);
  int64_t offset = clip(first), end = clip(first + length);
  return {offset, std::max<int64_t>(0, end - offset)};
}
inline std::optional<StaticBounds> staticBounds(mlir::Operation *op) {
  auto source = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto starts = op->getAttrOfType<mlir::ArrayAttr>("start_indices");
  auto sizes = op->getAttrOfType<mlir::ArrayAttr>("slice_sizes");
  if (!source || !source.hasStaticShape() || !starts || !sizes ||
      starts.size() != size_t(source.getRank()) || sizes.size() != starts.size())
    return std::nullopt;
  StaticBounds result;
  for (int64_t axis = 0; axis < source.getRank(); ++axis) {
    auto start = mlir::dyn_cast<mlir::IntegerAttr>(starts[axis]);
    auto size = mlir::dyn_cast<mlir::IntegerAttr>(sizes[axis]);
    if (!start || !size || start.getValue().getBitWidth() != 64 ||
        size.getValue().getBitWidth() != 64) return std::nullopt;
    auto [offset, count] = normalizeAxis(source.getDimSize(axis), start.getInt(), size.getInt());
    result.offsets.push_back(offset); result.sizes.push_back(count);
  }
  return result;
}
inline mlir::LogicalResult verify(mlir::Operation *op) {
  auto source = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto result = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto starts = op->getAttrOfType<mlir::ArrayAttr>("start_indices");
  auto sizes = op->getAttrOfType<mlir::ArrayAttr>("slice_sizes");
  if (!source || !result || source.getRank() != result.getRank())
    return op->emitOpError("requires ranked source and result with equal rank");
  if (!starts || !sizes || starts.size() != size_t(source.getRank()) ||
      sizes.size() != starts.size())
    return op->emitOpError("start_indices and slice_sizes must match input rank");
  for (int64_t axis = 0; axis < source.getRank(); ++axis) {
    auto start = mlir::dyn_cast<mlir::IntegerAttr>(starts[axis]);
    auto size = mlir::dyn_cast<mlir::IntegerAttr>(sizes[axis]);
    if (!start || !size || start.getValue().getBitWidth() != 64 || size.getValue().getBitWidth() != 64)
      return op->emitOpError("slice coordinates require signed i64 attributes");
    if (source.isDynamicDim(axis)) {
      if (!result.isDynamicDim(axis))
        return op->emitOpError("dynamic source extent requires dynamic slice result extent");
      continue;
    }
    auto bounds = normalizeAxis(source.getDimSize(axis), start.getInt(), size.getInt());
    if (!result.isDynamicDim(axis) && result.getDimSize(axis) != bounds.second)
      return op->emitOpError("result dimension disagrees with normalized slice extent at axis ") << axis;
  }
  return mlir::success();
}
} // namespace tessera::slice_contract
