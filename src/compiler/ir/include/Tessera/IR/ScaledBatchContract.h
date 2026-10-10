#ifndef TESSERA_IR_SCALED_BATCH_CONTRACT_H
#define TESSERA_IR_SCALED_BATCH_CONTRACT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include <algorithm>

// Logical prefix broadcasting is independent for all four operands. Matrix
// and scale suffixes are checked by the owning scaled-product verifier.
namespace tessera {
inline bool hasExactScaledBroadcastPrefix(
    llvm::ArrayRef<mlir::RankedTensorType> operands,
    mlir::RankedTensorType result) {
  if (!result || result.getRank() < 2 || !result.hasStaticShape())
    return false;
  int64_t prefix = result.getRank() - 2, maximum = 0;
  for (auto type : operands) {
    if (!type || type.getRank() < 2 || !type.hasStaticShape())
      return false;
    maximum = std::max(maximum, int64_t(type.getRank() - 2));
    for (int64_t extent : type.getShape())
      if (extent <= 0) return false;
  }
  if (maximum != prefix) return false;
  for (int64_t axis = 0; axis < prefix; ++axis) {
    int64_t joined = 1;
    for (auto type : operands) {
      int64_t own = type.getRank() - 2, aligned = axis - (prefix - own);
      int64_t extent = aligned < 0 ? 1 : type.getDimSize(aligned);
      if (extent != 1 && joined != 1 && joined != extent) return false;
      joined = std::max(joined, extent);
    }
    if (result.getDimSize(axis) != joined) return false;
  }
  return true;
}
// A rank-two typed product may need the same bounded physical plane contract
// as an independent prefix. This derives a lowering profile, not Graph batching.
inline bool needsScalarScaledPlane(mlir::Operation *op) {
  using namespace mlir;
  if (op->getName().getStringRef() != "tessera.scaled_matmul" ||
      op->getNumOperands() != 4 || op->getNumResults() != 1 ||
      op->hasAttr("batching") || op->hasAttr("physical_contract"))
    return false;
  for (Value value : op->getOperands()) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || type.getRank() != 2 || !type.hasStaticShape())
      return false;
  }
  auto a = cast<RankedTensorType>(op->getOperand(0).getType());
  if (!isa<Float8E4M3FNType>(a.getElementType())) return false;
  auto orientation = op->getAttrOfType<BoolAttr>("transposeA");
  bool ta = orientation && orientation.getValue();
  auto layout = op->getAttrOfType<DictionaryAttr>("scale_layout");
  auto block = layout ? layout.getAs<ArrayAttr>("block") : ArrayAttr{};
  auto group = block && block.size() == 2 ? dyn_cast<IntegerAttr>(block[1]) : IntegerAttr{};
  int64_t k = a.getDimSize(ta ? 0 : 1);
  return group && group.getInt() > 0 && k > 0 &&
         (ta || k % group.getInt() != 0);
}
} // namespace tessera
#endif
