#ifndef TESSERA_IR_TRANSPOSEUTILS_H
#define TESSERA_IR_TRANSPOSEUTILS_H
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace tessera {
// Only a plain permutation can be removed or composed. Unknown metadata may
// carry a layout/policy obligation and must reach its consumer unchanged.
inline std::optional<llvm::SmallVector<int64_t>> plainTransposePermutation(
    mlir::Operation *op) {
  if (!op || op->getName().getStringRef() != "tessera.transpose" ||
      op->getNumOperands() != 1 || op->getNumResults() != 1) return std::nullopt;
  auto input = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto output = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!input || !output || input.getRank() != output.getRank() ||
      input.getElementType() != output.getElementType()) return std::nullopt;
  for (auto attr : op->getAttrs())
    if (attr.getName() != "permutation") return std::nullopt;
  llvm::SmallVector<int64_t> perm;
  if (auto attr = op->getAttr("permutation")) {
    auto dense = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr);
    if (!dense) return std::nullopt;
    perm.append(dense.asArrayRef().begin(), dense.asArrayRef().end());
  } else {
    for (int64_t i = input.getRank(); i > 0; --i) perm.push_back(i - 1);
  }
  if (perm.size() != static_cast<size_t>(input.getRank())) return std::nullopt;
  llvm::SmallVector<bool> seen(perm.size(), false);
  for (size_t i = 0; i < perm.size(); ++i) {
    auto axis = perm[i];
    if (axis < 0 || axis >= input.getRank() || seen[axis]) return std::nullopt;
    seen[axis] = true;
    auto a = input.getDimSize(axis), b = output.getDimSize(i);
    if (!mlir::ShapedType::isDynamic(a) && !mlir::ShapedType::isDynamic(b) && a != b)
      return std::nullopt;
  }
  return perm;
}
inline bool isMatrixTranspose(mlir::Operation *op) {
  auto p = plainTransposePermutation(op);
  return p && p->size() == 2 && (*p)[0] == 1 && (*p)[1] == 0;
}
} // namespace tessera
#endif
