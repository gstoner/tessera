#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>

namespace tessera::native_tensor_program {

// Dimension identities come from typed semantic edges, never equal trace
// extents. Contraction equality is retained independently of output rows/cols.
// This analysis has no target, launch geometry or runtime ABI dependency.
class TensorShapeRelations {
public:
  struct Dimension {
    mlir::Value tensor;
    unsigned axis;
  };

  mlir::LogicalResult addValue(mlir::Value value) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    if (!type) return mlir::failure();
    for (unsigned axis = 0; axis < unsigned(type.getRank()); ++axis) {
      auto key = std::make_pair(value, axis);
      if (indices.contains(key)) continue;
      unsigned index = dimensions.size();
      indices[key] = index;
      dimensions.push_back({value, axis});
      parents.push_back(index);
    }
    return mlir::success();
  }

  mlir::LogicalResult addOperation(mlir::Operation *op) {
    using namespace mlir;
    if (op->getNumResults() != 1) return failure();
    for (Value value : op->getOperands())
      if (failed(addValue(value))) return failure();
    if (failed(addValue(op->getResult(0)))) return failure();
    auto output = cast<RankedTensorType>(op->getResult(0).getType());
    auto name = op->getName().getStringRef();
    if (name == "tessera.matmul") {
      if (op->getNumOperands() != 2 || output.getRank() != 2) return failure();
      auto lhs = cast<RankedTensorType>(op->getOperand(0).getType());
      auto rhs = cast<RankedTensorType>(op->getOperand(1).getType());
      if (lhs.getRank() != 2 || rhs.getRank() != 2) return failure();
      auto a = op->getAttrOfType<BoolAttr>("transposeA");
      auto b = op->getAttrOfType<BoolAttr>("transposeB");
      if ((op->hasAttr("transposeA") && !a) ||
          (op->hasAttr("transposeB") && !b)) return failure();
      unsigned m = a && a.getValue() ? 1 : 0;
      unsigned n = b && b.getValue() ? 0 : 1;
      if (failed(unite(op->getResult(0), 0, op->getOperand(0), m)) ||
          failed(unite(op->getResult(0), 1, op->getOperand(1), n)) ||
          failed(unite(op->getOperand(0), 1 - m, op->getOperand(1), 1 - n)))
        return failure();
      return success();
    }
    bool binary = name == "tessera.add";
    if ((!binary && name != "tessera.relu" && name != "tessera.rmsnorm" &&
         name != "tessera.layer_norm" && name != "tessera.softmax") ||
        op->getNumOperands() != (binary ? 2u : 1u)) return failure();
    for (Value operand : op->getOperands()) {
      auto input = cast<RankedTensorType>(operand.getType());
      if (input.getRank() != output.getRank()) return failure();
      for (unsigned axis = 0; axis < unsigned(output.getRank()); ++axis)
        if (failed(unite(operand, axis, op->getResult(0), axis))) return failure();
    }
    return success();
  }

  bool equivalent(mlir::Value a, unsigned axisA,
                  mlir::Value b, unsigned axisB) const {
    auto left = indices.find(std::make_pair(a, axisA));
    auto right = indices.find(std::make_pair(b, axisB));
    return left != indices.end() && right != indices.end() &&
           root(left->second) == root(right->second);
  }

  mlir::FailureOr<unsigned> getClass(mlir::Value value, unsigned axis) const {
    auto found = indices.find(std::make_pair(value, axis));
    if (found == indices.end()) return mlir::failure();
    return root(found->second);
  }

  llvm::ArrayRef<Dimension> getDimensions() const { return dimensions; }

private:
  unsigned root(unsigned index) const {
    while (parents[index] != index) index = parents[index];
    return index;
  }

  mlir::LogicalResult unite(mlir::Value a, unsigned axisA,
                            mlir::Value b, unsigned axisB) {
    auto leftType = mlir::cast<mlir::RankedTensorType>(a.getType());
    auto rightType = mlir::cast<mlir::RankedTensorType>(b.getType());
    int64_t left = leftType.getDimSize(axisA), right = rightType.getDimSize(axisB);
    if (!mlir::ShapedType::isDynamic(left) && !mlir::ShapedType::isDynamic(right) &&
        left != right) return mlir::failure();
    unsigned x = root(indices.lookup(std::make_pair(a, axisA)));
    unsigned y = root(indices.lookup(std::make_pair(b, axisB)));
    if (x != y) parents[y] = x;
    return mlir::success();
  }

  llvm::DenseMap<std::pair<mlir::Value, unsigned>, unsigned> indices;
  llvm::SmallVector<Dimension> dimensions;
  llvm::SmallVector<unsigned> parents;
};

} // namespace tessera::native_tensor_program
