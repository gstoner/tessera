#pragma once
#include "Tessera/IR/TensorShapeRelations.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include <climits>

namespace tessera::native_tensor_program {

// Logical capacities and variable identities are separate: the same trace
// extent can inhabit multiple independent classes with different capacities.
struct TensorShapeProfile {
  struct Symbol { unsigned argument, axis; int64_t capacity; };
  llvm::SmallVector<Symbol> variables;
  llvm::SmallVector<mlir::Value> values;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<int64_t>> capacities, symbols;

  static mlir::FailureOr<TensorShapeProfile> analyze(mlir::func::FuncOp function,
                                                   mlir::DictionaryAttr bounds) {
    using namespace mlir;
    if (!bounds || bounds.empty() || !function.getBody().hasOneBlock()) return failure();
    SmallVector<Operation *> operations;
    for (Operation &op : function.getBody().front().without_terminator()) operations.push_back(&op);
    if (operations.empty() || operations.back()->getName().getStringRef() != "tessera.matmul")
      return failure();
    auto returned = dyn_cast<func::ReturnOp>(function.getBody().front().getTerminator());
    if (!returned || function.getNumResults() != 1 || returned.getNumOperands() != 1 ||
        operations.back()->getNumResults() != 1 || operations.back()->getNumOperands() != 2 ||
        returned.getOperand(0) != operations.back()->getResult(0)) return failure();
    TensorShapeProfile profile;
    TensorShapeRelations relations;
    for (Value value : function.getArguments()) profile.values.push_back(value);
    for (Operation *op : operations) {
      if (op->getNumResults() != 1) return failure();
      profile.values.push_back(op->getResult(0));
    }
    for (Value value : profile.values) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || type.getRank() != 2 || type.getEncoding() || failed(relations.addValue(value)))
        return failure();
    }
    for (Operation *op : operations)
      if (failed(relations.addOperation(op))) return failure();
    Operation *product = operations.back();
    auto a = product->getAttrOfType<BoolAttr>("transposeA");
    auto b = product->getAttrOfType<BoolAttr>("transposeB");
    unsigned m = a && a.getValue() ? 1 : 0, n = b && b.getValue() ? 0 : 1;
    auto mc = relations.getClass(product->getOperand(0), m);
    auto nc = relations.getClass(product->getOperand(1), n);
    auto kc = relations.getClass(product->getOperand(0), 1 - m);
    if (failed(mc) || failed(nc) || failed(kc)) return failure();
    llvm::DenseMap<StringRef, unsigned> roles{{"M", *mc}, {"N", *nc}, {"K", *kc}};
    llvm::DenseMap<unsigned, int64_t> declared, fixed, capacity;
    for (NamedAttribute field : bounds) {
      auto role = roles.find(field.getName().getValue());
      auto value = dyn_cast<IntegerAttr>(field.getValue());
      if (role == roles.end() || !value || !value.getType().isInteger(64) ||
          value.getInt() <= 0 || value.getInt() > INT32_MAX) return failure();
      auto found = declared.find(role->second);
      if (found != declared.end() && found->second != value.getInt()) return failure();
      declared[role->second] = value.getInt();
    }
    // Unrequested terminal roles remain fixed, even if semantic constraints
    // equate them with another role. Never silently widen the caller's policy.
    for (auto role : roles)
      if (!bounds.get(role.first) && declared.contains(role.second)) return failure();
    for (auto dimension : relations.getDimensions()) {
      unsigned group = *relations.getClass(dimension.tensor, dimension.axis);
      auto type = cast<RankedTensorType>(dimension.tensor.getType());
      int64_t extent = type.getDimSize(dimension.axis);
      if (ShapedType::isDynamic(extent)) continue;
      if (extent <= 0 || extent > INT32_MAX) return failure();
      if (declared.contains(group)) {
        if (extent > declared.lookup(group)) return failure();
      } else {
        auto known = fixed.find(group);
        if (known != fixed.end() && known->second != extent) return failure();
        fixed[group] = extent;
      }
    }
    for (auto item : fixed) capacity[item.first] = item.second;
    for (auto item : declared) capacity[item.first] = item.second;
    llvm::DenseMap<unsigned, int64_t> identities;
    for (auto [index, value] : llvm::enumerate(function.getArguments()))
      for (unsigned axis = 0; axis < 2; ++axis) {
        unsigned group = *relations.getClass(value, axis);
        if (!declared.contains(group) || identities.contains(group)) continue;
        identities[group] = profile.variables.size();
        profile.variables.push_back({unsigned(index), axis, declared.lookup(group)});
      }
    if (identities.size() != declared.size()) return failure();
    for (Value value : profile.values) {
      for (unsigned axis = 0; axis < 2; ++axis) {
        unsigned group = *relations.getClass(value, axis);
        if (!capacity.contains(group)) return failure();
        profile.capacities[value].push_back(capacity.lookup(group));
        auto id = identities.find(group);
        profile.symbols[value].push_back(id == identities.end() ? -1 : id->second);
      }
    }
    return profile;
  }

  void projectTypes(mlir::func::FuncOp function) const {
    using namespace mlir;
    for (Value value : values) {
      auto type = cast<RankedTensorType>(value.getType());
      SmallVector<int64_t> shape(capacities.find(value)->second);
      auto bindings = symbols.find(value)->second;
      for (unsigned axis = 0; axis < shape.size(); ++axis)
        if (bindings[axis] >= 0) shape[axis] = ShapedType::kDynamic;
      value.setType(RankedTensorType::get(shape, type.getElementType()));
    }
    auto ret = cast<func::ReturnOp>(function.getBody().front().back());
    SmallVector<Type> arguments;
    for (Value value : function.getArguments()) arguments.push_back(value.getType());
    function.setType(FunctionType::get(function.getContext(), arguments, ret.getOperandTypes()));
  }
};
} // namespace tessera::native_tensor_program
