#ifndef TESSERA_IR_ATTENTION_CHECKPOINT_STORAGE_H
#define TESSERA_IR_ATTENTION_CHECKPOINT_STORAGE_H

#include <cassert>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace tessera {
struct NativeCheckpointStorage {
  mlir::Type inputElement;
  mlir::Type outputElement;
};

inline llvm::StringRef checkpointStorageSpelling(mlir::Type element) {
  if (element.isF16()) return "f16";
  if (element.isBF16()) return "bf16";
  assert(element.isF32() && "checkpoint storage must have been verified");
  return "f32";
}

// Q/K/V and their gradients use homogeneous storage. Saved output and dO
// share independently declared storage; bias, row LSE and their seeds are f32.
inline mlir::FailureOr<NativeCheckpointStorage>
resolveNativeCheckpointStorage(mlir::Operation *op, bool backward) {
  using namespace mlir;
  unsigned base = backward ? 1 : 0;
  if (op->getNumOperands() < (backward ? 6u : 3u) ||
      (!backward && op->getNumResults() != 2) ||
      (backward && op->getNumResults() != 3 && op->getNumResults() != 4))
    return op->emitError("checkpoint storage requires complete input/result roles"), failure();
  SmallVector<RankedTensorType> operands, results;
  for (Value value : op->getOperands()) {
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!tensor || tensor.getEncoding())
      return op->emitError("checkpoint storage requires unencoded ranked tensors"), failure();
    operands.push_back(tensor);
  }
  for (Value value : op->getResults()) {
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!tensor || tensor.getEncoding())
      return op->emitError("checkpoint storage requires unencoded ranked results"), failure();
    results.push_back(tensor);
  }
  Type input = operands[base].getElementType();
  Type output = backward ? operands[4].getElementType() : results[0].getElementType();
  if (!input.isF16() && !input.isBF16() && !input.isF32())
    return op->emitError("checkpoint Q/K/V storage requires f16/bf16/f32"), failure();
  if (output != input && !output.isF32())
    return op->emitError("checkpoint output storage must be f32 or match Q/K/V"), failure();
  for (unsigned index : {base + 1, base + 2})
    if (operands[index].getElementType() != input)
      return op->emitError("checkpoint Q/K/V storage must agree"), failure();
  for (auto [index, tensor] : llvm::enumerate(operands)) {
    Type expected = index >= base && index < base + 3 ? input :
        backward && (index == 0 || index == 4) ? output : Float32Type::get(op->getContext());
    if (tensor.getElementType() != expected)
      return op->emitError("checkpoint auxiliary and backward seed storage disagrees with its typed role"), failure();
  }
  for (auto [index, tensor] : llvm::enumerate(results)) {
    Type expected = backward ? (index < 3 ? input : Type(Float32Type::get(op->getContext()))) :
        (index == 0 ? output : Type(Float32Type::get(op->getContext())));
    if (tensor.getElementType() != expected) {
      if (!backward && index == 1)
        return op->emitError("checkpoint row LSE storage must be f32"), failure();
      if (backward && index == 3)
        return op->emitError("checkpoint bias gradient result storage must be f32"), failure();
      return op->emitError("checkpoint result storage disagrees with its gradient/output/f32 auxiliary role"), failure();
    }
  }
  if (!input.isF32() || !output.isF32()) {
    auto module = op->getParentOfType<ModuleOp>();
    auto target = module ? module->getAttrOfType<StringAttr>("tessera.target") : StringAttr();
    auto arch = module ? module->getAttrOfType<StringAttr>("tessera.arch") : StringAttr();
    if (!target || target.getValue() != "nvidia_sm120" || !arch || arch.getValue() != "sm_120")
      return op->emitError("half checkpoint storage requires the SM120 native contract"), failure();
  }
  return NativeCheckpointStorage{input, output};
}
} // namespace tessera
#endif

