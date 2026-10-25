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
  // Q storage. K and V share kvElement, which differs from Q only on the
  // forward role (SM120-SAVED-LSE-MIXED-STORAGE-20261025).
  mlir::Type inputElement;
  mlir::Type outputElement;
  mlir::Type kvElement;
  bool mixed() const { return kvElement != inputElement; }
};

inline llvm::StringRef checkpointStorageSpelling(mlir::Type element) {
  if (element.isF16()) return "f16";
  if (element.isBF16()) return "bf16";
  assert(element.isF32() && "checkpoint storage must have been verified");
  return "f32";
}

// Backward Q/K/V and their gradients use homogeneous storage. Forward K/V
// share one storage that may differ from Q; accumulation is the declared f32
// and each operand widens to it exactly. Saved output and dO share
// independently declared storage (f32 or Q's); bias, row LSE and their seeds
// are f32.
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
  Type kv = operands[base + 1].getElementType();
  Type output = backward ? operands[4].getElementType() : results[0].getElementType();
  auto supported = [](Type type) { return type.isF16() || type.isBF16() || type.isF32(); };
  if (!supported(input) || !supported(kv))
    return op->emitError("checkpoint Q/K/V storage requires f16/bf16/f32"), failure();
  if (output != input && !output.isF32())
    return op->emitError("checkpoint output storage must be f32 or match Q"), failure();
  if (operands[base + 2].getElementType() != kv)
    return op->emitError("checkpoint K/V storage must agree"), failure();
  if (backward && kv != input)
    return op->emitError("checkpoint backward requires homogeneous Q/K/V storage"), failure();
  for (auto [index, tensor] : llvm::enumerate(operands)) {
    Type expected = index == base ? input : index > base && index < base + 3 ? kv :
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
  if (!input.isF32() || !output.isF32() || !kv.isF32()) {
    auto module = op->getParentOfType<ModuleOp>();
    auto target = module ? module->getAttrOfType<StringAttr>("tessera.target") : StringAttr();
    auto arch = module ? module->getAttrOfType<StringAttr>("tessera.arch") : StringAttr();
    if (!target || target.getValue() != "nvidia_sm120" || !arch || arch.getValue() != "sm_120")
      return op->emitError("half checkpoint storage requires the SM120 native contract"), failure();
  }
  return NativeCheckpointStorage{input, output, kv};
}
} // namespace tessera
#endif

