#ifndef TESSERA_IR_ATTENTION_CHECKPOINT_STORAGE_H
#define TESSERA_IR_ATTENTION_CHECKPOINT_STORAGE_H

#include <cassert>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
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

// One storage rule for the registered semantic operation and Schedule exporter.
// Q/K/V are homogeneous; bias and LSE retain f32 independently of the output.
// The paired backward ABI is still its existing f32 contract.
inline mlir::FailureOr<NativeCheckpointStorage>
resolveNativeCheckpointStorage(mlir::Operation *op, bool backward) {
  using namespace mlir;
  unsigned base = backward ? 1 : 0;
  if (op->getNumOperands() < base + 3 ||
      (!backward && op->getNumResults() != 2) ||
      (backward && op->getNumResults() != 3 && op->getNumResults() != 4))
    return op->emitError("checkpoint storage requires complete input/result roles"), failure();
  for (auto [index, value] : llvm::enumerate(op->getOperands())) {
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!tensor || tensor.getEncoding())
      return op->emitError("checkpoint storage requires unencoded ranked tensors"), failure();
    Type element = tensor.getElementType();
    bool data = !backward && index < 3;
    if (data ? (!element.isF16() && !element.isBF16() && !element.isF32())
             : !element.isF32())
      return op->emitError("checkpoint Q/K/V forward storage requires f16/bf16/f32; auxiliary and backward operands require f32"), failure();
  }
  auto query = cast<RankedTensorType>(op->getOperand(base).getType());
  Type input = query.getElementType();
  for (unsigned index : {base + 1, base + 2})
    if (cast<RankedTensorType>(op->getOperand(index).getType()).getElementType() != input)
      return op->emitError("checkpoint Q/K/V storage must agree"), failure();
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!output || output.getEncoding() ||
      (!output.getElementType().isF32() && output.getElementType() != input) ||
      (backward && !output.getElementType().isF32()))
    return op->emitError("checkpoint output storage must be f32 or match Q/K/V"), failure();
  if (!backward) {
    auto lse = dyn_cast<RankedTensorType>(op->getResult(1).getType());
    if (!lse || lse.getEncoding() || !lse.getElementType().isF32())
      return op->emitError("checkpoint row LSE storage must be f32"), failure();
  }
  if (!input.isF32() || !output.getElementType().isF32()) {
    auto module = op->getParentOfType<ModuleOp>();
    auto target = module ? module->getAttrOfType<StringAttr>("tessera.target") : StringAttr();
    auto arch = module ? module->getAttrOfType<StringAttr>("tessera.arch") : StringAttr();
    if (backward || !target || target.getValue() != "nvidia_sm120" ||
        !arch || arch.getValue() != "sm_120")
      return op->emitError("half checkpoint forward storage requires the SM120 native contract"), failure();
  }
  return NativeCheckpointStorage{input, output.getElementType()};
}
} // namespace tessera
#endif

