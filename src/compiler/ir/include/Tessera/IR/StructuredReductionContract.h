#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/SHA256.h"
#include "llvm/ADT/StringExtras.h"

namespace tessera {
inline std::string structuredReductionBodyHash(mlir::Operation *body) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  body->print(stream, mlir::OpPrintingFlags().useLocalScope());
  stream.flush();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}
inline mlir::LogicalResult verifyStructuredReductionCarrier(mlir::Operation *op) {
  using namespace mlir;
  auto arch = op->getAttrOfType<StringAttr>("arch");
  auto name = op->getAttrOfType<StringAttr>("name");
  auto count = op->getAttrOfType<IntegerAttr>("count");
  auto inputs = op->getAttrOfType<IntegerAttr>("input_count");
  auto threads = op->getAttrOfType<IntegerAttr>("workgroup_size");
  auto algorithm = op->getAttrOfType<StringAttr>("algorithm");
  bool wave = algorithm && algorithm.getValue() == "wave_per_scale_element";
  bool carrier = algorithm && algorithm.getValue() == "serial_tensor_carrier";
  auto digest = op->getAttrOfType<StringAttr>("artifact_hash");
  auto bodyHash = op->getAttrOfType<StringAttr>("body_hash");
  if (!arch || arch.getValue() != "gfx1201" || !name || name.getValue().empty() ||
      !count || count.getInt() <= 0 || count.getInt() > INT32_MAX ||
      !inputs || inputs.getInt() != (carrier ? 1 : 4) || !threads || !algorithm ||
      (!wave && !carrier && algorithm.getValue() != "serial_per_scale_element") ||
      threads.getInt() != (wave ? 32 : 128) ||
      !digest || digest.getValue().size() != 64 || !bodyHash ||
      op->getNumRegions() != 1 || !op->getRegion(0).hasOneBlock() ||
      op->getRegion(0).front().getNumArguments() != 0 ||
      op->getRegion(0).front().getOperations().size() != 1)
    return op->emitOpError("requires one gfx1201 structured reduction body and checked algorithm/thread ABI");
  auto &module = op->getRegion(0).front().front();
  if (module.getName().getStringRef() != "gpu.module" ||
      module.getNumRegions() != 1 || !module.getRegion(0).hasOneBlock() ||
      module.getRegion(0).front().getOperations().size() != 1 ||
      structuredReductionBodyHash(&module) != bodyHash.getValue())
    return op->emitOpError("structured reduction GPU body differs from its sealed Tile contract");
  auto &kernel = module.getRegion(0).front().front();
  auto gpuKernel = dyn_cast<gpu::GPUFuncOp>(&kernel);
  auto type = gpuKernel ? gpuKernel.getFunctionType() : FunctionType{};
  if (!gpuKernel || !gpuKernel.isKernel() ||
      gpuKernel.getName() != name.getValue() ||
      !type || type.getNumInputs() != inputs.getInt()+2 || type.getNumResults() != 0 ||
      !type.getInput(inputs.getInt()+1).isInteger(64))
    return op->emitOpError("structured reduction kernel lost its native buffer/scalar ABI");
  unsigned bytes = 0, floats = 0;
  for (unsigned i = 0; i < unsigned(inputs.getInt()+1); ++i) {
    auto memref = dyn_cast<MemRefType>(type.getInput(i));
    if (!memref || memref.getShape() != ArrayRef<int64_t>{ShapedType::kDynamic} ||
        !memref.getLayout().isIdentity() || memref.getMemorySpaceAsInt() != 0)
      return op->emitOpError("structured reduction requires flattened contiguous buffers");
    if (i == inputs.getInt() && !memref.getElementType().isF32())
      return op->emitOpError("structured reduction output must be f32");
    if (i < inputs.getInt()) {
      bytes += memref.getElementType().isSignlessInteger(8);
      floats += memref.getElementType().isF32();
    }
  }
  if (!(carrier ? (floats==1 && bytes==0 && !wave) :
        ((bytes == 2 && floats == 2) || (floats == 4 && !wave))))
    return op->emitOpError("structured scaled reduction requires checked E4M3/f32 or serial four-f32 inputs");
  return success();
}
} // namespace tessera
