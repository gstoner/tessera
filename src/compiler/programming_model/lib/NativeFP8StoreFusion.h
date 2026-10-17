// Single-use output conversion is a native store epilogue, not a host cast.
#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"

namespace tessera {
static mlir::LogicalResult fuseNativeFP8StoreCast(mlir::ModuleOp module) {
  using namespace mlir;
  constexpr llvm::StringLiteral key = "tessera.native.fp8_store_cast";
  if (!module->hasAttr(key)) return success();
  auto refuse = [&]() {
    return module.emitError("native FP8 store fusion requires one exact gfx1201 FP32 product and single-use BF16 cast");
  };
  auto target = module->getAttrOfType<StringAttr>("tessera.target");
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  SmallVector<func::FuncOp> roots;
  for (auto fn : module.getOps<func::FuncOp>())
    if (!fn.isDeclaration()) roots.push_back(fn);
  if (!isa<UnitAttr>(module->getAttr(key)) || !target ||
      target.getValue() != "rocm_gfx1201" || !arch || arch.getValue() != "gfx1201" ||
      roots.size() != 1 || !roots[0].getBody().hasOneBlock())
    return refuse();
  auto fn = roots[0];
  auto &body = fn.getBody().front();
  if (body.getOperations().size() != 3 || fn.getNumArguments() != 4 ||
      fn.getNumResults() != 1) return refuse();
  auto it = body.begin();
  Operation *product = &*it++;
  Operation *store = &*it++;
  auto ret = dyn_cast<func::ReturnOp>(&*it);
  if (product->getName().getStringRef() != "tessera.scaled_matmul" ||
      product->getNumOperands() != 4 || product->getNumResults() != 1 ||
      store->getName().getStringRef() != "tessera.cast" ||
      store->getNumOperands() != 1 || store->getNumResults() != 1 ||
      store->getOperand(0) != product->getResult(0) ||
      !product->getResult(0).hasOneUse() || !ret || ret.getNumOperands() != 1 ||
      ret.getOperand(0) != store->getResult(0)) return refuse();
  auto input = dyn_cast<RankedTensorType>(product->getResult(0).getType());
  auto output = dyn_cast<RankedTensorType>(store->getResult(0).getType());
  auto dtype = store->getAttrOfType<StringAttr>("dtype");
  if (!input || !output || !input.hasStaticShape() || !output.hasStaticShape() ||
      input.getRank() != 2 || output.getRank() != 2 || input.getEncoding() ||
      output.getEncoding() || input.getShape() != output.getShape() ||
      !input.getElementType().isF32() || !output.getElementType().isBF16() ||
      !dtype || dtype.getValue() != "bf16") return refuse();
  for (auto attr : store->getAttrs())
    if (attr.getName() != "dtype" && attr.getName() != "pure" &&
            attr.getName() != "tessera.effect_kind") return refuse();
  if (auto effect = store->getAttr("tessera.effect_kind"))
    if (!isa<StringAttr>(effect) || cast<StringAttr>(effect).getValue() != "pure")
      return refuse();
  if (auto pure = store->getAttr("pure"))
    if (!isa<BoolAttr>(pure) || !cast<BoolAttr>(pure).getValue()) return refuse();
  auto scale = product->getAttrOfType<DictionaryAttr>("scale_layout");
  auto policy = product->getAttrOfType<DictionaryAttr>("numeric_policy");
  auto granularity = scale ? scale.getAs<StringAttr>("granularity") : StringAttr{};
  auto format = scale ? scale.getAs<StringAttr>("format") : StringAttr{};
  auto block = scale ? scale.getAs<ArrayAttr>("block") : ArrayAttr{};
  auto accum = policy ? policy.getAs<StringAttr>("accum") : StringAttr{};
  auto mode = policy ? policy.getAs<StringAttr>("execution_mode") : StringAttr{};
  if (!granularity || granularity.getValue() != "block" ||
      !format || format.getValue() != "fp32" || !block || block.size() != 2 ||
      !isa<IntegerAttr>(block[0]) || cast<IntegerAttr>(block[0]).getInt() != 1 ||
      !isa<IntegerAttr>(block[1]) || cast<IntegerAttr>(block[1]).getInt() != 16 ||
      !accum || accum.getValue() != "fp32" ||
      !mode || mode.getValue() != "exact_per_block") return refuse();
  for (unsigned index = 0; index < 4; ++index) {
    auto type = dyn_cast<RankedTensorType>(product->getOperand(index).getType());
    if (!type || type.getRank() != 2 || !type.hasStaticShape() || type.getEncoding() ||
        product->getOperand(index) != fn.getArgument(index) ||
        (index < 2 ? !isa<Float8E4M3FNType>(type.getElementType())
                   : !type.getElementType().isF32())) return refuse();
  }
  // Accumulation stays FP32. The existing scaled-kernel lowering rounds once
  // at the BF16 store, exactly where this single-use Graph cast occurred.
  product->getResult(0).setType(output);
  store->getResult(0).replaceAllUsesWith(product->getResult(0));
  store->erase();
  module->removeAttr(key);
  return verify(module);
}
} // namespace tessera
