// Machine-readable member launch ABI from resolved backend code generation.
// This adapter never constructs numerical operations or a physical schedule.
#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Base64.h"

static mlir::LogicalResult projectROCMNativeProgramMember(
    mlir::ModuleOp module, llvm::StringRef entry, unsigned inputCount,
    llvm::ArrayRef<int64_t> scalars, llvm::ArrayRef<int64_t> geometry,
    llvm::StringRef scaleAdjointSchedule = "") {
  using namespace mlir;
  auto raw = module->getAttr("tessera.autodiff.scaled_member");
  if (!raw) return success();
  auto member = dyn_cast<DictionaryAttr>(raw);
  auto program = module->getAttrOfType<StringAttr>("tessera.autodiff.scaled_program_json");
  auto inputs = member ? member.getAs<DenseI64ArrayAttr>("inputs") : DenseI64ArrayAttr{};
  auto output = member ? member.getAs<IntegerAttr>("output") : IntegerAttr{};
  auto index = member ? member.getAs<IntegerAttr>("step") : IntegerAttr{};
  auto typeAttr = member ? member.getAs<TypeAttr>("type") : TypeAttr{};
  auto type = typeAttr ? dyn_cast<FunctionType>(typeAttr.getValue()) : FunctionType{};
  if (!program || !inputs || inputs.size() != inputCount || !output || !index ||
      !type || type.getNumInputs() != inputCount || type.getNumResults() != 1 ||
      geometry.size() != 6 || scalars.size() > 8 || entry.empty() ||
      module->hasAttr("tessera.rocm.program_member_json"))
    return module.emitError("native ROCm program member lost its checked SSA/ABI binding");
  llvm::json::Array inputIds, scalarValues, dimensions;
  for (auto id : inputs.asArrayRef()) inputIds.push_back(id);
  for (auto value : scalars) scalarValues.push_back(value);
  for (auto value : geometry) {
    if (value <= 0 || value > INT32_MAX)
      return module.emitError("native ROCm program launch extent is invalid");
    dimensions.push_back(value);
  }
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  if (!arch) return module.emitError("native ROCm program member has no architecture");
  llvm::json::Object manifest{
      {"schema", 1}, {"step", index.getInt()}, {"entry", entry.str()},
      {"architecture", arch.getValue().str()},
      {"abi", "flattened_rank1_memref_then_i64"},
      {"inputs", std::move(inputIds)}, {"output", output.getInt()},
      {"scalars", std::move(scalarValues)}, {"geometry", std::move(dimensions)},
      {"program_base64", program.getValue().str()}};
  if (!scaleAdjointSchedule.empty())
    manifest["scale_adjoint_schedule"] = scaleAdjointSchedule.str();
  if (auto policy = module->getAttrOfType<StringAttr>("tessera.rocm.primal_image_policy"))
    manifest["image_policy"] = policy.getValue().str();
  std::string text;
  llvm::raw_string_ostream os(text);
  os << llvm::json::Value(std::move(manifest));
  os.flush();
  module->setAttr("tessera.rocm.program_member_json",
      StringAttr::get(module.getContext(), llvm::encodeBase64(text)));
  return success();
}
