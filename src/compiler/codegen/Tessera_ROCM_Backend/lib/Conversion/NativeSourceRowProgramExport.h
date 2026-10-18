#pragma once
#include "Tessera/IR/RowRegionContract.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Base64.h"

// Export the actual native row frame. The enclosing SSA program owns its
// private intermediate; this member has two readonly roots and one result.
inline mlir::LogicalResult exportNativeSourceRowMember(
    mlir::ModuleOp module, mlir::Operation *carrier) {
  using namespace mlir;
  if (failed(tessera::row_region::verifyKernel(carrier))) return failure();
  auto schema = carrier->getAttrOfType<IntegerAttr>("schema");
  auto arch = carrier->getAttrOfType<StringAttr>("arch");
  auto moduleArch = module->getAttrOfType<StringAttr>("tessera.arch");
  auto target = module->getAttrOfType<StringAttr>("tessera.target");
  auto original = module->getAttrOfType<StringAttr>("tessera.rocm.source_row_graph");
  if (!schema || schema.getInt()!=7 || !arch || !moduleArch || moduleArch!=arch ||
      !target || target.getValue()!=(llvm::Twine("rocm_")+arch.getValue()).str() ||
      !original || original.getValue().empty())
    return module.emitError("native source row frame requires its owning retained Graph");
  auto rows = carrier->getAttrOfType<IntegerAttr>("rows").getInt();
  auto columns = carrier->getAttrOfType<IntegerAttr>("columns").getInt();
  auto type = carrier->getAttrOfType<TypeAttr>("storage").getValue();
  int64_t elements = rows*columns;
  llvm::json::Array buffers;
  for (int64_t i=0;i<3;++i)
    buffers.push_back(llvm::json::Object{
      {"ordinal",i},{"shape",llvm::json::Array{rows,columns}},
      {"storage",type.isF16()?"f16":"bf16"},{"elements",elements},
      {"bytes",2*elements},{"direction",i<2?"input":"output"}});
  llvm::json::Object frame{
    {"schema","tessera.rocm.native_source_row.v1"},{"row_schema",int64_t(7)},
    {"architecture",arch.getValue().str()},
    {"entry",carrier->getAttrOfType<StringAttr>("name").getValue().str()},
    {"abi","flattened_rank1_memrefs_then_i64"},
    {"buffers",std::move(buffers)},{"inputs",llvm::json::Array{0,1}},{"output",int64_t(2)},
    {"scalars",llvm::json::Array{rows,columns}},
    {"geometry",llvm::json::Array{(rows+127)/128,int64_t(1),int64_t(1),int64_t(128),int64_t(1),int64_t(1)}},
    {"source_graph_ir",original.getValue().str()},
    {"source_graph_sha256",llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(original.getValue())),true)},
    {"source_hash",carrier->getAttrOfType<StringAttr>("source_hash").getValue().str()},
    {"artifact_hash",carrier->getAttrOfType<StringAttr>("artifact_hash").getValue().str()},
    {"tile_body_hash",carrier->getAttrOfType<StringAttr>("tile_body_hash").getValue().str()},
    {"target_body_hash",carrier->getAttrOfType<StringAttr>("body_hash").getValue().str()}};
  std::string text; llvm::raw_string_ostream stream(text);
  stream<<llvm::json::Value(std::move(frame));stream.flush();
  auto value=StringAttr::get(module.getContext(),llvm::encodeBase64(text));
  if (auto old=module->getAttr("tessera.rocm.source_row_member_json");old && old!=value)
    return module.emitError("native source row member frame changed after export");
  module->setAttr("tessera.rocm.source_row_member_json",value);
  return success();
}
