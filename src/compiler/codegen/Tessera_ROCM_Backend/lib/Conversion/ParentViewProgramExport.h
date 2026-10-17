#pragma once
#include "Tessera/IR/RowRegionContract.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Base64.h"

inline mlir::LogicalResult exportParentViewProgram(mlir::ModuleOp module, mlir::Operation *carrier) {
  using namespace mlir;
  if (failed(tessera::row_region::verifyGateUpKernel(carrier))) return failure();
  OpBuilder b(carrier);
  auto name = carrier->getAttrOfType<StringAttr>("name").getValue();
  int64_t rows = carrier->getAttrOfType<IntegerAttr>("rows").getInt();
  int64_t columns = carrier->getAttrOfType<IntegerAttr>("columns").getInt();
  int64_t elements = rows * columns;
  auto original = module->getAttrOfType<StringAttr>("tessera.rocm.parent_view_source");
  if (!original || original.getValue().empty())
    return module.emitError("parent-view frame requires the retained original Graph source witness");
  llvm::json::Array buffers;
  buffers.push_back(llvm::json::Object{
    {"shape",llvm::json::Array{rows,2*columns}},{"storage","f32"},
    {"elements",2*elements},{"bytes",8*elements},{"first_write",-1},{"last_read",0},{"ownership",0}});
  buffers.push_back(llvm::json::Object{
    {"shape",llvm::json::Array{rows,columns}},{"storage","f32"},
    {"elements",elements},{"bytes",4*elements},{"first_write",0},{"last_read",1},{"ownership",2}});
  llvm::json::Array steps;
  steps.push_back(llvm::json::Object{{"step",0},{"operation","tessera.silu_mul"},
    {"lowering","native_parent_view"},{"inputs",llvm::json::Array{0}},{"output",1}});
  llvm::json::Object program{
    {"schema",1},{"kind","parent_view"},{"argument_count",1},{"outputs",llvm::json::Array{1}},
    {"buffers",std::move(buffers)},{"steps",std::move(steps)},
    {"root_ir",original.getValue().str()},
    {"root_ir_sha256",llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(original.getValue())),true)}};
  std::string programText; llvm::raw_string_ostream programStream(programText);
  programStream << llvm::json::Value(std::move(program)); programStream.flush();
  auto programAttr = b.getStringAttr(llvm::encodeBase64(programText));
  if (auto existing = module->getAttr("tessera.rocm.parent_view_program_json"); existing && existing != programAttr)
    return module.emitError("parent-view program witness changed after native export");
  module->setAttr("tessera.rocm.parent_view_program_json",programAttr);
  llvm::json::Array names;
  auto bindings = module->getAttrOfType<ArrayAttr>("tessera.launch_bindings");
  if (bindings) {
    if (bindings.size() != 2 || !isa<StringAttr>(bindings[0]) || !isa<StringAttr>(bindings[1]) ||
        cast<StringAttr>(bindings[0]).getValue().empty() || cast<StringAttr>(bindings[1]).getValue().empty() ||
        bindings[0] == bindings[1]) return module.emitError("parent-view native bindings differ");
    for (auto name : bindings) names.push_back(cast<StringAttr>(name).getValue().str());
  } else { names.push_back("parent"); names.push_back("out"); }
  llvm::json::Object manifest{
    {"schema",1},{"profile","gfx1201_parent_view_f32_v1"},{"architecture","gfx1201"},
    {"entry",name.str()},{"abi","flattened_rank1_memref_then_i64"},
    {"step",0},{"inputs",llvm::json::Array{0}},{"output",1},
    {"bindings",std::move(names)},
    {"program_base64",llvm::encodeBase64(programText)},
    {"parent_shape",llvm::json::Array{rows,2*columns}},
    {"output_shape",llvm::json::Array{rows,columns}},
    {"scalars",llvm::json::Array{elements}},
    {"geometry",llvm::json::Array{(elements+255)/256,1,1,256,1,1}},
    {"source_hash",carrier->getAttrOfType<StringAttr>("source_hash").getValue().str()},
    {"artifact_hash",carrier->getAttrOfType<StringAttr>("artifact_hash").getValue().str()},
    {"tile_body_hash",carrier->getAttrOfType<StringAttr>("tile_body_hash").getValue().str()},
    {"target_body_hash",carrier->getAttrOfType<StringAttr>("body_hash").getValue().str()}};
  std::string text; llvm::raw_string_ostream os(text); os << llvm::json::Value(std::move(manifest)); os.flush();
  auto memberAttr = b.getStringAttr(llvm::encodeBase64(text));
  if (auto existing = module->getAttr("tessera.rocm.parent_view_member_json"); existing && existing != memberAttr)
    return module.emitError("parent-view member witness changed after native export");
  module->setAttr("tessera.rocm.parent_view_member_json",memberAttr);
  return success();
}
