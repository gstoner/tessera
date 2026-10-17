#pragma once
#include "Tessera/IR/RowRegionContract.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Base64.h"

// The envelope describes owned bytes and roots; all quantization arithmetic
// remains in the source-derived Target SSA rather than this launch adapter.
inline mlir::LogicalResult exportBlockQuantProgram(mlir::ModuleOp module,mlir::Operation *carrier) {
  using namespace mlir;
  if (failed(tessera::row_region::verifyBlockQuantKernel(carrier))) return failure();
  auto original=module->getAttrOfType<StringAttr>("tessera.rocm.block_quant_source");
  if (!original || original.getValue().empty())
    return module.emitError("K16 quantizer frame requires its original Graph source");
  auto rows=carrier->getAttrOfType<IntegerAttr>("rows").getInt();
  auto columns=carrier->getAttrOfType<IntegerAttr>("columns").getInt();
  int64_t elements=rows*columns,groups=elements/16;
  auto name=carrier->getAttrOfType<StringAttr>("name").getValue();
  llvm::json::Array buffers;
  buffers.push_back(llvm::json::Object{{"shape",llvm::json::Array{rows,columns}},
    {"storage","bf16"},{"elements",elements},{"bytes",2*elements},
    {"first_write",-1},{"last_read",0},{"ownership",0}});
  buffers.push_back(llvm::json::Object{{"shape",llvm::json::Array{rows,columns}},
    {"storage","f8E4M3FN"},{"elements",elements},{"bytes",elements},
    {"first_write",0},{"last_read",1},{"ownership",2}});
  buffers.push_back(llvm::json::Object{{"shape",llvm::json::Array{rows,columns/16}},
    {"storage","f32"},{"elements",groups},{"bytes",4*groups},
    {"first_write",0},{"last_read",1},{"ownership",2}});
  llvm::json::Array steps;
  steps.push_back(llvm::json::Object{{"step",0},{"operation","explicit_k16_quantization"},
    {"lowering","native_block_quant_ssa"},{"inputs",llvm::json::Array{0}},
    {"outputs",llvm::json::Array{1,2}}});
  llvm::json::Object program{{"schema",1},{"kind","block_quant"},
    {"argument_count",1},{"outputs",llvm::json::Array{1,2}},
    {"buffers",std::move(buffers)},{"steps",std::move(steps)},
    {"root_ir",original.getValue().str()},
    {"root_ir_sha256",llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(original.getValue())),true)}};
  std::string programText;llvm::raw_string_ostream ps(programText);
  ps<<llvm::json::Value(std::move(program));ps.flush();
  llvm::json::Array bindings;
  if (auto names=module->getAttrOfType<ArrayAttr>("tessera.launch_bindings")) {
    if (names.size()!=3) return module.emitError("K16 native binding arity differs");
    for (unsigned i=0;i<3;++i) {
      auto name=dyn_cast<StringAttr>(names[i]);
      if (!name || name.getValue().empty()) return module.emitError("K16 native binding name differs");
      for (unsigned j=0;j<i;++j) if (names[i]==names[j]) return module.emitError("K16 native binding roles alias");
      bindings.push_back(name.getValue().str());
    }
  } else bindings=llvm::json::Array{"input","codes","scales"};
  llvm::json::Object member{{"schema",1},{"profile","gfx1201_block_quant_bf16_e4m3_f32_k16_v1"},
    {"architecture","gfx1201"},{"entry",name.str()},
    {"abi","flattened_rank1_memref_then_i64"},{"step",0},
    {"inputs",llvm::json::Array{0}},{"outputs",llvm::json::Array{1,2}},
    {"bindings",std::move(bindings)},
    {"program_base64",llvm::encodeBase64(programText)},
    {"scalars",llvm::json::Array{groups}},
    {"geometry",llvm::json::Array{(groups+255)/256,1,1,256,1,1}},
    {"source_hash",carrier->getAttrOfType<StringAttr>("source_hash").getValue().str()},
    {"artifact_hash",carrier->getAttrOfType<StringAttr>("artifact_hash").getValue().str()},
    {"tile_body_hash",carrier->getAttrOfType<StringAttr>("tile_body_hash").getValue().str()},
    {"target_body_hash",carrier->getAttrOfType<StringAttr>("body_hash").getValue().str()}};
  std::string memberText;llvm::raw_string_ostream ms(memberText);
  ms<<llvm::json::Value(std::move(member));ms.flush();
  Builder b(module.getContext());
  for (auto pair:{std::make_pair("tessera.rocm.block_quant_program_json",programText),
                 std::make_pair("tessera.rocm.block_quant_member_json",memberText)}) {
    auto value=b.getStringAttr(llvm::encodeBase64(pair.second));
    if (auto existing=module->getAttr(pair.first);existing && existing!=value)
      return module.emitError("native K16 frame changed after export");
    module->setAttr(pair.first,value);
  }
  return success();
}

inline mlir::LogicalResult materializeBlockQuant(mlir::ModuleOp module,mlir::Operation *carrier) {
  using namespace mlir;
  if (failed(tessera::row_region::verifyBlockQuantKernel(carrier))) return failure();
  if (failed(exportBlockQuantProgram(module,carrier))) return failure();
  auto body=cast<LLVM::LLVMFuncOp>(carrier->getRegion(0).front().front());
  auto name=carrier->getAttrOfType<StringAttr>("name").getValue();
  OpBuilder b(carrier);auto loc=carrier->getLoc();
  auto gpuModule=gpu::GPUModuleOp::create(b,loc,name.str()+"_module");
  b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
  auto memref=[&](Type element){return MemRefType::get({ShapedType::kDynamic},element);};
  auto fn=gpu::GPUFuncOp::create(b,loc,name,b.getFunctionType(
    {memref(b.getBF16Type()),memref(b.getI8Type()),memref(b.getF32Type()),b.getI64Type()},{}));
  fn.setKernel(true);b.setInsertionPointToStart(&fn.getBody().front());
  auto c=[&](int64_t value)->Value {return arith::ConstantIntOp::create(b,loc,value,64);};
  auto wide=[&](Value value)->Value {return arith::IndexCastOp::create(b,loc,b.getI64Type(),value);};
  auto cmp=[&](arith::CmpIPredicate p,Value a,Value z)->Value {return arith::CmpIOp::create(b,loc,p,a,z);};
  Value zero=c(0),one=c(1),sixteen=c(16),groups=fn.getArgument(3);
  Value tid=wide(gpu::ThreadIdOp::create(b,loc,gpu::Dimension::x));
  Value bid=wide(gpu::BlockIdOp::create(b,loc,gpu::Dimension::x));
  Value threads=wide(gpu::BlockDimOp::create(b,loc,gpu::Dimension::x));
  Value gid=arith::AddIOp::create(b,loc,arith::MulIOp::create(b,loc,bid,threads),tid);
  Value valid=cmp(arith::CmpIPredicate::sgt,groups,zero);
  auto require=[&](Value condition){valid=arith::AndIOp::create(b,loc,valid,condition);};
  require(cmp(arith::CmpIPredicate::ule,groups,c(INT32_MAX)));
  require(cmp(arith::CmpIPredicate::eq,threads,c(256)));
  SmallVector<Value> pointers;
  for (unsigned i=0;i<3;++i) {
    auto metadata=memref::ExtractStridedMetadataOp::create(b,loc,fn.getArgument(i));
    require(cmp(arith::CmpIPredicate::eq,wide(metadata.getOffset()),zero));
    require(cmp(arith::CmpIPredicate::eq,wide(metadata.getStrides()[0]),one));
    Value expected=i<2?arith::MulIOp::create(b,loc,groups,sixteen).getResult():groups;
    require(cmp(arith::CmpIPredicate::eq,wide(metadata.getSizes()[0]),expected));
    Value address=wide(memref::ExtractAlignedPointerAsIndexOp::create(b,loc,fn.getArgument(i)));
    pointers.push_back(LLVM::IntToPtrOp::create(b,loc,LLVM::LLVMPointerType::get(b.getContext()),address));
  }
  auto guard=scf::IfOp::create(b,loc,valid,false);
  b.setInsertionPointToStart(&guard.getThenRegion().front());
  IRMapping mapping;
  for (unsigned i=0;i<3;++i) mapping.map(body.getArgument(i),pointers[i]);
  mapping.map(body.getArgument(3),gid);mapping.map(body.getArgument(4),groups);
  for (Operation &op:body.getBody().front().without_terminator()) b.clone(op,mapping);
  b.setInsertionPointToEnd(&fn.getBody().front());gpu::ReturnOp::create(b,loc);
  carrier->erase();return success();
}
