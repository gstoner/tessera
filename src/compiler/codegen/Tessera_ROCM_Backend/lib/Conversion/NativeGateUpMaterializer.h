#include "ParentViewProgramExport.h"
#pragma once
#include "Tessera/IR/RowRegionContract.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Base64.h"

// The Target body already contains all numerical/addressing SSA. This adapter
// adds ROCm launch coordinates and the flattened memref frame, never a second
// SiLU recipe. Source-specific counts remain a checked package responsibility.
inline mlir::LogicalResult materializeGateUpViews(mlir::ModuleOp module) {
  using namespace mlir;
  SmallVector<Operation *> carriers;
  for (Operation &op : module.getBody()->getOperations())
    if (op.getName().getStringRef() == "tessera_rocm.row_region_kernel") carriers.push_back(&op);
  if (carriers.empty()) return success();
  if (carriers.size() != 1 || module.getBody()->getOperations().size() != 1)
    return module.emitError("parent-view native materialization requires one isolated Target body");
  Operation *carrier = carriers.front();
  if (failed(tessera::row_region::verifyGateUpKernel(carrier))) return failure();
  auto body = cast<LLVM::LLVMFuncOp>(carrier->getRegion(0).front().front());
  auto name = carrier->getAttrOfType<StringAttr>("name").getValue();
  int64_t columns = carrier->getAttrOfType<IntegerAttr>("columns").getInt();
  OpBuilder b(carrier); auto loc = carrier->getLoc();
  auto gpuModule = gpu::GPUModuleOp::create(b,loc,name.str()+"_module");
  b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
  auto storage = carrier->getAttrOfType<TypeAttr>("storage").getValue();
  auto memref = MemRefType::get({ShapedType::kDynamic},storage);
  auto function = gpu::GPUFuncOp::create(b,loc,name,b.getFunctionType({memref,memref,b.getI64Type()},{}));
  function.setKernel(true);
  b.setInsertionPointToStart(&function.getBody().front());
  auto c = [&](int64_t v) -> Value { return arith::ConstantIntOp::create(b,loc,v,64); };
  auto wide = [&](Value v) -> Value { return arith::IndexCastOp::create(b,loc,b.getI64Type(),v); };
  Value tid = wide(gpu::ThreadIdOp::create(b,loc,gpu::Dimension::x));
  Value bid = wide(gpu::BlockIdOp::create(b,loc,gpu::Dimension::x));
  Value threads = wide(gpu::BlockDimOp::create(b,loc,gpu::Dimension::x));
  Value gid = arith::AddIOp::create(b,loc,arith::MulIOp::create(b,loc,bid,threads),tid);
  Value count = function.getArgument(2), zero = c(0), one = c(1), two = c(2);
  Value width = c(columns);
  auto compare = [&](arith::CmpIPredicate p,Value a,Value z) -> Value { return arith::CmpIOp::create(b,loc,p,a,z); };
  Value valid = compare(arith::CmpIPredicate::sgt,count,zero);
  auto require = [&](Value test) { valid = arith::AndIOp::create(b,loc,valid,test); };
  require(compare(arith::CmpIPredicate::ule,count,c(INT64_MAX/8)));
  require(compare(arith::CmpIPredicate::eq,arith::RemUIOp::create(b,loc,count,width),zero));
  require(compare(arith::CmpIPredicate::ule,arith::DivUIOp::create(b,loc,count,width),c(INT32_MAX)));
  require(compare(arith::CmpIPredicate::eq,threads,c(256)));
  SmallVector<Value> pointers;
  for (unsigned i=0;i<2;++i) {
    auto metadata = memref::ExtractStridedMetadataOp::create(b,loc,function.getArgument(i));
    require(compare(arith::CmpIPredicate::eq,wide(metadata.getOffset()),zero));
    require(compare(arith::CmpIPredicate::eq,wide(metadata.getStrides()[0]),one));
    Value expected = i == 0 ? arith::MulIOp::create(b,loc,count,two).getResult() : count;
    require(compare(arith::CmpIPredicate::eq,wide(metadata.getSizes()[0]),expected));
    Value address = wide(memref::ExtractAlignedPointerAsIndexOp::create(b,loc,function.getArgument(i)));
    pointers.push_back(LLVM::IntToPtrOp::create(b,loc,LLVM::LLVMPointerType::get(b.getContext()),address));
  }
  auto guarded = scf::IfOp::create(b,loc,valid,false);
  b.setInsertionPointToStart(&guarded.getThenRegion().front());
  IRMapping mapping;
  mapping.map(body.getArgument(0),pointers[0]); mapping.map(body.getArgument(1),pointers[1]);
  mapping.map(body.getArgument(2),gid); mapping.map(body.getArgument(3),count);
  for (Operation &op : body.getBody().front().without_terminator()) b.clone(op,mapping);
  b.setInsertionPointToEnd(&function.getBody().front()); gpu::ReturnOp::create(b,loc);
  if (failed(exportParentViewProgram(module,carrier))) return failure();
  carrier->erase();
  return success();
}
