#pragma once
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "NativeSourceRowProgramExport.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

// Adapt the verified Target SSA to the existing HIP owner's flattened memref
// ABI. All derivative arithmetic and row addressing remain the carried body.
inline mlir::LogicalResult materializeNativeSourceRow(
    mlir::ModuleOp module, mlir::Operation *carrier) {
  using namespace mlir;
  if (failed(exportNativeSourceRowMember(module,carrier))) return failure();
  auto body=cast<LLVM::LLVMFuncOp>(carrier->getRegion(0).front().front());
  auto storage=carrier->getAttrOfType<TypeAttr>("storage").getValue();
  auto name=carrier->getAttrOfType<StringAttr>("name").getValue();
  int64_t rows=carrier->getAttrOfType<IntegerAttr>("rows").getInt();
  int64_t columns=carrier->getAttrOfType<IntegerAttr>("columns").getInt();
  OpBuilder b(carrier);auto loc=carrier->getLoc();
  auto gpuModule=gpu::GPUModuleOp::create(b,loc,name.str()+"_module");
  b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
  auto flat=MemRefType::get({ShapedType::kDynamic},storage);
  auto fn=gpu::GPUFuncOp::create(b,loc,name,b.getFunctionType(
      {flat,flat,flat,b.getI64Type(),b.getI64Type()},{}));
  fn.setKernel(true);b.setInsertionPointToStart(&fn.getBody().front());
  auto constant=[&](int64_t v)->Value{return arith::ConstantIntOp::create(b,loc,v,64);};
  auto wide=[&](Value v)->Value{return arith::IndexCastOp::create(b,loc,b.getI64Type(),v);};
  auto eq=[&](Value a,Value z)->Value{return arith::CmpIOp::create(b,loc,arith::CmpIPredicate::eq,a,z);};
  Value valid=eq(fn.getArgument(3),constant(rows));
  auto require=[&](Value v){valid=arith::AndIOp::create(b,loc,valid,v);};
  require(eq(fn.getArgument(4),constant(columns)));
  require(eq(wide(gpu::BlockDimOp::create(b,loc,gpu::Dimension::x)),constant(128)));
  SmallVector<Value> pointers;
  for (unsigned i=0;i<3;++i) {
    auto metadata=memref::ExtractStridedMetadataOp::create(b,loc,fn.getArgument(i));
    require(eq(wide(metadata.getOffset()),constant(0)));
    require(eq(wide(metadata.getSizes()[0]),constant(rows*columns)));
    require(eq(wide(metadata.getStrides()[0]),constant(1)));
    Value address=wide(memref::ExtractAlignedPointerAsIndexOp::create(b,loc,fn.getArgument(i)));
    pointers.push_back(LLVM::IntToPtrOp::create(b,loc,LLVM::LLVMPointerType::get(b.getContext()),address));
  }
  auto guard=scf::IfOp::create(b,loc,valid,false);
  b.setInsertionPointToStart(&guard.getThenRegion().front());
  IRMapping mapping;
  for (unsigned i=0;i<3;++i) mapping.map(body.getArgument(i),pointers[i]);
  mapping.map(body.getArgument(3),fn.getArgument(3));
  mapping.map(body.getArgument(4),fn.getArgument(4));
  for (Operation &op:body.getBody().front().without_terminator()) b.clone(op,mapping);
  b.setInsertionPointToEnd(&fn.getBody().front());gpu::ReturnOp::create(b,loc);
  carrier->erase();
  return success();
}
