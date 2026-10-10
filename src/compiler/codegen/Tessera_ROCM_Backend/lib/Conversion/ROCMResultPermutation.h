#pragma once
#include "Tessera/IR/StaticPermutationContract.h"
#include "ROCMNativeProgramMember.h"
#include "llvm/Support/SHA256.h"
namespace {
static LogicalResult lowerROCMResultPermutations(ModuleOp module, StringRef arch) {
  SmallVector<LLVM::LLVMFuncOp> functions;
  module.walk([&](LLVM::LLVMFuncOp fn) {
    if (fn->hasAttr("tessera.result_permutation_contract")) functions.push_back(fn);
  });
  for (auto fn: functions) {
    auto contract=fn->getAttrOfType<DictionaryAttr>("tessera.result_permutation_contract");
    auto shape=contract ? contract.getAs<DenseI64ArrayAttr>("source_shape") : DenseI64ArrayAttr{};
    auto axes=contract ? contract.getAs<DenseI64ArrayAttr>("permutation") : DenseI64ArrayAttr{};
    auto output=contract ? contract.getAs<DenseI64ArrayAttr>("output_shape") : DenseI64ArrayAttr{};
    auto count=contract ? contract.getAs<IntegerAttr>("elements") : IntegerAttr{};
    auto block=contract ? contract.getAs<IntegerAttr>("block_size") : IntegerAttr{};
    auto architecture=contract ? contract.getAs<StringAttr>("architecture") : StringAttr{};
    auto hash=fn->getAttrOfType<StringAttr>("tessera.schedule_hash");
    auto elements=shape && axes ? tessera::staticPermutationElements(shape.asArrayRef(),axes.asArrayRef()) : std::nullopt;
    if (!elements || !count || count.getInt()!=*elements || !block || block.getInt()!=256 ||
        !architecture || architecture.getValue()!=arch || arch!="gfx1201" || !output ||
        output.size()!=shape.size() || !hash || !fn.getBody().hasOneBlock() ||
        fn.getBody().front().getOperations().size()!=2 || fn.getNumArguments()!=3)
      return fn.emitError("ROCm result permutation lost its checked static Schedule/ABI");
    for (size_t i=0;i<shape.size();++i)
      if(output[i]!=shape[axes[i]]) return fn.emitError("ROCm result permutation output axes differ");
    auto *tile=&fn.getBody().front().front();
    if (tile->getName().getStringRef()!="tile.transpose_kernel" ||
        tile->getNumOperands()!=3 || tile->getAttr("source_shape")!=shape ||
        tile->getAttr("permutation")!=axes || tile->getAttr("tessera.schedule_hash")!=hash ||
        !isa<LLVM::ReturnOp>(fn.getBody().front().back()))
      return fn.emitError("ROCm result permutation Tile differs from its Schedule");
    for(unsigned i=0;i<3;++i)
      if(tile->getOperand(i)!=fn.getArgument(i)) return fn.emitError("ROCm result permutation source/output binding differs");
    std::string text;llvm::raw_string_ostream os(text);contract.print(os);os.flush();
    if(hash.getValue()!=llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)),true))
      return fn.emitError("ROCm result permutation Schedule digest differs");
    OpBuilder b(module.getBody(),module.getBody()->end());
    auto loc=tile->getLoc();
    std::string name=fn.getSymName().str();
    auto gpuMod=b.create<gpu::GPUModuleOp>(loc,name+"_mod");
    b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
    auto mem=MemRefType::get({ShapedType::kDynamic},b.getF32Type());
    auto kernel=b.create<gpu::GPUFuncOp>(loc,name,b.getFunctionType({mem,mem,b.getIndexType()},{}));
    kernel.setKernelAttr(b.getUnitAttr());
    b.setInsertionPointToStart(&kernel.getBody().front());
    auto constant=[&](int64_t v){return b.create<arith::ConstantIndexOp>(loc,v).getResult();};
    Value gid=b.create<arith::AddIOp>(loc,
      b.create<arith::MulIOp>(loc,b.create<gpu::BlockIdOp>(loc,gpu::Dimension::x),constant(256)),
      b.create<gpu::ThreadIdOp>(loc,gpu::Dimension::x));
    Value inside=b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::ult,gid,kernel.getArgument(2));
    auto guarded=b.create<scf::IfOp>(loc,inside,false);
    b.setInsertionPointToStart(guarded.thenBlock());
    SmallVector<int64_t> strides(shape.size(),1);
    for(size_t i=shape.size()-1;i>0;--i) strides[i-1]=strides[i]*shape[i];
    Value quotient=gid, offset=constant(0);
    for(size_t axis=shape.size();axis>0;--axis) {
      size_t i=axis-1;
      Value extent=constant(output[i]);
      Value coord=b.create<arith::RemUIOp>(loc,quotient,extent);
      quotient=b.create<arith::DivUIOp>(loc,quotient,extent);
      offset=b.create<arith::AddIOp>(loc,offset,b.create<arith::MulIOp>(loc,coord,constant(strides[axes[i]])));
    }
    Value value=b.create<memref::LoadOp>(loc,kernel.getArgument(0),ValueRange{offset});
    b.create<memref::StoreOp>(loc,value,kernel.getArgument(1),ValueRange{gid});
    b.setInsertionPointAfter(guarded);b.create<gpu::ReturnOp>(loc);
    SmallVector<int64_t> scalars{*elements}, geometry{(*elements-1)/256+1,1,1,256,1,1};
    if(failed(projectROCMNativeProgramMember(module,name,1,scalars,geometry))) return failure();
    fn.erase();
  }
  return success();
}
} // namespace
