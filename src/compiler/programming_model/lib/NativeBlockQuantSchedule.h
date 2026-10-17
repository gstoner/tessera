#pragma once
#include "Tessera/IR/RowRegionContract.h"
#include "Tessera/IR/BlockQuantContract.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include <functional>
#include <array>

namespace tessera::block_quant {
// Scalarize the admitted SSA graph into typed, non-owning physical views.
// Logical reshape/expand edges allocate no storage. One thread owns a K16
// group and writes its scale once; max and normalization retain FP32 order.
inline mlir::FailureOr<mlir::ModuleOp> buildScalarModule(mlir::ModuleOp source) {
  using namespace mlir;
  auto profile = inspect(source);
  if (failed(profile)) return failure();
  auto graph = profile->graph;
  auto loc = graph.getLoc();
  OpBuilder b(source.getContext());
  ModuleOp output = ModuleOp::create(loc);
  b.setInsertionPointToEnd(output.getBody());
  auto ptr = LLVM::LLVMPointerType::get(source.getContext());
  auto f32 = b.getF32Type();
  auto bf16 = b.getBF16Type();
  auto f8 = Float8E4M3FNType::get(source.getContext());
  auto functionType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(source.getContext()),
      {ptr,ptr,ptr,b.getI64Type(),b.getI64Type()},false);
  auto fn = LLVM::LLVMFuncOp::create(b,loc,"block_quant_scalar",functionType);
  Block *entry = fn.addEntryBlock(b);
  b.setInsertionPointToStart(entry);
  auto c = [&](int64_t n) -> Value {
    return arith::ConstantIntOp::create(b,loc,n,64);
  };
  Value zero=c(0), one=c(1), sixteen=c(16);
  Value group=entry->getArgument(3), groups=entry->getArgument(4);
  auto view = [&](unsigned root,Type element,StringRef access,Value columns) -> Value {
    auto type=tessera::tile::BufferViewType::get(source.getContext(),element,access,"row_major");
    return tessera::tile::BufferViewOp::create(b,loc,type,entry->getArgument(root),groups,columns,columns);
  };
  Value input=view(0,bf16,"read",sixteen);
  Value codes=view(1,f8,"write",sixteen);
  Value scales=view(2,f32,"write",one);
  auto active=arith::CmpIOp::create(b,loc,arith::CmpIPredicate::ult,group,groups);
  auto guard=scf::IfOp::create(b,loc,active,false);
  b.setInsertionPointToStart(&guard.getThenRegion().front());
  SmallVector<Operation *> ops;
  for (Operation &op:graph.getBody().front().without_terminator()) ops.push_back(&op);
  Value maximum, scale;
  std::function<FailureOr<Value>(Value,Value)> scalar;
  scalar = [&](Value value,Value lane) -> FailureOr<Value> {
    if (auto arg=dyn_cast<BlockArgument>(value)) {
      if (arg!=graph.getArgument(0)) return failure();
      return Value(tessera::tile::LoadElementOp::create(b,loc,bf16,input,group,lane));
    }
    auto op=value.getDefiningOp();
    if (!op) return failure();
    StringRef name=op->getName().getStringRef();
    if (name=="tessera.reshape" || name=="tessera.expand")
      return scalar(op->getOperand(0),lane);
    if (name=="arith.constant") {
      auto attr=cast<DenseElementsAttr>(op->getAttr("value"));
      return Value(arith::ConstantOp::create(b,loc,FloatAttr::get(f32,attr.getSplatValue<APFloat>())));
    }
    if (name=="tessera.reduce") { if (!maximum) return failure(); return maximum; }
    if (op==ops[7] && scale) return scale;
    auto lhs=scalar(op->getOperand(0),lane);
    if (failed(lhs)) return failure();
    if (name=="tessera.cast") {
      auto type=cast<RankedTensorType>(value.getType()).getElementType();
      if (type.isF32()) return Value(arith::ExtFOp::create(b,loc,type,*lhs));
      return Value(arith::TruncFOp::create(b,loc,type,*lhs));
    }
    if (name=="tessera.absolute") return Value(math::AbsFOp::create(b,loc,*lhs));
    auto rhs=scalar(op->getOperand(1),lane);
    if (failed(rhs)) return failure();
    if (name=="tessera.div") return Value(arith::DivFOp::create(b,loc,*lhs,*rhs));
    if (name=="tessera.maximum") return Value(arith::MaximumFOp::create(b,loc,*lhs,*rhs));
    return failure();
  };
  Value initial=arith::ConstantOp::create(b,loc,b.getF32FloatAttr(0.0));
  auto reduction=scf::ForOp::create(b,loc,zero,sixteen,one,ValueRange{initial});
  b.setInsertionPointToStart(reduction.getBody());
  auto absolute=scalar(ops[2]->getResult(0),reduction.getInductionVar());
  if (failed(absolute)) { output->destroy(); return failure(); }
  Value next=arith::MaximumFOp::create(b,loc,reduction.getRegionIterArgs()[0],*absolute);
  scf::YieldOp::create(b,loc,next);
  b.setInsertionPointAfter(reduction);
  maximum=reduction.getResult(0);
  auto computedScale=scalar(ops[7]->getResult(0),zero);
  if (failed(computedScale)) { output->destroy(); return failure(); }
  scale=*computedScale;
  tessera::tile::StoreElementOp::create(b,loc,scales,group,zero,scale);
  auto encoding=scf::ForOp::create(b,loc,zero,sixteen,one);
  b.setInsertionPointToStart(encoding.getBody());
  auto encoded=scalar(ops[10]->getResult(0),encoding.getInductionVar());
  if (failed(encoded)) { output->destroy(); return failure(); }
  tessera::tile::StoreElementOp::create(b,loc,codes,group,encoding.getInductionVar(),*encoded);
  b.setInsertionPointAfter(guard);
  LLVM::ReturnOp::create(b,loc,ValueRange{});
  if (failed(verify(output))) { output->destroy(); return failure(); }
  return output;
}

inline mlir::DictionaryAttr artifactAttributes(Profile p) {
  using namespace mlir;
  Builder b(p.graph.getContext());
  auto source = row_region::sourceDigest(p.graph);
  auto identity = source + ";gfx1201;native_block_quant_schema=4;K16;bf16_e4m3_f32;RNE;threads=256";
  auto hash = llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(identity)),true);
  return b.getDictionaryAttr({
    b.getNamedAttr("hash",b.getStringAttr(hash)),
    b.getNamedAttr("arch",b.getStringAttr("gfx1201")),
    b.getNamedAttr("shape_key",b.getStringAttr("family=native_block_quant;rows="+
      std::to_string(p.rows)+";columns="+std::to_string(p.columns)+";groups="+std::to_string(p.groups))),
    b.getNamedAttr("numeric_policy",b.getStringAttr("explicit_fp32;K16_maxabs;floor_min_normal;RNE_e4m3;propagate_NaN")),
    b.getNamedAttr("tile",b.getDictionaryAttr({
      b.getNamedAttr("source_hash",b.getStringAttr(source)),
      b.getNamedAttr("algorithm",b.getStringAttr("serial_k16_ssa")),
      b.getNamedAttr("workgroup_size",b.getI64IntegerAttr(256)),
      b.getNamedAttr("schema",b.getI64IntegerAttr(4))}))});
}

inline mlir::LogicalResult schedule(mlir::ModuleOp module,bool &selected) {
  using namespace mlir;
  selected=module->hasAttr("tessera.native.block_quant");
  if (!selected) return success();
  auto profile = inspect(module);
  if (failed(profile)) return failure();
  if (module.getBody()->getOperations().size()!=1)
    return module.emitError("native K16 quantization Schedule requires its isolated original Graph");
  OpBuilder b(module.getContext());b.setInsertionPointToEnd(module.getBody());
  OperationState state(profile->graph.getLoc(),"schedule.artifact");
  state.addAttributes(artifactAttributes(*profile).getValue());
  b.create(state);
  return success();
}
inline mlir::LogicalResult lower(mlir::ModuleOp module,bool &selected) {
  using namespace mlir;
  selected=module->hasAttr("tessera.native.block_quant");
  if (!selected) return success();
  auto profile=inspect(module);
  if (failed(profile)) return failure();
  Operation *artifact=nullptr;
  for (Operation &op:module.getBody()->getOperations()) {
    if (&op==profile->graph.getOperation()) continue;
    if (op.getName().getStringRef()!="schedule.artifact" || artifact)
      return module.emitError("native K16 Tile requires one original Schedule witness");
    artifact=&op;
  }
  if (!artifact || artifact->getAttrDictionary()!=artifactAttributes(*profile))
    return module.emitError("native K16 Schedule differs from its original Graph");
  auto built=buildScalarModule(module);
  if (failed(built)) return module.emitError("native K16 scalarization failed");
  OwningOpRef<ModuleOp> temporary(*built);
  auto fn=*temporary->getOps<LLVM::LLVMFuncOp>().begin();
  auto physical=row_region::digest(fn);
  fn.setSymName("tessera_block_quant_"+physical.substr(0,16));
  OpBuilder b(artifact);
  OperationState state(profile->graph.getLoc(),"tile.row_region_kernel");
  state.addAttribute("name",b.getStringAttr(fn.getName()));
  state.addAttribute("arch",b.getStringAttr("gfx1201"));
  state.addAttribute("schema",b.getI64IntegerAttr(4));
  state.addAttribute("storage",TypeAttr::get(b.getBF16Type()));
  state.addAttribute("rows",b.getI64IntegerAttr(profile->rows));
  state.addAttribute("columns",b.getI64IntegerAttr(profile->columns));
  state.addAttribute("workgroup_size",b.getI64IntegerAttr(256));
  state.addAttribute("source_hash",b.getStringAttr(row_region::sourceDigest(profile->graph)));
  state.addAttribute("artifact_hash",artifact->getAttr("hash"));
  state.addAttribute("body_hash",b.getStringAttr(row_region::digest(fn)));
  state.addRegion();
  auto carrier=b.create(state);
  carrier->getRegion(0).push_back(new Block);
  fn->moveBefore(&carrier->getRegion(0).front(),carrier->getRegion(0).front().end());
  std::string original;
  llvm::raw_string_ostream stream(original);
  profile->graph->print(stream,OpPrintingFlags().useLocalScope()); stream.flush();
  module->setAttr("tessera.rocm.block_quant_source",b.getStringAttr(original));
  profile->graph.erase(); artifact->erase();
  return row_region::verifyKernel(carrier);
}
} // namespace tessera::block_quant
