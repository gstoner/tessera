#pragma once
#include "Tessera/IR/RowRegionContract.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include <functional>
#include <array>

namespace tessera::block_quant {
struct Profile {
  mlir::func::FuncOp graph;
  int64_t rows, columns, groups;
};

inline mlir::FailureOr<Profile> inspect(mlir::ModuleOp module) {
  using namespace mlir;
  auto refuse = [&]() -> FailureOr<Profile> {
    module.emitError("native K16 quantization requires its explicit BF16/max/scale/normalize/E4M3 Graph contract");
    return failure();
  };
  auto target = module->getAttrOfType<StringAttr>("tessera.target");
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  if (!isa<UnitAttr>(module->getAttr("tessera.native.block_quant")) ||
      !target || target.getValue() != "rocm_gfx1201" ||
      !arch || arch.getValue() != "gfx1201" || failed(verify(module)))
    return refuse();
  for (auto attr : module->getAttrs()) {
    StringRef key = attr.getName().getValue();
    if (key != "tessera.target" && key != "tessera.arch" &&
        key != "tessera.native.block_quant" && key != "tessera.ir.version" &&
        key != "tessera.frontend.authority") return refuse();
  }
  auto roots = module.getOps<func::FuncOp>();
  if (llvm::range_size(roots) != 1) return refuse();
  auto fn = *roots.begin();
  if (fn.isDeclaration() || !fn.getBody().hasOneBlock() ||
      fn.getNumArguments() != 1 || fn.getNumResults() != 2) return refuse();
  for (auto attr : fn->getAttrs()) {
    StringRef key = attr.getName().getValue();
    if (key != "sym_name" && key != "function_type" && key != "sym_visibility" &&
        key != "tessera.frontend.authority" &&
        key != "tessera.structured_cfg.schema" && key != "tessera.structured_cfg.digest" &&
        key != "tessera.structured_cfg.blocks") return refuse();
  }
  auto input = dyn_cast<RankedTensorType>(fn.getArgumentTypes()[0]);
  if (!input || input.getRank() != 2 || !input.hasStaticShape() ||
      input.getEncoding() || !input.getElementType().isBF16()) return refuse();
  int64_t m = input.getDimSize(0), k = input.getDimSize(1);
  if (m <= 0 || m > INT32_MAX || k <= 0 || k > INT32_MAX || k % 16 ||
      m > INT32_MAX / (k / 16)) return refuse();
  int64_t groups = m * (k / 16);
  SmallVector<Operation *> ops;
  for (Operation &op : fn.getBody().front().without_terminator()) ops.push_back(&op);
  const char *names[] = {
    "tessera.reshape", "tessera.cast", "tessera.absolute", "tessera.reduce",
    "arith.constant", "tessera.div", "arith.constant", "tessera.maximum",
    "tessera.expand", "tessera.div", "tessera.cast", "tessera.reshape", "tessera.reshape"};
  if (ops.size() != std::size(names)) return refuse();
  for (unsigned i=0;i<ops.size();++i)
    if (ops[i]->getName().getStringRef() != names[i] ||
        ops[i]->getNumResults() != 1 || ops[i]->getNumRegions()) return refuse();
  auto f32 = Float32Type::get(module.getContext());
  auto f8 = Float8E4M3FNType::get(module.getContext());
  auto blocks = RankedTensorType::get({groups,16}, input.getElementType());
  auto wide = RankedTensorType::get({groups,16}, f32);
  auto scale = RankedTensorType::get({groups,1}, f32);
  auto encoded = RankedTensorType::get({groups,16}, f8);
  auto codes = RankedTensorType::get({m,k}, f8);
  auto scales = RankedTensorType::get({m,k/16}, f32);
  Type expected[] = {blocks,wide,wide,scale,scale,scale,scale,scale,wide,wide,encoded,codes,scales};
  for (unsigned i=0;i<ops.size();++i)
    if (ops[i]->getResult(0).getType() != expected[i]) return refuse();
  std::array<SmallVector<Value>,13> operands = {
    SmallVector<Value>{fn.getArgument(0)}, {ops[0]->getResult(0)},
    {ops[1]->getResult(0)}, {ops[2]->getResult(0)}, {},
    {ops[3]->getResult(0),ops[4]->getResult(0)}, {},
    {ops[5]->getResult(0),ops[6]->getResult(0)}, {ops[7]->getResult(0)},
    {ops[1]->getResult(0),ops[8]->getResult(0)}, {ops[9]->getResult(0)},
    {ops[10]->getResult(0)}, {ops[7]->getResult(0)}};
  for (unsigned i=0;i<ops.size();++i) {
    if (!llvm::equal(ops[i]->getOperands(),operands[i])) return refuse();
    for (auto attr : ops[i]->getAttrs()) {
      StringRef key = attr.getName().getValue();
      if (key == "tessera.effect_kind" &&
          attr.getValue() == StringAttr::get(module.getContext(),"pure")) continue;
      if ((i==0 || i==8 || i==11 || i==12) && key=="shape") {
        auto shape = dyn_cast<ArrayAttr>(attr.getValue());
        auto result = cast<RankedTensorType>(expected[i]);
        if (!shape || shape.size() != 2) return refuse();
        for (unsigned axis=0;axis<2;++axis) {
          auto dim = dyn_cast<IntegerAttr>(shape[axis]);
          if (!dim || dim.getInt() != result.getDimSize(axis)) return refuse();
        }
        continue;
      }
      if ((i==1 || i==10) && key=="dtype" &&
          attr.getValue()==StringAttr::get(module.getContext(),i==1?"fp32":"fp8_e4m3")) continue;
      if (i==3 && (key=="kind" || key=="axis" || key=="keepdims")) continue;
      if ((i==4 || i==6) && key=="value") continue;
      return refuse();
    }
  }
  auto kind = ops[3]->getAttrOfType<StringAttr>("kind");
  auto axis = ops[3]->getAttrOfType<IntegerAttr>("axis");
  auto keep = ops[3]->getAttrOfType<BoolAttr>("keepdims");
  if (!kind || kind.getValue()!="max" || !axis || (axis.getInt()!=1 && axis.getInt()!=-1) ||
      !keep || !keep.getValue()) return refuse();
  for (unsigned index : {4u,6u}) {
    auto value = ops[index]->getAttrOfType<DenseElementsAttr>("value");
    if (!value || !value.isSplat() || value.getElementType()!=f32 ||
        value.getSplatValue<APFloat>().bitcastToAPInt().getZExtValue() !=
          (index==4 ? uint64_t(0x43e00000) : uint64_t(0x00800000))) return refuse();
  }
  auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
  if (!ret || ret.getNumOperands()!=2 ||
      ret.getOperand(0)!=ops[11]->getResult(0) || ret.getOperand(1)!=ops[12]->getResult(0))
    return refuse();
  return Profile{fn,m,k,groups};
}

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
