// Native SSA partition for BF16 -> K16 quantization -> scaled product.
#pragma once
#include "Tessera/IR/BlockQuantContract.h"

namespace tessera {
static mlir::LogicalResult emitNativeQuantizedGateUpProgram(mlir::ModuleOp module);
static mlir::LogicalResult emitNativeQuantizedMLPProgram(mlir::ModuleOp module);
static mlir::func::FuncOp cloneQuantizedProductMember(mlir::func::FuncOp member,
                                                    int64_t index) {
  auto child = mlir::cast<mlir::func::FuncOp>(member->clone());
  bool expansion = member.getBody().front().front().getName().getStringRef() ==
                   "tessera.nvfp4_expand";
  auto projectedName=member->getAttrOfType<mlir::StringAttr>("tessera.native.quantized_product_projection_name");
  if (projectedName) child.setSymName(projectedName.getValue());
  else child.setSymName(index == 0 ? "quantize" : expansion ? "expand" : "quantized_product");
  child->removeAttr("tessera.native.quantized_product_projection");
  child->removeAttr("tessera.native.quantized_product_projection_name");
  if (expansion) {
    mlir::OpBuilder b(member.getContext());
    child->setAttr("tessera.bindings", b.getStrArrayAttr(
        {"codes", "scales", "globals", "values", "scale_plane"}));
  }
  child.setPublic();
  child->removeAttr("tessera.native.quantized_product_member");
  return child;
}
static mlir::LogicalResult emitNativeQuantizedProductProgram(mlir::ModuleOp module) {
  using namespace mlir;
  auto refuse = [&]() -> LogicalResult {
    return module.emitError("resident quantized product requires its original BF16 K16 producer and exact E4M3 scaled-product SSA");
  };
  if (failed(verify(module))) return failure();
  auto target = module->getAttrOfType<StringAttr>("tessera.target");
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  if (!target || target.getValue() != "rocm_gfx1201" || !arch ||
      arch.getValue() != "gfx1201" ||
      !isa<UnitAttr>(module->getAttr("tessera.native.quantized_product"))) return refuse();
  for (auto attr : module->getAttrs()) {
    auto name = attr.getName().getValue();
    if (name != "tessera.target" && name != "tessera.arch" &&
        name != "tessera.native.quantized_product" && name != "tessera.ir.version" &&
        name != "tessera.frontend.authority") return refuse();
  }
  auto roots = module.getOps<func::FuncOp>();
  if (llvm::range_size(roots) != 1) return refuse();
  auto root = *roots.begin();
  if (root.isDeclaration() || !root.getBody().hasOneBlock() ||
      (root.getNumArguments() != 3 && root.getNumArguments() != 4 && root.getNumArguments() != 7) || root.getNumResults() != 1) return refuse();
  for (unsigned i=0; i<root.getNumArguments(); ++i)
    if (auto attrs = root.getArgAttrDict(i); attrs && !attrs.empty()) return refuse();
  SmallVector<Operation *> ops;
  for (Operation &op : root.getBody().front().without_terminator()) ops.push_back(&op);
  if (ops.size()==37 || ops.size()==38) return emitNativeQuantizedMLPProgram(module);
  if (ops.size()==22) return emitNativeQuantizedGateUpProgram(module);
  bool expansion = (ops.size() == 15 || ops.size() == 16) &&
      ops[13]->getName().getStringRef() == "tessera.nvfp4_expand";
  if ((!expansion && ops.size() != 14) ||
      root.getNumArguments() != (expansion ? 4u : 3u)) return refuse();
  bool storeCast=expansion && ops.size()==16;
  if (storeCast && ops.back()->getNumResults()!=1) return refuse();
  auto product = dyn_cast<ScaledMatmulOp>(ops[storeCast?14:ops.size()-1]);
  auto ret = dyn_cast<func::ReturnOp>(root.getBody().front().back());
  if (!product || product->getNumOperands() != 4 || product->getNumResults() != 1 ||
      !ret || ret.getNumOperands() != 1 ||
      ret.getOperand(0) != (storeCast?ops.back()->getResult(0):product->getResult(0)) ||
      ops[0]->getNumOperands() != 1 || ops[11]->getNumResults() != 1 || ops[12]->getNumResults() != 1 ||
      product->getOperand(0) != ops[11]->getResult(0) ||
      product->getOperand(2) != ops[12]->getResult(0)) return refuse();
  if (storeCast && failed(verifyNativeFP32ToBF16Store(product->getResult(0),ops.back()))) return failure();
  SmallVector<Value> roles{ops[0]->getOperand(0)};
  if (expansion) {
    auto weight = ops[13];
    if (weight->getNumOperands() != 3 || weight->getNumResults() != 2 ||
        product->getOperand(1) != weight->getResult(0) ||
        product->getOperand(3) != weight->getResult(1)) return refuse();
    llvm::append_range(roles, weight->getOperands());
  } else {
    roles.push_back(product->getOperand(1));
    roles.push_back(product->getOperand(3));
  }
  llvm::SmallDenseSet<unsigned> arguments;
  for (Value value : roles) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &root.getBody().front() ||
        !arguments.insert(argument.getArgNumber()).second) return refuse();
  }
  OpBuilder b(module.getContext());
  auto prefix = ModuleOp::create(module.getLoc());
  prefix->setAttr("tessera.target", target); prefix->setAttr("tessera.arch", arch);
  prefix->setAttr("tessera.native.block_quant", b.getUnitAttr());
  b.setInsertionPointToEnd(prefix.getBody());
  auto quant = b.create<func::FuncOp>(root.getLoc(), "quantize",
      b.getFunctionType(TypeRange{roles[0].getType()},
                       TypeRange{ops[11]->getResult(0).getType(),ops[12]->getResult(0).getType()}));
  for (auto attr : root->getAttrs())
    if (attr.getName() != "sym_name" && attr.getName() != "function_type")
      quant->setAttr(attr.getName(), attr.getValue());
  auto block = quant.addEntryBlock();
  OpBuilder qb(block,block->begin()); IRMapping qm;
  qm.map(roles[0],block->getArgument(0));
  for (unsigned i=0; i<13; ++i) qb.clone(*ops[i],qm);
  qb.create<func::ReturnOp>(root.getLoc(),ValueRange{
      qm.lookup(ops[11]->getResult(0)),qm.lookup(ops[12]->getResult(0))});
  auto profile = block_quant::inspect(prefix);
  if (failed(profile)) { prefix.erase(); return failure(); }
  int64_t m=profile->rows,k=profile->columns;
  auto rhs = dyn_cast<RankedTensorType>(product->getOperand(1).getType());
  auto scales = dyn_cast<RankedTensorType>(product->getOperand(3).getType());
  auto output = dyn_cast<RankedTensorType>(ret.getOperand(0).getType());
  auto trans = product->getAttrOfType<BoolAttr>("transposeB");
  auto scale = product->getAttrOfType<DictionaryAttr>("scale_layout");
  auto numeric = product->getAttrOfType<DictionaryAttr>("numeric_policy");
  auto granularity = scale ? scale.getAs<StringAttr>("granularity") : StringAttr{};
  auto format = scale ? scale.getAs<StringAttr>("format") : StringAttr{};
  auto sizes = scale ? scale.getAs<ArrayAttr>("block") : ArrayAttr{};
  auto accum = numeric ? numeric.getAs<StringAttr>("accum") : StringAttr{};
  auto mode = numeric ? numeric.getAs<StringAttr>("execution_mode") : StringAttr{};
  bool metadataValid = true;
  for (auto attr : product->getAttrs()) {
    auto key=attr.getName().getValue();
    if (key=="scale_layout" || key=="numeric_policy" || key=="transposeB") continue;
    if (key=="transposeA" && attr.getValue()==b.getBoolAttr(false)) continue;
    if (key=="tessera.effect_kind" && attr.getValue()==b.getStringAttr("pure")) continue;
    if (key=="pure" && attr.getValue()==b.getBoolAttr(true)) continue;
    metadataValid=false;
  }
  bool valid = metadataValid && rhs && rhs.hasStaticShape() && rhs.getRank()==2 && !rhs.getEncoding() &&
      isa<Float8E4M3FNType>(rhs.getElementType()) && trans && trans.getValue() &&
      rhs.getDimSize(0)>0 && rhs.getDimSize(1)==k &&
      scales && scales.hasStaticShape() && !scales.getEncoding() && scales.getElementType().isF32() &&
      scales.getShape()==ArrayRef<int64_t>({k/16,rhs.getDimSize(0)}) &&
      output && output.hasStaticShape() && !output.getEncoding() &&
      output.getShape()==ArrayRef<int64_t>({m,rhs.getDimSize(0)}) &&
      (output.getElementType().isF32() || output.getElementType().isBF16()) &&
      scale && numeric && scale.size()==3 && numeric.size()==2 &&
      granularity && granularity.getValue()=="block" && format && format.getValue()=="fp32" &&
      sizes && sizes.size()==2 && isa<IntegerAttr>(sizes[0]) && isa<IntegerAttr>(sizes[1]) &&
      cast<IntegerAttr>(sizes[0]).getInt()==1 && cast<IntegerAttr>(sizes[1]).getInt()==16 &&
      accum && accum.getValue()=="fp32" && mode && mode.getValue()=="exact_per_block";
  if (!valid) { prefix.erase(); return refuse(); }
  std::string source; llvm::raw_string_ostream sourceStream(source);
  module.print(sourceStream); sourceStream.flush();
  SmallVector<SmallVector<Operation *>> groups;
  groups.emplace_back(ops.begin(),ops.begin()+13);
  SmallVector<SmallVector<Value>> inputs{{roles[0]}};
  SmallVector<SmallVector<Value>> outputs{{ops[11]->getResult(0),ops[12]->getResult(0)}};
  if (expansion) {
    groups.push_back({ops[13]});
    inputs.push_back({roles[1],roles[2],roles[3]});
    outputs.push_back({ops[13]->getResult(0),ops[13]->getResult(1)});
  }
  if (storeCast) groups.push_back({product,ops.back()});
  else groups.push_back({product});
  inputs.push_back({product->getOperand(0),product->getOperand(1),
                    product->getOperand(2),product->getOperand(3)});
  outputs.push_back({ret.getOperand(0)});
  llvm::DenseMap<Value,int64_t> ids; SmallVector<Value> values;
  SmallVector<int64_t> writes,reads;
  for (Value value : root.getArguments()) {
    ids[value]=values.size(); values.push_back(value); writes.push_back(-1); reads.push_back(-1);
  }
  for (unsigned i=0;i<groups.size();++i) {
    for (Value value : inputs[i]) reads[ids.lookup(value)]=i;
    for (Value value : outputs[i]) {
      ids[value]=values.size(); values.push_back(value); writes.push_back(i); reads.push_back(i);
    }
  }
  reads[ids.lookup(ret.getOperand(0))]=groups.size();
  llvm::json::Array buffers,steps,members;
  for (auto [index,value] : llvm::enumerate(values)) {
    auto type=cast<RankedTensorType>(value.getType());
    int64_t bytes=type.getElementType().getIntOrFloatBitWidth()/8;
    llvm::json::Array shape;
    for (int64_t dim : type.getShape()) {
      if (dim<=0 || bytes<=0 || dim>INT64_MAX/bytes) { prefix.erase(); return refuse(); }
      bytes*=dim; shape.push_back(dim);
    }
    std::string storage; llvm::raw_string_ostream stream(storage); type.getElementType().print(stream); stream.flush();
    buffers.push_back(llvm::json::Object{{"id",int64_t(index)},{"bytes",bytes},
      {"shape",std::move(shape)},{"storage",storage},
      {"ownership",index<root.getNumArguments()?"readonly_input":value==ret.getOperand(0)?"returned_output":"private_scratch"},
      {"first_write",writes[index]},{"last_read",reads[index]}});
  }
  SmallVector<Attribute> symbols;
  for (unsigned i=0;i<groups.size();++i) {
    auto name=(root.getSymName()+"__quantized_member_"+llvm::Twine(i)).str();
    if (SymbolTable::lookupSymbolIn(module,name)) { prefix.erase(); return refuse(); }
    SmallVector<Type> ins,outs;
    for (Value value:inputs[i]) ins.push_back(value.getType());
    for (Value value:outputs[i]) outs.push_back(value.getType());
    b.setInsertionPointToEnd(module.getBody());
    auto member=b.create<func::FuncOp>(root.getLoc(),name,b.getFunctionType(ins,outs));
    member.setPrivate(); member->setAttr("tessera.native.quantized_product_member",b.getI64IntegerAttr(i));
    symbols.push_back(FlatSymbolRefAttr::get(member));
    auto entry=member.addEntryBlock(); OpBuilder body(entry,entry->begin()); IRMapping mapping;
    llvm::json::Array inIDs,outIDs;
    for (auto [value,arg]:llvm::zip(inputs[i],entry->getArguments())) {mapping.map(value,arg);inIDs.push_back(ids.lookup(value));}
    for (Operation *op:groups[i]) body.clone(*op,mapping);
    SmallVector<Value> returned;
    for (Value value:outputs[i]) {returned.push_back(mapping.lookup(value));outIDs.push_back(ids.lookup(value));}
    body.create<func::ReturnOp>(root.getLoc(),returned);
    auto projected=ModuleOp::create(module.getLoc());
    projected->setAttr("tessera.target",i+1==groups.size()?b.getStringAttr("rocm"):target);
    projected->setAttr("tessera.arch",arch);
    if (i==0) projected->setAttr("tessera.native.block_quant",b.getUnitAttr());
    if (storeCast && i+1==groups.size()) {
      projected->setAttr("tessera.target",target);
      projected->setAttr("tessera.native.fp8_store_cast",b.getUnitAttr());
      member->setAttr("tessera.native.quantized_product_projection",b.getDictionaryAttr(projected->getAttrs()));
      member->setAttr("tessera.native.quantized_product_projection_name",b.getStringAttr("quantized_product"));
    }
    projected.getBody()->push_back(cloneQuantizedProductMember(member,i));
    std::string text; llvm::raw_string_ostream os(text); projected.print(os);os.flush();
    members.push_back(text+"\n"); projected.erase();
    steps.push_back(llvm::json::Object{{"step",int64_t(i)},
      {"operation",i==0?"native_block_quant":groups[i][0]->getName().getStringRef()},
      {"member",name},{"inputs",std::move(inIDs)},{"outputs",std::move(outIDs)}});
  }
  prefix.erase();
  llvm::json::Object plan{{"schema",expansion?"tessera.native.quantized_product.v2":"tessera.native.quantized_product.v1"},
    {"source_graph_ir",source+"\n"},{"member_graphs",std::move(members)},
    {"buffers",std::move(buffers)},{"steps",std::move(steps)},
    {"output",ids.lookup(ret.getOperand(0))}};
  std::string json; llvm::raw_string_ostream os(json);os<<llvm::json::Value(std::move(plan));os.flush();
  module->setAttr("tessera.native.quantized_product_json",b.getStringAttr(llvm::encodeBase64(json)));
  module->setAttr("tessera.native.quantized_product_members",b.getArrayAttr(symbols));
  return success();
}
static mlir::LogicalResult projectNativeQuantizedProductMember(mlir::ModuleOp module,int64_t index) {
  using namespace mlir;
  auto members=module->getAttrOfType<ArrayAttr>("tessera.native.quantized_product_members");
  if (!members || index<0 || index>=int64_t(members.size()))
    return module.emitError("quantized product member index is outside its native program");
  auto member=dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupSymbolIn(module,cast<FlatSymbolRefAttr>(members[index])));
  auto lineage=member?member->getAttrOfType<IntegerAttr>("tessera.native.quantized_product_member"):IntegerAttr{};
  if (!member || !lineage || lineage.getInt()!=index)
    return module.emitError("quantized product member lost its source lineage");
  auto projection=member->getAttrOfType<DictionaryAttr>("tessera.native.quantized_product_projection");
  auto projectedName=member->getAttrOfType<StringAttr>("tessera.native.quantized_product_projection_name");
  if (bool(projection)!=bool(projectedName))
    return module.emitError("quantized product projection lost its native module contract");
  auto child=cloneQuantizedProductMember(member,index);
  if (projection) {
    module.getBody()->clear();module->setAttrs(projection);module.getBody()->push_back(child);
    return success();
  }
  auto arch=module->getAttr("tessera.arch"); OpBuilder b(module.getContext());
  module.getBody()->clear(); module->setAttrs(b.getDictionaryAttr({}));
  bool expansion = child.getBody().front().front().getName().getStringRef() ==
                   "tessera.nvfp4_expand";
  module->setAttr("tessera.target",b.getStringAttr(index==0 || expansion ? "rocm_gfx1201":"rocm"));
  module->setAttr("tessera.arch",arch);
  if (index==0) module->setAttr("tessera.native.block_quant",b.getUnitAttr());
  module.getBody()->push_back(child);
  return success();
}
} // namespace tessera

#include "NativeQuantizedGateUpProgram.h"
