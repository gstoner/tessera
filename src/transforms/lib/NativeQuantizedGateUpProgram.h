// Compose verified quantizer and NVFP4 store/view regions from original SSA.
#pragma once
#include "mlir/IR/OwningOpRef.h"

namespace tessera {
static mlir::OwningOpRef<mlir::ModuleOp> cloneQuantizedValidationRegion(
    mlir::ModuleOp source, mlir::func::FuncOp root,
    llvm::ArrayRef<mlir::Operation *> operations,
    llvm::ArrayRef<mlir::Value> inputs, llvm::ArrayRef<mlir::Value> outputs,
    bool quantized) {
  using namespace mlir;
  auto copy = ModuleOp::create(source.getLoc());
  OwningOpRef<ModuleOp> owner(copy);
  OpBuilder b(source.getContext());
  if (quantized) copy->setAttrs(source->getAttrs());
  else {
    copy->setAttr("tessera.target", b.getStringAttr("rocm_gfx1201"));
    copy->setAttr("tessera.arch", b.getStringAttr("gfx1201"));
  }
  SmallVector<Type> ins, outs;
  for (Value value : inputs) ins.push_back(value.getType());
  for (Value value : outputs) outs.push_back(value.getType());
  b.setInsertionPointToEnd(copy.getBody());
  auto fn = b.create<func::FuncOp>(root.getLoc(), root.getSymName(), b.getFunctionType(ins,outs));
  // Preserve caller metadata so the shared quantizer admission can reject
  // unknown function semantics rather than silently dropping them.
  for (auto attr : root->getAttrs())
    if (attr.getName() != "sym_name" && attr.getName() != "function_type")
      fn->setAttr(attr.getName(),attr.getValue());
  auto block = fn.addEntryBlock(); OpBuilder body(block,block->begin()); IRMapping mapping;
  for (auto [value,arg] : llvm::zip(inputs,block->getArguments())) mapping.map(value,arg);
  for (Operation *op : operations) body.clone(*op,mapping);
  SmallVector<Value> returned;
  for (Value value : outputs) returned.push_back(mapping.lookup(value));
  body.create<func::ReturnOp>(root.getLoc(),returned);
  return owner;
}

static mlir::LogicalResult outlineNativeQuantizedMembers(
    mlir::ModuleOp module, mlir::func::FuncOp root,
    llvm::ArrayRef<llvm::SmallVector<mlir::Operation *>> groups,
    llvm::SmallVectorImpl<mlir::OwningOpRef<mlir::ModuleOp>> &projections,
    llvm::StringRef schema, llvm::ArrayRef<llvm::StringRef> subprograms = {}) {
  using namespace mlir;
  auto refuse = [&]() -> LogicalResult {
    return module.emitError("native quantized member interfaces and SSA lifetimes must match their certified projections");
  };
  auto ret=cast<func::ReturnOp>(root.getBody().front().back());
  SmallVector<SmallVector<Value>> inputs,outputs;
  // Derive the complete physical frame from actual SSA uses. Views/casts stay
  // inside members; the FP32 product has no external use and no allocation.
  for (const auto &group : groups) {
    llvm::SmallDenseSet<Value> internal;
    SmallVector<Value> in,out;
    for (Operation *op : group) {
      for (Value value : op->getOperands())
        if (!internal.contains(value) && !llvm::is_contained(in,value)) in.push_back(value);
      for (Value value : op->getResults()) internal.insert(value);
    }
    for (Operation *op : group)
      for (Value value : op->getResults())
        if (llvm::any_of(value.getUsers(),[&](Operation *user) {return !llvm::is_contained(group,user);}))
          out.push_back(value);
    inputs.push_back(std::move(in)); outputs.push_back(std::move(out));
  }
  for (unsigned i=0;i<groups.size();++i) {
    auto functions=projections[i].get().getOps<func::FuncOp>();
    if (llvm::range_size(functions)!=1) return refuse();
    auto fn=*functions.begin(); SmallVector<Type> ins,outs;
    for (Value value:inputs[i]) ins.push_back(value.getType());
    for (Value value:outputs[i]) outs.push_back(value.getType());
    if (!llvm::equal(fn.getFunctionType().getInputs(),ins) ||
        !llvm::equal(fn.getFunctionType().getResults(),outs)) return refuse();
  }
  llvm::DenseMap<Value,int64_t> ids; SmallVector<Value> values;
  SmallVector<int64_t> writes,reads;
  for (Value value:root.getArguments()) {
    ids[value]=values.size();values.push_back(value);writes.push_back(-1);reads.push_back(-1);
  }
  for (unsigned i=0;i<groups.size();++i) {
    for (Value value:inputs[i]) {
      if (!ids.contains(value)) return refuse();
      reads[ids.lookup(value)]=i;
    }
    for (Value value:outputs[i]) {
      if (ids.contains(value)) return refuse();
      ids[value]=values.size();values.push_back(value);writes.push_back(i);reads.push_back(i);
    }
  }
  if (!ids.contains(ret.getOperand(0))) return refuse();
  reads[ids.lookup(ret.getOperand(0))]=groups.size();
  for (unsigned i=0;i<root.getNumArguments();++i)
    if (reads[i]<0) return refuse();
  llvm::json::Array buffers,steps,members;
  for (auto [i,value]:llvm::enumerate(values)) {
    auto type=dyn_cast<RankedTensorType>(value.getType());
    if (!type || !type.hasStaticShape() || type.getEncoding()) return refuse();
    int64_t bits=type.getElementType().getIntOrFloatBitWidth();
    if (bits<=0 || bits%8) return refuse();
    int64_t bytes=bits/8;llvm::json::Array shape;
    for (int64_t dim:type.getShape()) {
      if (dim<=0 || dim>INT64_MAX/bytes) return refuse();
      bytes*=dim;shape.push_back(dim);
    }
    std::string storage;llvm::raw_string_ostream os(storage);type.getElementType().print(os);os.flush();
    buffers.push_back(llvm::json::Object{{"id",int64_t(i)},{"bytes",bytes},{"shape",std::move(shape)},
      {"storage",storage},{"ownership",i<root.getNumArguments()?"readonly_input":value==ret.getOperand(0)?"returned_output":"private_scratch"},
      {"first_write",writes[i]},{"last_read",reads[i]}});
  }
  std::string source;llvm::raw_string_ostream sourceStream(source);module.print(sourceStream);sourceStream.flush();
  for (unsigned i=0;i<groups.size();++i)
    if (SymbolTable::lookupSymbolIn(module,(root.getSymName()+"__quantized_member_"+llvm::Twine(i)).str())) return refuse();
  OpBuilder b(module.getContext());SmallVector<Attribute> symbols;
  for (unsigned i=0;i<groups.size();++i) {
    auto projection=projections[i].get();auto certified=*projection.getOps<func::FuncOp>().begin();
    auto name=(root.getSymName()+"__quantized_member_"+llvm::Twine(i)).str();
    b.setInsertionPointToEnd(module.getBody());
    auto member=b.create<func::FuncOp>(root.getLoc(),name,certified.getFunctionType());member.setPrivate();
    for (auto attr:certified->getAttrs())
      if (attr.getName()!="sym_name" && attr.getName()!="function_type" && attr.getName()!="sym_visibility")
        member->setAttr(attr.getName(),attr.getValue());
    member->setAttr("tessera.native.quantized_product_member",b.getI64IntegerAttr(i));
    member->setAttr("tessera.native.quantized_product_projection",b.getDictionaryAttr(projection->getAttrs()));
    member->setAttr("tessera.native.quantized_product_projection_name",certified.getSymNameAttr());
    symbols.push_back(FlatSymbolRefAttr::get(member));
    auto block=member.addEntryBlock();OpBuilder body(block,block->begin());IRMapping mapping;
    llvm::json::Array inIDs,outIDs;
    for (auto [value,arg]:llvm::zip(inputs[i],block->getArguments())) {mapping.map(value,arg);inIDs.push_back(ids.lookup(value));}
    for (Operation *op:groups[i]) body.clone(*op,mapping);
    SmallVector<Value> returned;
    for (Value value:outputs[i]) {returned.push_back(mapping.lookup(value));outIDs.push_back(ids.lookup(value));}
    body.create<func::ReturnOp>(root.getLoc(),returned);
    auto projected=ModuleOp::create(module.getLoc());projected->setAttrs(projection->getAttrs());
    projected.getBody()->push_back(cloneQuantizedProductMember(member,i));
    std::string text;llvm::raw_string_ostream os(text);projected.print(os);os.flush();members.push_back(text+"\n");projected.erase();
    steps.push_back(llvm::json::Object{{"step",int64_t(i)},{"operation",groups[i].size()==13?"native_block_quant":groups[i].size()==6?"tessera.silu_mul":groups[i][0]->getName().getStringRef()},
      {"member",name},{"inputs",std::move(inIDs)},{"outputs",std::move(outIDs)}});
  }
  llvm::json::Object plan{{"schema",schema},{"source_graph_ir",source+"\n"},
    {"member_graphs",std::move(members)},{"buffers",std::move(buffers)},{"steps",std::move(steps)},
    {"output",ids.lookup(ret.getOperand(0))}};
  if (!subprograms.empty()) {
    llvm::json::Array regions;
    for (auto value:subprograms) regions.push_back(value);
    plan["subprograms"]=std::move(regions);
  }
  std::string json;llvm::raw_string_ostream os(json);os<<llvm::json::Value(std::move(plan));os.flush();
  module->setAttr("tessera.native.quantized_product_json",b.getStringAttr(llvm::encodeBase64(json)));
  module->setAttr("tessera.native.quantized_product_members",b.getArrayAttr(symbols));
  return success();
}

static mlir::LogicalResult emitNativeQuantizedGateUpProgram(mlir::ModuleOp module) {
  using namespace mlir;
  auto refuse = [&]() -> LogicalResult {
    return module.emitError("quantized gate/up requires original quantizer, NVFP4 expansion, FP32 product, BF16 store and view/SiLU SSA");
  };
  auto root = *module.getOps<func::FuncOp>().begin();
  SmallVector<Operation *> ops;
  for (Operation &op : root.getBody().front().without_terminator()) ops.push_back(&op);
  auto ret = dyn_cast<func::ReturnOp>(root.getBody().front().back());
  if (root.getNumArguments()!=4 || ops.size()!=22 || !isa<ScaledMatmulOp>(ops[14]) ||
      !ret || ret.getNumOperands()!=1 || ops[21]->getNumResults()!=1 ||
      ret.getOperand(0)!=ops[21]->getResult(0)) return refuse();
  SmallVector<Value> rootInputs(root.getArguments().begin(),root.getArguments().end());
  SmallVector<Value> prefixOutputs{ops[14]->getResult(0)};
  auto prefix = cloneQuantizedValidationRegion(module,root,
      ArrayRef<Operation *>(ops).take_front(15),rootInputs,prefixOutputs,true);
  // Reuse the complete v2 semantic admission, not a second quantizer matcher.
  if (failed(emitNativeQuantizedProductProgram(prefix.get()))) return failure();
  SmallVector<Value> tailInputs(ops[13]->getOperands().begin(),ops[13]->getOperands().end());
  tailInputs.push_back(ops[11]->getResult(0)); tailInputs.push_back(ops[12]->getResult(0));
  SmallVector<Value> tailOutputs{ret.getOperand(0)};
  auto tail = cloneQuantizedValidationRegion(module,root,
      ArrayRef<Operation *>(ops).drop_front(13),tailInputs,tailOutputs,false);
  // The existing native v5 validator owns explicit BF16 store semantics,
  // view aliasing, SiLU operand orientation and the activation store.
  if (failed(emitNativeNVFP4Program(tail.get()))) return failure();
  auto tailPlan=tail.get()->getAttrOfType<DictionaryAttr>("tessera.native.nvfp4_program");
  auto tailMembers=tailPlan ? tailPlan.getAs<ArrayAttr>("members") : ArrayAttr{};
  if (!tailMembers || tailMembers.size()!=3) return refuse();
  SmallVector<OwningOpRef<ModuleOp>> projections;
  {
    OwningOpRef<ModuleOp> copy(cast<ModuleOp>(prefix.get()->clone()));
    if (failed(projectNativeQuantizedProductMember(copy.get(),0))) return failure();
    projections.push_back(std::move(copy));
  }
  for (int64_t i=0;i<3;++i) {
    OwningOpRef<ModuleOp> copy(cast<ModuleOp>(tail.get()->clone()));
    if (failed(projectNativeNVFP4Member(copy.get(),i))) return failure();
    projections.push_back(std::move(copy));
  }
  SmallVector<SmallVector<Operation *>> groups;
  groups.emplace_back(ops.begin(),ops.begin()+13);
  groups.push_back({ops[13]}); groups.push_back({ops[14],ops[15]});
  groups.emplace_back(ops.begin()+16,ops.end());
  return outlineNativeQuantizedMembers(module,root,groups,projections,
      "tessera.native.quantized_product.v3");
}

static mlir::LogicalResult emitNativeQuantizedMLPProgram(mlir::ModuleOp module) {
  using namespace mlir;
  auto refuse = [&]() -> LogicalResult {
    return module.emitError("native quantized MLP requires original gate/up, BF16 activation quantizer and exact down-projection SSA");
  };
  auto root=*module.getOps<func::FuncOp>().begin();
  SmallVector<Operation *> ops;
  for (Operation &op:root.getBody().front().without_terminator()) ops.push_back(&op);
  auto ret=dyn_cast<func::ReturnOp>(root.getBody().front().back());
  if (root.getNumArguments()!=7 || ops.size()!=37 || !ret ||
      ops[0]->getNumOperands()!=1 || ops[13]->getNumOperands()!=3 ||
      ops[21]->getNumResults()!=1 || ops[22]->getNumOperands()!=1 ||
      ops[22]->getOperand(0)!=ops[21]->getResult(0) ||
      ops[35]->getNumOperands()!=3 || ops[36]->getNumResults()!=1 ||
      ret.getNumOperands()!=1 || ret.getOperand(0)!=ops[36]->getResult(0)) return refuse();
  SmallVector<Value> gateInputs{ops[0]->getOperand(0)};
  llvm::append_range(gateInputs,ops[13]->getOperands());
  SmallVector<Value> gateOutputs{ops[21]->getResult(0)};
  auto gate=cloneQuantizedValidationRegion(module,root,
      ArrayRef<Operation *>(ops).take_front(22),gateInputs,gateOutputs,true);
  if (failed(emitNativeQuantizedProductProgram(gate.get()))) return failure();
  SmallVector<Value> downInputs{ops[21]->getResult(0)};
  llvm::append_range(downInputs,ops[35]->getOperands());
  SmallVector<Value> downOutputs{ret.getOperand(0)};
  auto down=cloneQuantizedValidationRegion(module,root,
      ArrayRef<Operation *>(ops).drop_front(22),downInputs,downOutputs,true);
  if (failed(emitNativeQuantizedProductProgram(down.get()))) return failure();
  SmallVector<OwningOpRef<ModuleOp>> projections;
  for (int64_t i=0;i<7;++i) {
    auto source=i<4?gate.get():down.get();
    OwningOpRef<ModuleOp> copy(cast<ModuleOp>(source->clone()));
    if (failed(projectNativeQuantizedProductMember(copy.get(),i<4?i:i-4))) return failure();
    projections.push_back(std::move(copy));
  }
  SmallVector<SmallVector<Operation *>> groups;
  groups.emplace_back(ops.begin(),ops.begin()+13);
  groups.push_back({ops[13]});groups.push_back({ops[14],ops[15]});
  groups.emplace_back(ops.begin()+16,ops.begin()+22);
  groups.emplace_back(ops.begin()+22,ops.begin()+35);
  groups.push_back({ops[35]});groups.push_back({ops[36]});
  SmallVector<llvm::StringRef> subprograms{
      gate.get()->getAttrOfType<StringAttr>("tessera.native.quantized_product_json").getValue(),
      down.get()->getAttrOfType<StringAttr>("tessera.native.quantized_product_json").getValue()};
  return outlineNativeQuantizedMembers(module,root,groups,projections,
      "tessera.native.quantized_product.v4",subprograms);
}

} // namespace tessera
