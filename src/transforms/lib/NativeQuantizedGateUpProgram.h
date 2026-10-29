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
    llvm::StringRef schema, llvm::ArrayRef<llvm::StringRef> subprograms = {},
    llvm::function_ref<mlir::LogicalResult(llvm::json::Object &,
        const llvm::DenseMap<mlir::Value,int64_t> &)> extend = nullptr) {
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
  if (extend && failed(extend(plan,ids))) return failure();
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

// Capacity retyping for the admitted seven-member MLP. Every operation in the
// activation's forward slice is row-local: K16 row blocks, an axis-1 reduce,
// elementwise math, column slices of the gate/up parent and row-major scaled
// products. Running that program at capacity R and keeping its first m rows is
// the program at m rows, which is the claim the bounded owner relies on. Each
// leading dimension in the slice is a multiple of m and scales by R/m; any
// other operation, a row-axis reduce/slice, or a shared constant refuses.
static mlir::LogicalResult retypeQuantizedMLPRows(
    mlir::ModuleOp module, mlir::func::FuncOp root, mlir::Value activation,
    int64_t rows, int64_t capacity, llvm::DenseSet<mlir::Value> &rowValues) {
  using namespace mlir;
  auto refuse = [&]() -> LogicalResult {
    return module.emitError("quantized bounded rows require row-local MLP SSA with row-multiple leading dimensions");
  };
  auto scaled = [&](int64_t dim, int64_t &out) {
    if (dim <= 0 || dim % rows || dim / rows > INT64_MAX / capacity) return false;
    out = dim / rows * capacity; return true;
  };
  auto scaledType = [&](Type type) -> RankedTensorType {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape() || tensor.getEncoding() || !tensor.getRank())
      return {};
    SmallVector<int64_t> shape(tensor.getShape());
    if (!scaled(shape[0], shape[0])) return {};
    return RankedTensorType::get(shape, tensor.getElementType());
  };
  auto scaledArray = [&](Operation *op, StringRef key) -> LogicalResult {
    auto array = op->getAttrOfType<ArrayAttr>(key);
    if (!array || array.empty()) return failure();
    SmallVector<Attribute> values(array.begin(), array.end());
    auto first = dyn_cast<IntegerAttr>(values[0]);
    int64_t next = 0;
    if (!first || !scaled(first.getInt(), next)) return failure();
    values[0] = IntegerAttr::get(first.getType(), next);
    op->setAttr(key, ArrayAttr::get(op->getContext(), values));
    return success();
  };
  rowValues.insert(activation);
  SmallVector<Operation *> slice;
  for (Operation &op : root.getBody().front().without_terminator())
    if (llvm::any_of(op.getOperands(), [&](Value v) { return rowValues.contains(v); })) {
      slice.push_back(&op);
      for (Value result : op.getResults()) rowValues.insert(result);
    }
  llvm::SetVector<Operation *> constants;
  for (Operation *op : slice) {
    StringRef name = op->getName().getStringRef();
    if (name != "tessera.reshape" && name != "tessera.cast" && name != "tessera.absolute" &&
        name != "tessera.reduce" && name != "tessera.div" && name != "tessera.maximum" &&
        name != "tessera.expand" && name != "tessera.scaled_matmul" &&
        name != "tessera.slice" && name != "tessera.silu_mul") return refuse();
    if (name == "tessera.reduce") {
      auto axis = op->getAttrOfType<IntegerAttr>("axis");
      if (!axis || axis.getInt() == 0) return refuse();
    }
    if (name == "tessera.slice") {
      auto start = op->getAttrOfType<ArrayAttr>("start_indices");
      if (!start || start.empty() || !isa<IntegerAttr>(start[0]) ||
          cast<IntegerAttr>(start[0]).getInt() != 0) return refuse();
    }
    for (Value operand : op->getOperands()) {
      if (rowValues.contains(operand)) continue;
      auto constant = operand.getDefiningOp<arith::ConstantOp>();
      if (!constant) continue;  // Invariant weights stay at their static shape.
      auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (!dense || !dense.isSplat() ||
          llvm::any_of(constant->getUsers(), [&](Operation *user) {
            return !llvm::is_contained(slice, user); })) return refuse();
      constants.insert(constant);
    }
  }
  for (Operation *op : constants) {
    auto constant = cast<arith::ConstantOp>(op);
    auto type = scaledType(constant.getType());
    if (!type) return refuse();
    // Retyped in place: the caller's matched operation list stays valid.
    auto dense = cast<DenseElementsAttr>(constant.getValue());
    constant.setValueAttr(DenseElementsAttr::get(type, dense.getSplatValue<Attribute>()));
    constant.getResult().setType(type);
  }
  auto argument = cast<BlockArgument>(activation);
  auto argumentType = scaledType(argument.getType());
  if (!argumentType) return refuse();
  argument.setType(argumentType);
  for (Operation *op : slice) {
    StringRef name = op->getName().getStringRef();
    if ((name == "tessera.reshape" || name == "tessera.expand") && failed(scaledArray(op, "shape")))
      return refuse();
    if (name == "tessera.slice" && failed(scaledArray(op, "slice_sizes"))) return refuse();
    for (Value result : op->getResults()) {
      auto type = scaledType(result.getType());
      if (!type) return refuse();
      result.setType(type);
    }
  }
  auto ret = cast<func::ReturnOp>(root.getBody().front().back());
  SmallVector<Type> inputs, results;
  for (Value value : root.getArguments()) inputs.push_back(value.getType());
  for (Value value : ret.getOperands()) {
    if (!rowValues.contains(value)) return refuse();
    results.push_back(value.getType());
  }
  root.setFunctionType(FunctionType::get(module.getContext(), inputs, results));
  return verify(module);
}

// Capacity retyping of the gate contraction axis. K reaches exactly the
// activation, the gate quantizer (flattened K16 blocks [G,16]/[G,1] whose G is
// M*K/16, and its [M,K] codes / [M,K/16] scales), the gate weights' packed
// [N,K/2] and scale [N,K/16] planes and the expansion's [N,K] values /
// [K/16,N] scale plane. The scaled product contracts it away. Each such
// extent scales by kCap/k exactly; anything else refuses, and the ordinary
// static admission then certifies the capacity program.
static mlir::LogicalResult retypeQuantizedMLPContraction(
    mlir::ModuleOp module, mlir::func::FuncOp root,
    llvm::ArrayRef<mlir::Operation *> ops, int64_t k, int64_t capacity,
    llvm::DenseSet<mlir::Value> &contractionValues) {
  using namespace mlir;
  auto refuse = [&]() -> LogicalResult {
    return module.emitError("quantized bounded K requires the gate quantizer, expansion and product contraction SSA");
  };
  auto scaledDim = [&](int64_t dim, int64_t &out) {
    if (dim <= 0 || dim > INT64_MAX / capacity || (dim * capacity) % k) return false;
    out = dim * capacity / k; return true;
  };
  auto retype = [&](Value value, unsigned axis) -> LogicalResult {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || type.getRank() != 2 || !type.hasStaticShape() || type.getEncoding())
      return failure();
    SmallVector<int64_t> shape(type.getShape());
    if (!scaledDim(shape[axis], shape[axis])) return failure();
    value.setType(RankedTensorType::get(shape, type.getElementType()));
    contractionValues.insert(value);
    return success();
  };
  auto shapeAttr = [&](Operation *op) {
    auto type = cast<RankedTensorType>(op->getResult(0).getType());
    auto old = op->getAttrOfType<ArrayAttr>("shape");
    if (!old || old.size() != 2 || !isa<IntegerAttr>(old[0])) return failure();
    auto element = cast<IntegerAttr>(old[0]).getType();
    op->setAttr("shape", ArrayAttr::get(op->getContext(),
        {IntegerAttr::get(element, type.getDimSize(0)), IntegerAttr::get(element, type.getDimSize(1))}));
    return success();
  };
  Operation *expand = ops[13], *product = ops[14];
  if (expand->getName().getStringRef() != "tessera.nvfp4_expand" ||
      product->getName().getStringRef() != "tessera.scaled_matmul" ||
      expand->getNumOperands() != 3 || expand->getNumResults() != 2 ||
      product->getNumOperands() != 4 ||
      product->getOperand(1) != expand->getResult(0) ||
      product->getOperand(3) != expand->getResult(1)) return refuse();
  Value codes = product->getOperand(0), scales = product->getOperand(2);
  auto activation = dyn_cast<BlockArgument>(ops[0]->getOperand(0));
  if (!activation || failed(retype(activation, 1))) return refuse();
  for (Operation *op : ArrayRef<Operation *>(ops).take_front(13)) {
    StringRef name = op->getName().getStringRef();
    if (op->getNumResults() != 1) return refuse();
    Value result = op->getResult(0);
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (!dense || !dense.isSplat() || failed(retype(result, 0)) ||
          llvm::any_of(result.getUsers(), [&](Operation *user) {
            return !llvm::is_contained(ops.take_front(13), user); })) return refuse();
      constant.setValueAttr(DenseElementsAttr::get(
          cast<ShapedType>(result.getType()), dense.getSplatValue<Attribute>()));
      continue;
    }
    // Contraction-axis outputs feed the product; the K16 block chain is [G,c].
    bool rowMajor = result == codes || result == scales;
    if (!rowMajor) {
      auto type = dyn_cast<RankedTensorType>(result.getType());
      if (!type || type.getRank() != 2 ||
          (type.getDimSize(1) != 16 && type.getDimSize(1) != 1)) return refuse();
    }
    if (failed(retype(result, rowMajor ? 1 : 0))) return refuse();
    if ((name == "tessera.reshape" || name == "tessera.expand") && failed(shapeAttr(op)))
      return refuse();
    if (name == "tessera.reduce") {
      auto axis = op->getAttrOfType<IntegerAttr>("axis");
      if (!axis || axis.getInt() != 1) return refuse();
    } else if (name != "tessera.reshape" && name != "tessera.cast" &&
               name != "tessera.absolute" && name != "tessera.div" &&
               name != "tessera.maximum" && name != "tessera.expand") return refuse();
  }
  if (!contractionValues.contains(codes) || !contractionValues.contains(scales))
    return refuse();
  auto packed = dyn_cast<BlockArgument>(expand->getOperand(0));
  auto blockScales = dyn_cast<BlockArgument>(expand->getOperand(1));
  if (!packed || !blockScales || failed(retype(packed, 1)) ||
      failed(retype(blockScales, 1)) || failed(retype(expand->getResult(0), 1)) ||
      failed(retype(expand->getResult(1), 0))) return refuse();
  auto ret = cast<func::ReturnOp>(root.getBody().front().back());
  SmallVector<Type> inputs, results;
  for (Value value : root.getArguments()) inputs.push_back(value.getType());
  for (Value value : ret.getOperands()) results.push_back(value.getType());
  root.setFunctionType(FunctionType::get(module.getContext(), inputs, results));
  return verify(module);
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
  // A bounded request retypes the whole row path to its capacity before the
  // ordinary static admission runs; the gate/down validators then certify the
  // capacity program and never see the request.
  constexpr StringLiteral rowBoundKey = "tessera.native.quantized_m_bound";
  constexpr StringLiteral contractionBoundKey = "tessera.native.quantized_k_bound";
  bool boundedRows = module->hasAttr(rowBoundKey);
  bool boundedK = module->hasAttr(contractionBoundKey);
  int64_t activeRows = 0, rowBound = 0, activeK = 0, kBound = 0;
  std::string originalIR;
  llvm::DenseSet<Value> rowValues, contractionValues;
  // A K bound is a refinement of the row frame; it never travels alone.
  if (boundedK && !boundedRows)
    return module.emitError("quantized bounded K requires an M bound");
  if (boundedK) {
    auto bound = module->getAttrOfType<IntegerAttr>(contractionBoundKey);
    auto activation = ops.size() < 15 || ops[0]->getNumOperands() != 1 ? Value{} : ops[0]->getOperand(0);
    auto type = activation ? dyn_cast<RankedTensorType>(activation.getType()) : RankedTensorType{};
    if (!bound || !bound.getType().isInteger(64) || !type || type.getRank() != 2 ||
        !type.hasStaticShape())
      return module.emitError("quantized bounded K requires a static rank-2 activation and an i64 bound");
    activeK = type.getDimSize(1); kBound = bound.getInt();
    if (activeK <= 0 || activeK % 16 || kBound < activeK || kBound % 16)
      return module.emitError("quantized bounded K exceeds its checked K16 capacity");
  }
  if (boundedRows) {
    auto bound = module->getAttrOfType<IntegerAttr>(rowBoundKey);
    auto activation = ops.empty() || ops[0]->getNumOperands() != 1 ? Value{} : ops[0]->getOperand(0);
    auto type = activation ? dyn_cast<RankedTensorType>(activation.getType()) : RankedTensorType{};
    if (!bound || !bound.getType().isInteger(64) || !type || type.getRank() != 2 ||
        !type.hasStaticShape() || !isa<BlockArgument>(activation))
      return module.emitError("quantized bounded rows require a static rank-2 activation and an i64 bound");
    activeRows = type.getDimSize(0); rowBound = bound.getInt();
    if (activeRows <= 0 || rowBound < activeRows)
      return module.emitError("quantized bounded rows exceed their checked capacity");
    llvm::raw_string_ostream originalStream(originalIR);
    module.print(originalStream); originalStream.flush();
    module->removeAttr(rowBoundKey);
    module->removeAttr(contractionBoundKey);
    if (failed(retypeQuantizedMLPRows(module, root, activation, activeRows, rowBound, rowValues)))
      return failure();
    if (boundedK && failed(retypeQuantizedMLPContraction(
            module, root, ops, activeK, kBound, contractionValues)))
      return failure();
  }
  if (root.getNumArguments()!=7 || (ops.size()!=37 && ops.size()!=38) || !ret ||
      ops[0]->getNumOperands()!=1 || ops[13]->getNumOperands()!=3 ||
      ops[21]->getNumResults()!=1 || ops[22]->getNumOperands()!=1 ||
      ops[22]->getOperand(0)!=ops[21]->getResult(0) ||
      ops[35]->getNumOperands()!=3 || ops[36]->getNumResults()!=1 || ops.back()->getNumResults()!=1 ||
      ret.getNumOperands()!=1 || ret.getOperand(0)!=ops.back()->getResult(0)) return refuse();
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
  groups.push_back({ops[35]});
  groups.emplace_back(ops.begin()+36,ops.end());
  SmallVector<llvm::StringRef> subprograms{
      gate.get()->getAttrOfType<StringAttr>("tessera.native.quantized_product_json").getValue(),
      down.get()->getAttrOfType<StringAttr>("tessera.native.quantized_product_json").getValue()};
  if (!boundedRows)
    return outlineNativeQuantizedMembers(module,root,groups,projections,
        "tessera.native.quantized_product.v4",subprograms);
  // Slots and steps a frame variable reaches, derived from SSA ids.
  auto frameOf = [](const llvm::DenseSet<Value> &variable,
                    const llvm::DenseMap<Value,int64_t> &ids, llvm::json::Object &plan,
                    llvm::json::Array &slotJSON, llvm::json::Array &stepJSON) {
    SmallVector<int64_t> slots;
    for (auto &[value,id] : ids)
      if (variable.contains(value)) slots.push_back(id);
    llvm::sort(slots);
    for (int64_t id : slots) slotJSON.push_back(id);
    for (auto &step : *plan.getArray("steps")) {
      auto *object = step.getAsObject();
      bool touches = false;
      for (auto key : {"inputs","outputs"})
        for (auto &id : *object->getArray(key))
          touches |= llvm::is_contained(slots, *id.getAsInteger());
      if (touches) stepJSON.push_back(*object->getInteger("step"));
    }
  };
  // v5 is v4 at capacity plus the declared row frame. v4 is never repurposed.
  auto frame = [&](llvm::json::Object &plan, const llvm::DenseMap<Value,int64_t> &ids) -> LogicalResult {
    SmallVector<std::pair<int64_t,Value>> slots;
    for (auto &[value,id] : ids)
      if (rowValues.contains(value)) slots.push_back({id,value});
    llvm::sort(slots, [](auto &a, auto &b) { return a.first < b.first; });
    llvm::json::Array slotJSON, strideJSON, stepJSON;
    for (auto &[id,value] : slots) {
      auto type = cast<RankedTensorType>(value.getType());
      int64_t bytes = type.getNumElements() * (type.getElementType().getIntOrFloatBitWidth() / 8);
      if (bytes % rowBound) return module.emitError("quantized bounded row slot is not row-major");
      slotJSON.push_back(id); strideJSON.push_back(bytes / rowBound);
    }
    auto *steps = plan.getArray("steps");
    for (auto &step : *steps) {
      auto *object = step.getAsObject();
      bool touches = false;
      for (auto key : {"inputs","outputs"})
        for (auto &id : *object->getArray(key))
          touches |= llvm::any_of(slots, [&](auto &slot) { return slot.first == *id.getAsInteger(); });
      if (touches) stepJSON.push_back(*object->getInteger("step"));
    }
    plan["active_m"] = activeRows;
    plan["m_bound"] = rowBound;
    plan["original_graph_ir"] = originalIR + "\n";
    plan["row_frame"] = llvm::json::Object{{"slots",std::move(slotJSON)},
        {"row_strides_bytes",std::move(strideJSON)},{"row_steps",std::move(stepJSON)}};
    if (boundedK) {
      // v6 is v5 at capacity K plus the contraction frame. Its K16 group
      // count is the second frame variable of the shape-frame owner.
      llvm::json::Array kSlots, kSteps;
      frameOf(contractionValues, ids, plan, kSlots, kSteps);
      plan["active_k"] = activeK;
      plan["k_bound"] = kBound;
      plan["k_frame"] = llvm::json::Object{{"slots",std::move(kSlots)},{"k_steps",std::move(kSteps)}};
    }
    return success();
  };
  return outlineNativeQuantizedMembers(module,root,groups,projections,
      boundedK ? "tessera.native.quantized_product.v6" : "tessera.native.quantized_product.v5",
      subprograms,frame);
}

} // namespace tessera
