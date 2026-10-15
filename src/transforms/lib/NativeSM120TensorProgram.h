// Compiler-owned outlining of actual SM120 tensor producer/matmul Graph edges.
#pragma once
#include "mlir/IR/Verifier.h"

namespace tessera {
static bool hasNativeSM120TensorGraph(mlir::ModuleOp module) {
  auto target = module->getAttrOfType<mlir::StringAttr>("tessera.target");
  if (!target || target.getValue() != "nvidia_sm120") return false;
  bool found = false;
  module.walk([&](mlir::Operation *op) {
    auto name = op->getName().getStringRef();
    found |= name == "tessera.rmsnorm" || name == "tessera.layer_norm" || name == "tessera.softmax";
    // ReLU selects this exporter only as an upstream ordinary matmul
    // producer. A post-product activation must keep its own export route.
    if (name == "tessera.matmul") {
      for (unsigned i = 0; i < 2 && i < op->getNumOperands(); ++i) {
        mlir::Value value = op->getOperand(i);
        while (mlir::Operation *producer = value.getDefiningOp()) {
          auto kind = producer->getName().getStringRef();
          if (producer->getNumOperands() != 1) break;
          if (kind == "tessera.relu") { found = true; break; }
          if (kind != "tessera.rmsnorm" && kind != "tessera.layer_norm" &&
              kind != "tessera.softmax") break;
          value = producer->getOperand(0);
        }
      }
    }
  });
  return found;
}

static mlir::func::FuncOp cloneNativeSM120TensorMember(
    mlir::func::FuncOp member, int64_t index) {
  using namespace mlir;
  auto child = cast<func::FuncOp>(member->clone());
  child.setPublic();
  child.setSymName(child.getBody().front().front().getName().getStringRef() != "tessera.matmul" ? "native_sm120_tensor_producer" :
                   "nvidia_sm120_scheduled_matmul_native_tensor_consumer");
  child->removeAttr("tessera.native.sm120_tensor_member");
  return child;
}

static mlir::LogicalResult emitNativeSM120TensorProgram(mlir::ModuleOp module, bool paired = false,
    llvm::StringRef originalSource = {}) {
  using namespace mlir;
  if (failed(verify(module))) return failure();
  if (module->hasAttr("tessera.native.sm120_tensor_program"))
    return module.emitError("SM120 tensor program export already exists");
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  SmallVector<func::FuncOp> roots;
  for (auto fn : module.getOps<func::FuncOp>())
    if (!fn.isDeclaration() && (!paired ||
        (fn->getAttrOfType<StringAttr>("tessera.autodiff.role") &&
         fn->getAttrOfType<StringAttr>("tessera.autodiff.role").getValue() == "jvp")))
      roots.push_back(fn);
  if (!arch || arch.getValue() != "sm_120" || roots.size() != 1 ||
      !roots[0].getBody().hasOneBlock())
    return module.emitError("SM120 tensor program requires one explicit sm_120 Graph root");
  auto root = roots[0];
  if (root.getNumArguments() < 2 || root.getNumArguments() > 4 ||
      root.getNumResults() != (paired ? 2u : 1u) ||
      (paired && (root.getNumArguments() != 3 || originalSource.empty())) ||
      (root.getBody().front().getOperations().size() < (paired ? 4u : 3u) ||
       root.getBody().front().getOperations().size() > 65))
    return root.emitError("SM120 tensor program requires its producer/matmul Graph");
  SmallVector<Operation *> ops;
  for (auto &op : root.getBody().front().without_terminator()) ops.push_back(&op);
  const size_t producerCount = ops.size() - (paired ? 2 : 1);
  auto consumer = ops[producerCount];
  auto ret = dyn_cast<func::ReturnOp>(root.getBody().front().back());
  if (consumer->getName().getStringRef() != "tessera.matmul" ||
      consumer->getNumResults() != 1 ||
      consumer->getNumOperands() != (paired ? 2u : root.getNumArguments()) ||
      !ret || ret.getNumOperands() != (paired ? 2u : 1u) || ret.getOperand(0) != consumer->getResult(0))
    return root.emitError("SM120 tensor program lost its original producer/consumer SSA edge");
  if (paired) {
    auto tangent = ops.back();
    auto rhs = dyn_cast<BlockArgument>(consumer->getOperand(1));
    auto parentRef = root->getAttrOfType<FlatSymbolRefAttr>("tessera.autodiff.forward");
    auto parent = parentRef ? dyn_cast_or_null<func::FuncOp>(
        SymbolTable::lookupSymbolIn(module,parentRef)) : func::FuncOp{};
    auto wrt = parent ? parent->getAttrOfType<ArrayAttr>("tessera.autodiff.wrt_indices") : ArrayAttr{};
    auto role = wrt && wrt.size() == 1 ? dyn_cast<IntegerAttr>(wrt[0]) : IntegerAttr{};
    if (!parent || parent.getNumArguments() != 2 || parent.getNumResults() != 1 ||
        !rhs || rhs.getOwner() != &root.getBody().front() || rhs.getArgNumber() >= 2 ||
        !role || role.getInt() != rhs.getArgNumber() ||
        tangent->getName().getStringRef() != "tessera.matmul" ||
        tangent->getNumOperands() != 2 || tangent->getNumResults() != 1 ||
        tangent->getOperand(0) != consumer->getOperand(0) ||
        tangent->getOperand(1) != root.getArgument(2) ||
        tangent->getOperand(1).getType() != consumer->getOperand(1).getType() ||
        tangent->getResult(0).getType() != consumer->getResult(0).getType() ||
        ret.getOperand(1) != tangent->getResult(0))
      return root.emitError("SM120 paired tensor program requires the native RHS-linear shared producer product");
    for (Operation *product : {consumer,tangent}) {
      auto activation = product->getAttrOfType<StringAttr>("activation");
      auto order = product->getAttrOfType<StringAttr>("rhs_storage_order");
      auto transA = product->getAttrOfType<BoolAttr>("transposeA");
      auto transB = product->getAttrOfType<BoolAttr>("transposeB");
      if ((activation && activation.getValue() != "none") ||
          (order && order.getValue() != "row_major") ||
          (transA && transA.getValue()) || (transB && transB.getValue()) ||
          product->hasAttr("bias") || product->hasAttr("residual"))
        return product->emitError("SM120 paired tensor consumers require plain row-major products");
    }
  }
  // Track each producer's actual root and retain topological SSA order.
  // The two operand chains may have different shapes and may be interleaved.
  llvm::DenseMap<Value,Value> origins;
  for (Value arg : root.getArguments()) origins[arg] = arg;
  for (Operation *producer : ArrayRef<Operation *>(ops).take_front(producerCount)) {
    auto name = producer->getName().getStringRef();
    Value input = producer->getNumOperands() == 1 ? producer->getOperand(0) : Value{};
    if ((name != "tessera.rmsnorm" && name != "tessera.layer_norm" && name != "tessera.softmax" && name != "tessera.relu") ||
        !input || producer->getNumResults() != 1 ||
        input.getType() != producer->getResult(0).getType() || !origins.count(input))
      return root.emitError("SM120 tensor producers must preserve storage and topological SSA lineage");
    origins[producer->getResult(0)] = origins.lookup(input);
  }
  Value lhsValue = consumer->getOperand(0), rhsValue = consumer->getOperand(1);
  if (!origins.count(lhsValue) || !origins.count(rhsValue))
    return root.emitError("SM120 tensor matmul operands lost their producer lineage");
  llvm::SmallDenseSet<Operation *> live;
  for (Value value : {lhsValue,rhsValue})
    while (auto *producer = value.getDefiningOp()) {
      live.insert(producer);
      value = producer->getOperand(0);
    }
  if (live.size() != producerCount)
    return root.emitError("SM120 tensor program lost its original producer/consumer SSA edge: producer outside matmul dependency closure");
  auto source = dyn_cast<RankedTensorType>(lhsValue.getType());
  if (!source || source.getRank() != 2 || !source.hasStaticShape() ||
      (!source.getElementType().isF16() && !source.getElementType().isBF16()))
    return root.emitError("SM120 tensor producer requires static rank-two matching f16/bf16 storage");
  const bool dag = rhsValue.getDefiningOp() != nullptr;
  if (paired && dag) return root.emitError("SM120 paired RHS must retain its original argument role");
  if (dag) {
    auto order = consumer->getAttrOfType<StringAttr>("rhs_storage_order");
    if (order && order.getValue() != "row_major")
      return consumer->emitError("computed RHS requires native row-major materialization");
  }
  SmallVector<Value> roles{origins.lookup(lhsValue),origins.lookup(rhsValue)};
  for (Value value : consumer->getOperands().drop_front(2)) roles.push_back(value);
  if (paired) roles.push_back(root.getArgument(2));
  llvm::SmallDenseSet<unsigned> seen;
  llvm::json::Array roleJSON;
  for (Value value : roles) {
    auto arg = dyn_cast<BlockArgument>(value);
    if (!arg || arg.getOwner() != &root.getBody().front() ||
        !seen.insert(arg.getArgNumber()).second)
      return root.emitError("SM120 tensor program requires distinct original argument roles");
    auto attrs = root.getArgAttrDict(arg.getArgNumber());
    if (attrs && !attrs.empty())
      return root.emitError("SM120 tensor argument metadata needs a separate native contract");
    roleJSON.push_back(int64_t(arg.getArgNumber()));
  }
  for (NamedAttribute attr : root->getAttrs()) {
    auto key = attr.getName().getValue();
    if (key != "function_type" && key != "sym_name" && key != "sym_visibility" &&
        key != "arg_attrs" && key != "res_attrs" &&
        !(paired && (key == "tessera.autodiff.forward" || key == "tessera.autodiff.role")) &&
        key != "tessera.frontend.authority" &&
        key != "tessera.structured_cfg.schema" &&
        key != "tessera.structured_cfg.digest" && key != "tessera.structured_cfg.blocks")
      return root.emitError("SM120 tensor function metadata needs a native contract");
  }
  if (auto attrs = root->getAttrOfType<ArrayAttr>("res_attrs"))
    for (Attribute attr : attrs)
      if (auto dict = dyn_cast<DictionaryAttr>(attr); !dict || !dict.empty())
        return root.emitError("SM120 tensor result metadata needs a native contract");
  auto rhs = dyn_cast<RankedTensorType>(roles[1].getType());
  auto out = dyn_cast<RankedTensorType>(consumer->getResult(0).getType());
  int64_t m = source.getDimSize(0), k = source.getDimSize(1);
  if (!rhs || !out || rhs.getRank() != 2 || out.getRank() != 2 ||
      !rhs.hasStaticShape() || !out.hasStaticShape() ||
      rhs.getDimSize(0) != k || out.getDimSize(0) != m ||
      out.getDimSize(1) != rhs.getDimSize(1) ||
      rhs.getElementType() != source.getElementType() ||
      (paired && !out.getElementType().isF32()))
    return root.emitError("SM120 tensor original matmul shape/storage differs");
  int64_t n = rhs.getDimSize(1);
  int64_t capacities[3] = {m,n,k};
  bool varying[3] = {false,false,false};
  auto request = module->getAttr("tessera.native.sm120_tensor_bounds");
  if (paired && request)
    return root.emitError("SM120 paired tensor bounds need their own native capacity contract");
  if (request) {
    auto bounds = dyn_cast<DictionaryAttr>(request);
    if (!bounds || bounds.empty())
      return root.emitError("SM120 tensor bounds require nonempty M/N/K capacities");
    for (NamedAttribute attr : bounds) {
      auto key = attr.getName().getValue();
      int axis = key == "M" ? 0 : key == "N" ? 1 : key == "K" ? 2 : -1;
      auto bound = dyn_cast<IntegerAttr>(attr.getValue());
      if (axis < 0 || !bound || !bound.getType().isInteger(64) ||
          bound.getInt() < capacities[axis] || bound.getInt() <= 0 ||
          bound.getInt() >= (int64_t(1) << 31))
        return root.emitError("SM120 tensor bounds must contain positive i64 capacities covering the trace");
      capacities[axis] = bound.getInt(); varying[axis] = true;
    }
  }
  bool dynamic = varying[0] || varying[1] || varying[2];
  bool bias = consumer->hasAttr("bias"), residual = consumer->hasAttr("residual");
  if (roles.size() != unsigned(2 + bias + residual + paired))
    return root.emitError("SM120 tensor epilogue roles differ");
  llvm::DenseMap<Value,RankedTensorType> capacityTypes;
  capacityTypes[roles[0]] = RankedTensorType::get({capacities[0],capacities[2]},source.getElementType());
  capacityTypes[roles[1]] = RankedTensorType::get({capacities[2],capacities[1]},rhs.getElementType());
  for (Operation *producer : ArrayRef<Operation *>(ops).take_front(producerCount))
    capacityTypes[producer->getResult(0)] = capacityTypes[origins.lookup(producer->getResult(0))];
  capacityTypes[consumer->getResult(0)] = RankedTensorType::get({capacities[0],capacities[1]},out.getElementType());
  if (paired) {
    capacityTypes[roles[2]] = capacityTypes[roles[1]];
    capacityTypes[ops.back()->getResult(0)] = capacityTypes[consumer->getResult(0)];
  }
  for (unsigned index = 2; !paired && index < roles.size(); ++index) {
    auto type = dyn_cast<RankedTensorType>(roles[index].getType());
    bool isBias = bias && index == 2;
    if (!type || !type.getElementType().isF32() || type.getEncoding() ||
        (isBias ? type.getShape() != ArrayRef<int64_t>{n} :
                  type.getShape() != ArrayRef<int64_t>{m,n}))
      return root.emitError("SM120 tensor original epilogue shape/storage differs");
    capacityTypes[roles[index]] = RankedTensorType::get(
        isBias ? SmallVector<int64_t>{capacities[1]} :
                 SmallVector<int64_t>{capacities[0],capacities[1]},type.getElementType());
  }
  auto memberType = [&](Value value, bool consumer) -> Type {
    auto type = capacityTypes.lookup(value);
    if (!dynamic || !consumer) return type;
    SmallVector<int64_t> shape(type.getShape());
    bool producedRow = origins.lookup(value) == roles[0];
    if (producedRow) {
      if (varying[0]) shape[0] = ShapedType::kDynamic;
      if (varying[2]) shape[1] = ShapedType::kDynamic;
    } else if (origins.lookup(value) == roles[1]) {
      if (varying[2]) shape[0] = ShapedType::kDynamic;
      if (varying[1]) shape[1] = ShapedType::kDynamic;
    } else if (bias && value == roles[2]) {
      if (varying[1]) shape[0] = ShapedType::kDynamic;
    } else {
      if (varying[0]) shape[0] = ShapedType::kDynamic;
      if (varying[1]) shape[1] = ShapedType::kDynamic;
    }
    return RankedTensorType::get(shape,type.getElementType());
  };
  llvm::DenseMap<Value,int64_t> ids;
  SmallVector<Value> values(root.getArguments());
  SmallVector<int64_t> writes(values.size(),-1), reads(values.size(),-1);
  for (auto [index,value] : llvm::enumerate(values)) ids[value] = index;
  for (auto [index,op] : llvm::enumerate(ops)) {
    for (Value input : op->getOperands()) {
      auto found = ids.find(input);
      if (found == ids.end()) return op->emitError("SM120 tensor captured a value outside its SSA prefix");
      reads[found->second] = index;
    }
    Value output = op->getResult(0);
    ids[output] = values.size(); values.push_back(output);
    writes.push_back(index); reads.push_back(index);
  }
  int64_t outputID = ids.lookup(ret.getOperand(0));
  llvm::SmallDenseSet<int64_t> outputIDs;
  for (Value output : ret.getOperands()) {
    outputIDs.insert(ids.lookup(output));
    reads[ids.lookup(output)] = ops.size();
  }
  llvm::json::Array buffers,steps,memberGraphs;
  for (auto [index,value] : llvm::enumerate(values)) {
    auto original = dyn_cast<RankedTensorType>(value.getType());
    auto type = capacityTypes.lookup(value);
    if (!original || original.getEncoding())
      return root.emitError("SM120 tensor original storage encoding needs a native contract");
    if (!type || !type.hasStaticShape() || type.getEncoding() ||
        !isa<FloatType>(type.getElementType()))
      return root.emitError("SM120 tensor program requires static unencoded floating storage");
    auto bits = type.getElementType().getIntOrFloatBitWidth();
    uint64_t bytes = bits / 8;
    llvm::json::Array shape;
    for (int64_t dim : type.getShape()) {
      if (dim <= 0 || !bytes || uint64_t(dim) > uint64_t(INT64_MAX)/bytes)
        return root.emitError("SM120 tensor buffer bytes overflow");
      bytes *= uint64_t(dim); shape.push_back(dim);
    }
    std::string storage; llvm::raw_string_ostream stream(storage);
    type.getElementType().print(stream); stream.flush();
    buffers.push_back(llvm::json::Object{
      {"id",int64_t(index)},{"bytes",int64_t(bytes)},{"shape",std::move(shape)},
      {"storage",storage},{"ownership",index < root.getNumArguments() ? "readonly_input" :
                                   outputIDs.contains(int64_t(index)) ? "returned_output" : "private_scratch"},
      {"first_write",writes[index]},{"last_read",reads[index]}});
  }
  for (unsigned index = 0; index < ops.size(); ++index)
    if (SymbolTable::lookupSymbolIn(module,(root.getSymName()+"__tensor_member_"+llvm::Twine(index)).str()))
      return root.emitError("SM120 tensor member symbol already exists");
  std::string sourceIR; llvm::raw_string_ostream sourceStream(sourceIR);
  module.print(sourceStream); sourceStream.flush();
  std::string originalIR = sourceIR;
  if (dynamic) {
    auto projected = cast<ModuleOp>(module->clone());
    auto fn = cast<func::FuncOp>(SymbolTable::lookupSymbolIn(projected,root.getSymName()));
    SmallVector<Type> inputs;
    for (auto [old,arg] : llvm::zip(root.getArguments(),fn.getArguments())) {
      Type type = memberType(old,true); arg.setType(type); inputs.push_back(type);
    }
    // Project each producer through its own operand-root capacity frame,
    // preserving both chains and the original topological SSA order.
    for (auto [original, cloned] :
         llvm::zip(ops, fn.getBody().front().without_terminator()))
      cloned.getResult(0).setType(memberType(original->getResult(0),true));
    Type result = memberType(consumer->getResult(0),true);
    auto &projectedConsumer = *std::prev(fn.getBody().front().end(), 2);
    projectedConsumer.setAttr("shape_bounds",ArrayAttr::get(module.getContext(),{
      IntegerAttr::get(IntegerType::get(module.getContext(),64),capacities[0]),
      IntegerAttr::get(IntegerType::get(module.getContext(),64),capacities[1]),
      IntegerAttr::get(IntegerType::get(module.getContext(),64),capacities[2])}));
    fn.setFunctionType(FunctionType::get(module.getContext(),inputs,TypeRange{result}));
    projected->removeAttr("tessera.native.sm120_tensor_bounds");
    if (failed(verify(projected))) { projected.erase(); return failure(); }
    sourceIR.clear();
    llvm::raw_string_ostream view(sourceIR); projected.print(view); view.flush();
    projected.erase();
  }
  OpBuilder b(module.getContext());
  SmallVector<Attribute> memberSymbols;
  for (auto [index,op] : llvm::enumerate(ops)) {
    b.setInsertionPointToEnd(module.getBody());
    auto symbol = (root.getSymName()+"__tensor_member_"+llvm::Twine(index)).str();
    SmallVector<Type> inputTypes;
    for (Value value : op->getOperands()) inputTypes.push_back(memberType(value,index >= producerCount));
    auto outputType = memberType(op->getResult(0),index >= producerCount);
    auto member = b.create<func::FuncOp>(op->getLoc(),symbol,
        b.getFunctionType(inputTypes,TypeRange{outputType}));
    member.setPrivate();
    member->setAttr("tessera.native.sm120_tensor_member",b.getI64IntegerAttr(index));
    memberSymbols.push_back(FlatSymbolRefAttr::get(member));
    auto block = member.addEntryBlock();
    OpBuilder body(block,block->begin());
    IRMapping mapping;
    llvm::json::Array inputs;
    for (auto [input,arg] : llvm::zip(op->getOperands(),block->getArguments())) {
      mapping.map(input,arg); inputs.push_back(ids.lookup(input));
    }
    auto cloned = body.clone(*op,mapping);
    cloned->getResult(0).setType(outputType);
    if ((dag || paired) && index >= producerCount)
      cloned->setAttr("rhs_storage_order",b.getStringAttr("row_major"));
    if (dynamic && index >= producerCount)
      cloned->setAttr("shape_bounds",b.getI64ArrayAttr({capacities[0],capacities[1],capacities[2]}));
    body.create<func::ReturnOp>(op->getLoc(),cloned->getResults());
    auto projected = ModuleOp::create(module.getLoc());
    projected->setAttrs(module->getAttrs());
    projected->removeAttr("tessera.native.sm120_tensor_bounds");
    projected.getBody()->push_back(cloneNativeSM120TensorMember(member,index));
    std::string text; llvm::raw_string_ostream stream(text);
    projected.print(stream); stream.flush();
    memberGraphs.push_back(text+"\n");
    projected.erase();
    steps.push_back(llvm::json::Object{
      {"step",int64_t(index)},{"operation",op->getName().getStringRef().str()},
      {"member",symbol},{"inputs",std::move(inputs)},
      {"outputs",llvm::json::Array{ids.lookup(op->getResult(0))}}});
  }
  llvm::json::Object plan{{"schema",paired ? "tessera.native.sm120_tensor_program.v7" : dag ? (dynamic ?
          "tessera.native.sm120_tensor_program.v6" : "tessera.native.sm120_tensor_program.v5") :
          dynamic ? (producerCount > 1 ?
          "tessera.native.sm120_tensor_program.v4" : "tessera.native.sm120_tensor_program.v2") :
          producerCount > 1 ? "tessera.native.sm120_tensor_program.v3" : "tessera.native.sm120_tensor_program.v1"},
      {"source_graph_ir",paired ? originalSource.str() : sourceIR+"\n"},{"root",root.getSymName().str()},
      {"role_indices",std::move(roleJSON)},{"buffers",std::move(buffers)},
      {"steps",std::move(steps)},{"output",outputID},{"member_graphs",std::move(memberGraphs)}};
  if (paired) {
    plan["kind"] = "paired_jvp";
    plan["paired_graph_ir"] = sourceIR+"\n";
    plan["outputs"] = llvm::json::Array{outputID,ids.lookup(ret.getOperand(1))};
  }
  if (dynamic) {
    plan["original_graph_ir"] = originalIR+"\n";
    plan["active_shape"] = llvm::json::Array{m,n,k};
    plan["shape_bounds"] = llvm::json::Array{capacities[0],capacities[1],capacities[2]};
    llvm::json::Array axes;
    for (unsigned axis = 0; axis < 3; ++axis)
      if (varying[axis]) axes.push_back(axis == 0 ? "M" : axis == 1 ? "N" : "K");
    plan["dynamic_axes"] = std::move(axes);
  }
  std::string json; llvm::raw_string_ostream stream(json);
  stream << llvm::json::Value(std::move(plan)); stream.flush();
  module->setAttr("tessera.native.sm120_tensor_program_json",b.getStringAttr(llvm::encodeBase64(json)));
  module->setAttr("tessera.native.sm120_tensor_program",b.getDictionaryAttr({
    b.getNamedAttr("members",b.getArrayAttr(memberSymbols))}));
  return success();
}

static mlir::LogicalResult projectNativeSM120TensorMember(mlir::ModuleOp module,int64_t index) {
  using namespace mlir;
  auto plan = module->getAttrOfType<DictionaryAttr>("tessera.native.sm120_tensor_program");
  auto members = plan ? dyn_cast_or_null<ArrayAttr>(plan.get("members")) : ArrayAttr{};
  if (!members || index < 0 || index >= int64_t(members.size()))
    return module.emitError("SM120 tensor member index is outside its native program");
  auto symbol = dyn_cast<FlatSymbolRefAttr>(members[index]);
  auto member = symbol ? dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupSymbolIn(module,symbol)) : func::FuncOp{};
  auto lineage = member ? member->getAttrOfType<IntegerAttr>("tessera.native.sm120_tensor_member") : IntegerAttr{};
  if (!member || !lineage || lineage.getInt() != index)
    return module.emitError("SM120 tensor member lineage differs");
  auto child = cloneNativeSM120TensorMember(member,index);
  module.getBody()->clear(); module.getBody()->push_back(child);
  module->removeAttr("tessera.native.sm120_tensor_program");
  module->removeAttr("tessera.native.sm120_tensor_program_json");
  module->removeAttr("tessera.native.sm120_tensor_bounds");
  return success();
}
} // namespace tessera
