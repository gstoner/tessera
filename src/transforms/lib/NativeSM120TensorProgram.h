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

static mlir::LogicalResult emitNativeSM120TensorProgram(mlir::ModuleOp module) {
  using namespace mlir;
  if (failed(verify(module))) return failure();
  if (module->hasAttr("tessera.native.sm120_tensor_program"))
    return module.emitError("SM120 tensor program export already exists");
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  SmallVector<func::FuncOp> roots;
  for (auto fn : module.getOps<func::FuncOp>())
    if (!fn.isDeclaration()) roots.push_back(fn);
  if (!arch || arch.getValue() != "sm_120" || roots.size() != 1 ||
      !roots[0].getBody().hasOneBlock())
    return module.emitError("SM120 tensor program requires one explicit sm_120 Graph root");
  auto root = roots[0];
  if (root.getNumArguments() < 2 || root.getNumArguments() > 4 ||
      root.getNumResults() != 1 || (root.getBody().front().getOperations().size() < 3 ||
       root.getBody().front().getOperations().size() > 65))
    return root.emitError("SM120 tensor program requires its producer/matmul Graph");
  SmallVector<Operation *> ops;
  for (auto &op : root.getBody().front().without_terminator()) ops.push_back(&op);
  auto consumer = ops.back();
  const size_t producerCount = ops.size() - 1;
  auto ret = dyn_cast<func::ReturnOp>(root.getBody().front().back());
  if (consumer->getName().getStringRef() != "tessera.matmul" ||
      consumer->getNumResults() != 1 ||
      consumer->getNumOperands() != root.getNumArguments() ||
      !ret || ret.getNumOperands() != 1 || ret.getOperand(0) != consumer->getResult(0))
    return root.emitError("SM120 tensor program lost its original producer/consumer SSA edge");
  Value edge;
  for (auto [index, producer] : llvm::enumerate(ArrayRef<Operation *>(ops).drop_back())) {
    auto name = producer->getName().getStringRef();
    if ((name != "tessera.rmsnorm" && name != "tessera.layer_norm" && name != "tessera.softmax") ||
        producer->getNumOperands() != 1 || producer->getNumResults() != 1 ||
        producer->getOperand(0).getType() != producer->getResult(0).getType() ||
        (index && producer->getOperand(0) != edge))
      return root.emitError("SM120 tensor producer chain must preserve storage and SSA lineage");
    edge = producer->getResult(0);
  }
  if (consumer->getOperand(0) != edge)
    return root.emitError("SM120 tensor program lost its original producer/consumer SSA edge");
  auto source = dyn_cast<RankedTensorType>(ops[0]->getOperand(0).getType());
  if (!source || source.getRank() != 2 || !source.hasStaticShape() ||
      (!source.getElementType().isF16() && !source.getElementType().isBF16()) ||
      ops[0]->getResult(0).getType() != source)
    return root.emitError("SM120 tensor producer requires static rank-two matching f16/bf16 storage");
  SmallVector<Value> roles{ops[0]->getOperand(0)};
  for (Value value : consumer->getOperands().drop_front()) roles.push_back(value);
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
      rhs.getElementType() != source.getElementType())
    return root.emitError("SM120 tensor original matmul shape/storage differs");
  int64_t n = rhs.getDimSize(1);
  int64_t capacities[3] = {m,n,k};
  bool varying[3] = {false,false,false};
  auto request = module->getAttr("tessera.native.sm120_tensor_bounds");
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
  if (dynamic && producerCount != 1)
    return root.emitError("SM120 dynamic producer chains require the extended runtime capacity contract");
  bool bias = consumer->hasAttr("bias"), residual = consumer->hasAttr("residual");
  if (roles.size() != unsigned(2 + bias + residual))
    return root.emitError("SM120 tensor epilogue roles differ");
  llvm::DenseMap<Value,RankedTensorType> capacityTypes;
  capacityTypes[roles[0]] = RankedTensorType::get({capacities[0],capacities[2]},source.getElementType());
  capacityTypes[roles[1]] = RankedTensorType::get({capacities[2],capacities[1]},rhs.getElementType());
  for (Operation *producer : ArrayRef<Operation *>(ops).drop_back())
    capacityTypes[producer->getResult(0)] = capacityTypes[roles[0]];
  capacityTypes[consumer->getResult(0)] = RankedTensorType::get({capacities[0],capacities[1]},out.getElementType());
  for (unsigned index = 2; index < roles.size(); ++index) {
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
    bool producedRow = value == roles[0];
    for (Operation *producer : ArrayRef<Operation *>(ops).drop_back())
      producedRow |= value == producer->getResult(0);
    if (producedRow) {
      if (varying[0]) shape[0] = ShapedType::kDynamic;
      if (varying[2]) shape[1] = ShapedType::kDynamic;
    } else if (value == roles[1]) {
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
  reads[outputID] = ops.size();
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
                                   int64_t(index) == outputID ? "returned_output" : "private_scratch"},
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
    auto it = fn.getBody().front().begin();
    it->getResult(0).setType(memberType(ops[0]->getResult(0),true)); ++it;
    Type result = memberType(consumer->getResult(0),true);
    it->getResult(0).setType(result);
    it->setAttr("shape_bounds",ArrayAttr::get(module.getContext(),{
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
    for (Value value : op->getOperands()) inputTypes.push_back(memberType(value,index == producerCount));
    auto outputType = memberType(op->getResult(0),index == producerCount);
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
    if (dynamic && index == producerCount)
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
  llvm::json::Object plan{{"schema",dynamic ? "tessera.native.sm120_tensor_program.v2" :
          producerCount > 1 ? "tessera.native.sm120_tensor_program.v3" : "tessera.native.sm120_tensor_program.v1"},
      {"source_graph_ir",sourceIR+"\n"},{"root",root.getSymName().str()},
      {"role_indices",std::move(roleJSON)},{"buffers",std::move(buffers)},
      {"steps",std::move(steps)},{"output",outputID},{"member_graphs",std::move(memberGraphs)}};
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
