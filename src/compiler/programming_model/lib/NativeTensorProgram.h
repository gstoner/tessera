// Typed whole-program ownership for actual static SM120 tensor producers.
// Numerical kernels remain the existing independently verified leaf lowerings.
#pragma once
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace tensor_program {
struct Sm120MatmulKernel {
  bool macroProducer, typedProducer, useMacroProducer, macroGeometry;
  mlir::SmallVector<int64_t> geometry;
};
// One selection supplies admission and emission; never infer a tile from
// workload alone because row-major RHS and typed tails retain 16x8 geometry.
inline Sm120MatmulKernel selectMatmulKernel(const MatmulSchedule &s) {
  const __int128 work = static_cast<__int128>(2) * s.m * s.n * s.k;
  bool macro = !s.scalarFma && s.bLayout == "col_major" &&
      s.m >= 32 && s.n >= 32 && s.k >= 16 &&
      (s.storage == "f16" || s.storage == "bf16") && s.accum == "f32" &&
      work >= 67108864;
  bool fused = s.bias || s.residual || s.activation != "none";
  bool typed = !s.scalarFma && s.m > 0 && s.n > 0 && s.k > 0 &&
      s.tileM == 16 && s.tileN == 8 && s.tileK == 16 &&
      (s.storage == "f16" || s.storage == "bf16") && s.accum == "f32" &&
      (s.output == "f32" || s.output == "f16");
  bool useMacro = macro && s.k % 8 == 0 && !fused && s.output == "f32";
  bool macroGeometry = macro && (useMacro || !typed);
  int64_t rows = macroGeometry ? 32 : 16, cols = macroGeometry ? 32 : 8;
  return {macro, typed, useMacro, macroGeometry,
      {(s.n + cols - 1) / cols, (s.m + rows - 1) / rows, 1,
       macroGeometry ? 128 : 32, 1, 1}};
}
inline mlir::SmallVector<int64_t> semanticGeometry(const SemanticKernelSchedule &s) {
  int64_t blocks = s.rows;
  if (s.family == "activation")
    blocks = s.reductionSchedule == "flat_128"
        ? (s.rows * s.columns + 127) / 128 : (s.rows + 127) / 128;
  else if (s.reductionSchedule != "cooperative_128")
    blocks = (s.rows + 127) / 128;
  return {blocks, 1, 1, 128, 1, 1};
}
inline mlir::ArrayAttr bufferTypes(mlir::Operation *operation) {
  using namespace mlir;
  SmallVector<Attribute> result;
  for (Type type : operation->getOperandTypes()) {
    auto tensor = cast<RankedTensorType>(type);
    result.push_back(TypeAttr::get(MemRefType::get(tensor.getShape(),tensor.getElementType())));
  }
  auto tensor = cast<RankedTensorType>(operation->getResult(0).getType());
  result.push_back(TypeAttr::get(MemRefType::get(tensor.getShape(),tensor.getElementType())));
  return ArrayAttr::get(operation->getContext(),result);
}
inline mlir::LogicalResult outline(mlir::ModuleOp module, bool &selected) {
  using namespace mlir;
  selected = false;
  if (module->hasAttr("tessera.native.tensor_program") ||
      module->hasAttr("tessera.native.tensor_program_member"))
    return success();
  auto target = module->getAttrOfType<StringAttr>("tessera.target");
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  if (!target || target.getValue() != "nvidia_sm120" || !arch ||
      arch.getValue() != "sm_120") return success();
  SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
  if (functions.size() != 1 || !functions[0].getBody().hasOneBlock())
    return success();
  auto root = functions[0];
  SmallVector<Operation *> operations;
  for (auto &op : root.getBody().front().without_terminator())
    operations.push_back(&op);
  if (operations.size() < 2) return success();
  auto product = operations.back();
  if (product->getName().getStringRef() != "tessera.matmul" ||
      !llvm::any_of(product->getOperands(), [](Value input) {
        return input.getDefiningOp() != nullptr;
      })) return success();
  auto reject = [&]() {
    return root.emitError("Graph->Schedule native tensor program requires a static compact typed "
                          "producer DAG ending in a plain matmul");
  };
  for (Operation *owner : {module.getOperation(), root.getOperation()})
    for (NamedAttribute attribute : owner->getAttrs()) {
      StringRef name = attribute.getName().getValue();
      if (name == "tessera.autodiff" || name.starts_with("tessera.autodiff."))
        return root.emitError("pending differentiation must be transformed before primal tensor program projection");
    }
  auto ret = dyn_cast<func::ReturnOp>(root.getBody().front().back());
  if (root.getNumArguments() < 2 || root.getNumResults() != 1 || !ret ||
      ret.getNumOperands() != 1 || product->getNumOperands() != 2 ||
      product->getNumResults() != 1 || ret.getOperand(0) != product->getResult(0))
    return reject();
  auto bufferType = [](Type type) -> MemRefType {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || tensor.getRank() != 2 || !tensor.hasStaticShape() ||
        tensor.getEncoding() ||
        (!tensor.getElementType().isF16() && !tensor.getElementType().isBF16() &&
         !tensor.getElementType().isF32())) return {};
    int64_t bytes = tensor.getElementType().getIntOrFloatBitWidth() / 8;
    for (int64_t extent : tensor.getShape()) {
      if (extent <= 0 || extent > INT32_MAX || bytes > INT64_MAX / extent)
        return {};
      bytes *= extent;
    }
    return MemRefType::get(tensor.getShape(), tensor.getElementType());
  };
  for (auto [index, arg] : llvm::enumerate(root.getArguments())) {
    auto attrs = root.getArgAttrDict(index);
    auto type = bufferType(arg.getType());
    if (!type || (attrs && !attrs.empty()))
      return reject();
  }
  llvm::DenseSet<Value> available(root.getArguments().begin(), root.getArguments().end());
  llvm::DenseSet<Operation *> live;
  SmallVector<Value> pending(product->getOperands());
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (auto *op = value.getDefiningOp(); op && live.insert(op).second)
      pending.append(op->getOperands().begin(), op->getOperands().end());
  }
  for (Operation *op : operations) {
    auto name = op->getName().getStringRef();
    bool terminal = op == product;
    bool matrix = name == "tessera.matmul";
    if (!matrix &&
        (name != "tessera.relu" && name != "tessera.rmsnorm" &&
         name != "tessera.layer_norm" && name != "tessera.softmax"))
      return reject();
    if (matrix) {
      auto order = op->getAttrOfType<StringAttr>("rhs_storage_order");
      auto activation = op->getAttrOfType<StringAttr>("activation");
      if (op->getNumOperands() != 2 ||
          (order && order.getValue() != "row_major") ||
          (activation && activation.getValue() != "none") ||
          op->hasAttr("bias") || op->hasAttr("residual"))
        return reject();
      for (StringRef key : {"transposeA", "transposeB"})
        if (auto flag = op->getAttrOfType<BoolAttr>(key); flag && flag.getValue())
          return reject();
    }
    if (op->getNumResults() != 1 || !bufferType(op->getResult(0).getType()) ||
        op->getNumRegions() || !isMemoryEffectFree(op) ||
        (!terminal && !live.contains(op)) ||
        (!matrix && (op->getNumOperands() != 1 ||
                     op->getOperand(0).getType() != op->getResult(0).getType())))
      return reject();
    for (Value input : op->getOperands())
      if (!available.contains(input)) return reject();
    available.insert(op->getResult(0));
  }
  SmallVector<Type> storage;
  for (Value arg : root.getArguments()) storage.push_back(bufferType(arg.getType()));
  for (Operation *op : operations)
    storage.push_back(bufferType(op->getResult(0).getType()));
  if (failed(tessera::native_tensor_program::verifyFrameCapacity(
          root, storage, operations.size())))
    return failure();
  // Work on detached members first. No mutation of the original source occurs
  // until its complete dependency closure and each native leaf are admitted.
  SmallVector<OwningOpRef<ModuleOp>> members;
  OpBuilder builder(module.getContext());
  for (auto [index, op] : llvm::enumerate(operations)) {
    auto child = ModuleOp::create(op->getLoc());
    OwningOpRef<ModuleOp> owner(child);
    child->setAttrs(module->getAttrs());
    child->setAttr(SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(("tensor_stage_" + Twine(index)).str()));
    child->setAttr("tessera.native.tensor_program_member", builder.getUnitAttr());
    SmallVector<Type> inputs(op->getOperandTypes());
    auto function = func::FuncOp::create(op->getLoc(), "source",
        builder.getFunctionType(inputs, op->getResultTypes()));
    child.push_back(function);
    Block *block = function.addEntryBlock();
    OpBuilder leaf(block, block->begin());
    IRMapping mapping;
    for (auto [input, arg] : llvm::zip(op->getOperands(), block->getArguments()))
      mapping.map(input, arg);
    Operation *cloned = leaf.clone(*op, mapping);
    bool matrix = op->getName().getStringRef() == "tessera.matmul";
    if (matrix)
      cloned->setAttr("rhs_storage_order", builder.getStringAttr("row_major"));
    func::ReturnOp::create(leaf, op->getLoc(), cloned->getResults());
    SmallVector<int64_t> geometry;
    if (matrix) {
      auto schedule = getMatmulSchedule(cloned);
      if (failed(schedule)) return reject();
      geometry = selectMatmulKernel(*schedule).geometry;
    } else {
      auto schedule = getSemanticKernelSchedule(cloned);
      if (failed(schedule)) return reject();
      geometry = semanticGeometry(*schedule);
    }
    if (!tessera::native_tensor_program::isLaunchGeometry(geometry))
      return root.emitError("Graph->Schedule launch geometry exceeds the native frame CUDA limits");
    members.push_back(std::move(owner));
  }
  std::string source;
  llvm::raw_string_ostream stream(source);
  module.print(stream); stream.flush();
  module->setAttr("tessera.native.tensor_program", builder.getI64IntegerAttr(1));
  // Module printers add a final newline only at the outermost scope.
  // Canonicalize that transport whitespace identically for source replay in a
  // nested compiler pipeline; preserve every interior semantic attribute.
  module->setAttr("tessera.native.tensor_program_source",
      builder.getStringAttr(StringRef(source).trim()));
  module->setAttr("tessera.native.tensor_program_source_sha256",
      builder.getStringAttr(llvm::toHex(llvm::SHA256::hash(
          llvm::arrayRefFromStringRef(StringRef(source).trim())), true)));

  SmallVector<Type> inputs;
  for (Value arg : root.getArguments()) inputs.push_back(bufferType(arg.getType()));
  auto host = func::FuncOp::create(root.getLoc(), root.getName(),
      builder.getFunctionType(inputs, {bufferType(root.getResultTypes()[0])}));
  host->setAttrs(root->getAttrs());
  host.setType(builder.getFunctionType(inputs, {bufferType(root.getResultTypes()[0])}));
  host->setAttr("tessera.native.tensor_program", builder.getI64IntegerAttr(1));
  module.push_back(host);
  Block *body = host.addEntryBlock();
  OpBuilder hb(body, body->begin());
  IRMapping buffers;
  for (auto [old, arg] : llvm::zip(root.getArguments(), body->getArguments()))
    buffers.map(old, arg);
  SmallVector<Value> privateBuffers;
  for (auto [index, op] : llvm::enumerate(operations)) {
    Value output = memref::AllocOp::create(hb, op->getLoc(),
        bufferType(op->getResult(0).getType()));
    if (op != product) privateBuffers.push_back(output);
    SmallVector<Value> arguments;
    for (Value input : op->getOperands()) arguments.push_back(buffers.lookup(input));
    arguments.push_back(output);
    auto child = *members[index];
    OperationState invoke(op->getLoc(), "tile.program_invoke");
    invoke.addOperands(arguments);
    invoke.addAttribute("callee", SymbolRefAttr::get(module.getContext(),
        child.getSymName().value(), {FlatSymbolRefAttr::get(module.getContext(), "source")}));
    invoke.addAttribute("input_count", hb.getI64IntegerAttr(op->getNumOperands()));
    // Original semantic policies remain present at the whole-program boundary.
    invoke.addAttribute("source_attributes", hb.getDictionaryAttr(op->getAttrs()));
    for (StringRef name : {"numeric_policy", "layout", "tessera.layout"})
      if (auto attr = op->getAttr(name)) invoke.addAttribute(name, attr);
    hb.create(invoke);
    buffers.map(op->getResult(0), output);
    module.push_back(members[index].release());
  }
  for (Value buffer : privateBuffers)
    memref::DeallocOp::create(hb, root.getLoc(), buffer);
  func::ReturnOp::create(hb, root.getLoc(), buffers.lookup(product->getResult(0)));
  root.erase();
  selected = true;
  return success();
}
// Read-only transport projected from actual typed memory SSA.
inline mlir::LogicalResult exportFrame(mlir::ModuleOp module) {
  using namespace mlir;
  auto schema = module->getAttrOfType<IntegerAttr>("tessera.native.tensor_program");
  if (!schema || !schema.getType().isInteger(64) || schema.getInt() != 1)
    return module.emitError("native tensor frame requires module schema 1");
  SmallVector<func::FuncOp> hosts(module.getOps<func::FuncOp>());
  if (hosts.size() != 1 || !hosts.front().getBody().hasOneBlock() ||
      hosts.front().getBody().front().empty())
    return module.emitError("native tensor frame needs one defined single-block host program");
  auto host = hosts.front();
  if (!llvm::any_of(host.getBody().front(), [](Operation &op) {
        return native_tensor_program::isInvocation(&op);
      }))
    return host.emitError("native tensor frame needs at least one verified native invocation");
  auto ret = dyn_cast<func::ReturnOp>(host.getBody().front().back());
  if (!ret || ret.getNumOperands() != 1)
    return host.emitError("native tensor frame needs one returned allocation");
  SmallVector<Type> storage(host.getArgumentTypes());
  size_t stageCount = 0;
  for (Operation &operation : host.getBody().front()) {
    if (auto allocation = dyn_cast<memref::AllocOp>(operation))
      storage.push_back(allocation.getType());
    if (native_tensor_program::isInvocation(&operation)) ++stageCount;
  }
  if (failed(tessera::native_tensor_program::verifyFrameCapacity(
          host, storage, stageCount)))
    return failure();
  llvm::DenseMap<Value,int64_t> ids;
  llvm::json::Array buffers, roots, stages;
  auto addBuffer = [&](Value value, StringRef role) {
    int64_t id = ids.size();
    ids[value] = id;
    auto type = cast<MemRefType>(value.getType());
    llvm::json::Array shape;
    for (int64_t dim : type.getShape()) shape.push_back(dim);
    std::string storage;
    llvm::raw_string_ostream stream(storage);
    type.getElementType().print(stream); stream.flush();
    buffers.push_back(llvm::json::Object{
      {"id",id},{"role",role.str()},{"shape",std::move(shape)},{"storage",storage},
      {"bytes",type.getNumElements() * (type.getElementType().getIntOrFloatBitWidth()/8)}});
    return id;
  };
  for (Value arg : host.getArguments()) roots.push_back(addBuffer(arg,"input"));
  for (Operation &operation : host.getBody().front()) {
    if (auto allocation = dyn_cast<memref::AllocOp>(operation))
      addBuffer(allocation, allocation.getResult() == ret.getOperand(0) ? "output" : "private");
    if (auto *invoke = &operation; native_tensor_program::isInvocation(invoke)) {
      llvm::json::Array inputs, scalars, geometry;
      for (Value buffer : invoke->getOperands().drop_back()) inputs.push_back(ids.lookup(buffer));
      for (int64_t value : invoke->getAttrOfType<DenseI64ArrayAttr>("scalars").asArrayRef())
        scalars.push_back(value);
      for (int64_t value : invoke->getAttrOfType<DenseI64ArrayAttr>("geometry").asArrayRef())
        geometry.push_back(value);
      auto symbol = invoke->getAttrOfType<SymbolRefAttr>("callee");
      stages.push_back(llvm::json::Object{
        {"member",symbol.getRootReference().getValue().str()},
        {"entry",symbol.getNestedReferences().back().getValue().str()},
        {"inputs",std::move(inputs)},{"output",ids.lookup(invoke->getOperands().back())},
        {"scalars",std::move(scalars)},{"geometry",std::move(geometry)}});
    }
  }
  llvm::json::Object frame{
    {"schema","tessera.native.tensor_program_frame.v1"},
    {"target","nvidia_sm120"},{"architecture","sm_120"},
    {"function",host.getName().str()},{"buffers",std::move(buffers)},
    {"roots",std::move(roots)},{"stages",std::move(stages)},
    {"output",ids.lookup(ret.getOperand(0))}};
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << llvm::json::Value(std::move(frame)); stream.flush();
  module->setAttr("tessera.native.tensor_program_frame_json",
      StringAttr::get(module.getContext(),llvm::encodeBase64(text)));
  return success();
}
} // namespace tensor_program
