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
  if (s.family == "binary")
    blocks = (s.rows * s.columns + 127) / 128;
  else if (s.family == "activation")
    blocks = s.reductionSchedule == "flat_128"
        ? (s.rows * s.columns + 127) / 128 : (s.rows + 127) / 128;
  else if (s.reductionSchedule != "cooperative_128")
    blocks = (s.rows + 127) / 128;
  return {blocks, 1, 1, 128, 1, 1};
}
inline mlir::ArrayAttr activeBufferTypes(mlir::Operation *operation) {
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
// The pointer kernel ABI binds capacity storage; active tensor types are kept
// independently so source replay and invocation verification cannot confuse an
// M bound with the actual invocation M.
inline mlir::ArrayAttr bufferTypes(mlir::Operation *operation) {
  using namespace mlir;
  auto types = activeBufferTypes(operation);
  auto member = operation->getParentOfType<ModuleOp>();
  auto bound = member->getAttrOfType<IntegerAttr>("tessera.native.tensor_program_row_bound");
  if (!bound) return types;
  SmallVector<Attribute> result;
  for (Attribute attr : types) {
    auto type = cast<MemRefType>(cast<TypeAttr>(attr).getValue());
    SmallVector<int64_t> shape(type.getShape());
    if (ShapedType::isDynamic(shape[0])) shape[0] = bound.getInt();
    result.push_back(TypeAttr::get(MemRefType::get(shape,type.getElementType())));
  }
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
  OpBuilder builder(module.getContext());
  auto ret = dyn_cast<func::ReturnOp>(root.getBody().front().back());
  if (root.getNumArguments() < 2 || root.getNumResults() != 1 || !ret ||
      ret.getNumOperands() != 1 || product->getNumOperands() != 2 ||
      product->getNumResults() != 1 || ret.getOperand(0) != product->getResult(0))
    return reject();
  if (module->hasAttr("tessera.native.tensor_program_row_bound_request")) {
    auto request = module->getAttrOfType<IntegerAttr>("tessera.native.tensor_program_row_bound_request");
    auto add = operations.front();
    auto lhs = dyn_cast<RankedTensorType>(product->getOperand(0).getType());
    auto rhs = dyn_cast<RankedTensorType>(product->getOperand(1).getType());
    auto out = dyn_cast<RankedTensorType>(product->getResult(0).getType());
    if (!request || !request.getType().isInteger(64) || request.getInt() <= 0 ||
        request.getInt() > INT32_MAX || operations.size() != 2 || root.getNumArguments() != 3 ||
        add->getName().getStringRef() != "tessera.add" || add->getNumOperands() != 2 ||
        add->getNumResults() != 1 || product->getOperand(0) != add->getResult(0) ||
        !isa<BlockArgument>(add->getOperand(0)) || !isa<BlockArgument>(add->getOperand(1)) ||
        !isa<BlockArgument>(product->getOperand(1)) ||
        !lhs || !rhs || !out || lhs.getRank() != 2 || rhs.getRank() != 2 || out.getRank() != 2 ||
        !lhs.hasStaticShape() || !rhs.hasStaticShape() || !out.hasStaticShape() ||
        lhs.getEncoding() || rhs.getEncoding() || out.getEncoding() ||
        lhs.getDimSize(0) <= 0 || lhs.getDimSize(0) > request.getInt() ||
        add->getOperand(0).getType() != lhs || add->getOperand(1).getType() != lhs)
      return reject();
    if (auto existing = product->getAttr("shape_bounds")) {
      auto wanted = builder.getI64ArrayAttr({request.getInt(),rhs.getDimSize(1),lhs.getDimSize(1)});
      if (existing != wanted) return reject();
    }
    std::string source;
    llvm::raw_string_ostream stream(source); module.print(stream); stream.flush();
    OwningOpRef<ModuleOp> projected(cast<ModuleOp>(module->clone()));
    projected->getOperation()->removeAttr("tessera.native.tensor_program_row_bound_request");
    auto function = *projected->getOps<func::FuncOp>().begin();
    auto &body = function.getBody().front();
    auto producer = &body.front();
    auto product = producer->getNextNode();
    auto dynamicType = [&](RankedTensorType type) {
      return RankedTensorType::get({ShapedType::kDynamic,type.getDimSize(1)},type.getElementType());
    };
    auto dynamicInput = dynamicType(lhs), dynamicOutput = dynamicType(out);
    for (Value input : producer->getOperands()) input.setType(dynamicInput);
    producer->getResult(0).setType(dynamicInput);
    product->getResult(0).setType(dynamicOutput);
    SmallVector<Type> arguments(function.getArgumentTypes());
    for (auto [index,arg] : llvm::enumerate(body.getArguments())) arguments[index]=arg.getType();
    function.setType(builder.getFunctionType(arguments,{dynamicOutput}));
    product->setAttr("shape_bounds",builder.getI64ArrayAttr({request.getInt(),rhs.getDimSize(1),lhs.getDimSize(1)}));
    bool admitted=false;
    if (failed(outline(*projected,admitted)) || !admitted) return reject();
    projected->getOperation()->setAttr("tessera.native.tensor_program_source",builder.getStringAttr(StringRef(source).trim()));
    projected->getOperation()->setAttr("tessera.native.tensor_program_source_sha256",
        builder.getStringAttr(llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(StringRef(source).trim())),true)));
    module->setAttrs(projected->getOperation()->getAttrs());
    module.getBody()->clear();
    module.getBody()->getOperations().splice(module.getBody()->end(),projected->getBody()->getOperations());
    selected=true;
    return success();
  }
  int64_t rowBound = 0, rowRoot = -1;
  bool dynamic = llvm::any_of(root.getArgumentTypes(), [](Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    return tensor && !tensor.hasStaticShape();
  });
  if (dynamic) {
    // Initial bounded program: the original two-input stored add feeds a
    // typed row-major half matmul. No static clone replaces the dynamic leaf.
    auto add = operations.front();
    auto lhs = dyn_cast<RankedTensorType>(product->getOperand(0).getType());
    auto rhs = dyn_cast<RankedTensorType>(product->getOperand(1).getType());
    auto out = dyn_cast<RankedTensorType>(product->getResult(0).getType());
    auto bounds = product->getAttrOfType<ArrayAttr>("shape_bounds");
    if (operations.size() != 2 || root.getNumArguments() != 3 ||
        add->getName().getStringRef() != "tessera.add" ||
        add->getNumOperands() != 2 || add->getNumResults() != 1 ||
        product->getOperand(0) != add->getResult(0) ||
        !isa<BlockArgument>(add->getOperand(0)) || !isa<BlockArgument>(add->getOperand(1)) ||
        !isa<BlockArgument>(product->getOperand(1)) ||
        !lhs || !rhs || !out || lhs.getRank() != 2 || rhs.getRank() != 2 || out.getRank() != 2 ||
        !lhs.isDynamicDim(0) || lhs.isDynamicDim(1) || !rhs.hasStaticShape() ||
        !out.isDynamicDim(0) || out.isDynamicDim(1) ||
        (!lhs.getElementType().isF16() && !lhs.getElementType().isBF16()) ||
        rhs.getElementType() != lhs.getElementType() || !out.getElementType().isF32() ||
        lhs.getDimSize(1) <= 0 || rhs.getDimSize(0) != lhs.getDimSize(1) ||
        out.getDimSize(1) != rhs.getDimSize(1) || !bounds || bounds.size() != 3)
      return reject();
    SmallVector<int64_t> sizes;
    for (Attribute value : bounds) {
      auto size = dyn_cast<IntegerAttr>(value);
      if (!size || !size.getType().isInteger(64) || size.getInt() <= 0 || size.getInt() > INT32_MAX)
        return reject();
      sizes.push_back(size.getInt());
    }
    if (sizes[1] != rhs.getDimSize(1) || sizes[2] != lhs.getDimSize(1)) return reject();
    rowBound = sizes[0];
    rowRoot = cast<BlockArgument>(add->getOperand(0)).getArgNumber();
  }
  auto bufferType = [&](Type type) -> MemRefType {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || tensor.getRank() != 2 ||
        (!tensor.hasStaticShape() && (!rowBound || !tensor.isDynamicDim(0) || tensor.isDynamicDim(1))) ||
        tensor.getEncoding() ||
        (!tensor.getElementType().isF16() && !tensor.getElementType().isBF16() &&
         !tensor.getElementType().isF32())) return {};
    SmallVector<int64_t> shape(tensor.getShape());
    if (ShapedType::isDynamic(shape[0])) shape[0] = rowBound;
    int64_t bytes = tensor.getElementType().getIntOrFloatBitWidth() / 8;
    for (int64_t extent : shape) {
      if (extent <= 0 || extent > INT32_MAX || bytes > INT64_MAX / extent)
        return {};
      bytes *= extent;
    }
    return MemRefType::get(shape, tensor.getElementType());
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
        (name != "tessera.add" && name != "tessera.relu" && name != "tessera.rmsnorm" &&
         name != "tessera.layer_norm" && name != "tessera.softmax"))
      return reject();
    if (matrix) {
      auto order = op->getAttrOfType<StringAttr>("rhs_storage_order");
      auto activation = op->getAttrOfType<StringAttr>("activation");
      if (op->getNumOperands() != 2 ||
          (op->hasAttr("rhs_storage_order") && !order) ||
          (op->hasAttr("activation") && !activation) ||
          (order && order.getValue() != "row_major") ||
          (activation && activation.getValue() != "none") ||
          op->hasAttr("bias") || op->hasAttr("residual"))
        return reject();
      for (StringRef key : {"layout", "tessera.layout"})
        if (op->hasAttr(key)) {
          auto layout = op->getAttrOfType<StringAttr>(key);
          if (!layout || layout.getValue() != "row_major") return reject();
        }
      for (StringRef key : {"transposeA", "transposeB"})
        if (op->hasAttr(key)) {
          auto flag = op->getAttrOfType<BoolAttr>(key);
          if (!flag || flag.getValue()) return reject();
        }
    }
    if (op->getNumResults() != 1 || !bufferType(op->getResult(0).getType()) ||
        op->getNumRegions() || !isMemoryEffectFree(op) ||
        (!terminal && !live.contains(op)) ||
        (!matrix && (op->getNumOperands() != (name == "tessera.add" ? 2u : 1u) ||
                     !llvm::all_of(op->getOperandTypes(), [&](Type input) {
                       return input == op->getResult(0).getType();
                     }))))
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
  for (auto [index, op] : llvm::enumerate(operations)) {
    auto child = ModuleOp::create(op->getLoc());
    OwningOpRef<ModuleOp> owner(child);
    child->setAttrs(module->getAttrs());
    child->setAttr(SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(("tensor_stage_" + Twine(index)).str()));
    child->setAttr("tessera.native.tensor_program_member", builder.getUnitAttr());
    if (rowBound)
      child->setAttr("tessera.native.tensor_program_row_bound", builder.getI64IntegerAttr(rowBound));
    SmallVector<Type> inputs(op->getOperandTypes());
    auto function = func::FuncOp::create(op->getLoc(), "source",
        builder.getFunctionType(inputs, op->getResultTypes()));
    if (rowBound && op->getName().getStringRef() == "tessera.add")
      function->setAttr("tessera.native.tensor_row_bound", builder.getI64IntegerAttr(rowBound));
    child.push_back(function);
    Block *block = function.addEntryBlock();
    OpBuilder leaf(block, block->begin());
    // Preserve operand positions even when two inputs are the same SSA
    // value. A Value->Value map would collapse both onto the last argument;
    // the checked invocation already records their intended runtime alias.
    Operation *cloned = leaf.clone(*op);
    cloned->setOperands(block->getArguments());
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
  const int64_t version = rowBound ? 2 : 1;
  module->setAttr("tessera.native.tensor_program", builder.getI64IntegerAttr(version));
  if (rowBound) {
    SmallVector<int64_t> active;
    for (Value value : root.getArguments())
      active.push_back(cast<RankedTensorType>(value.getType()).isDynamicDim(0));
    for (Operation *op : operations)
      active.push_back(cast<RankedTensorType>(op->getResult(0).getType()).isDynamicDim(0));
    module->setAttr("tessera.native.tensor_program_row_bound", builder.getI64IntegerAttr(rowBound));
    module->setAttr("tessera.native.tensor_program_row_root", builder.getI64IntegerAttr(rowRoot));
    module->setAttr("tessera.native.tensor_program_row_buffers", builder.getDenseI64ArrayAttr(active));
  }
  // Module printers add a final newline only at the outermost scope.
  // Canonicalize that transport whitespace identically for source replay in a
  // nested compiler pipeline; preserve every interior semantic attribute.
  module->setAttr("tessera.native.tensor_program_source",
      builder.getStringAttr(StringRef(source).trim()));
  module->setAttr("tessera.native.tensor_program_source_sha256",
      builder.getStringAttr(llvm::toHex(llvm::SHA256::hash(
          llvm::arrayRefFromStringRef(StringRef(source).trim())), true)));

  auto activeType = [&](Type raw) {
    auto tensor = cast<RankedTensorType>(raw);
    return MemRefType::get(tensor.getShape(),tensor.getElementType());
  };
  SmallVector<Type> inputs;
  for (Value arg : root.getArguments()) inputs.push_back(activeType(arg.getType()));
  auto host = func::FuncOp::create(root.getLoc(), root.getName(),
      builder.getFunctionType(inputs, {activeType(root.getResultTypes()[0])}));
  host->setAttrs(root->getAttrs());
  host.setType(builder.getFunctionType(inputs, {activeType(root.getResultTypes()[0])}));
  host->setAttr("tessera.native.tensor_program", builder.getI64IntegerAttr(version));
  module.push_back(host);
  Block *body = host.addEntryBlock();
  OpBuilder hb(body, body->begin());
  IRMapping buffers;
  for (auto [old, arg] : llvm::zip(root.getArguments(), body->getArguments()))
    buffers.map(old, arg);
  Value activeRows;
  if (rowBound)
    activeRows = memref::DimOp::create(hb,root.getLoc(),body->getArgument(rowRoot),0);
  SmallVector<Value> privateBuffers;
  for (auto [index, op] : llvm::enumerate(operations)) {
    Value allocation = memref::AllocOp::create(hb, op->getLoc(),
        bufferType(op->getResult(0).getType()));
    Value output = allocation;
    auto type = activeType(op->getResult(0).getType());
    if (rowBound && type.isDynamicDim(0)) {
      SmallVector<OpFoldResult> offsets{hb.getIndexAttr(0),hb.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes{activeRows,hb.getIndexAttr(type.getDimSize(1))};
      SmallVector<OpFoldResult> strides{hb.getIndexAttr(1),hb.getIndexAttr(1)};
      output = memref::SubViewOp::create(hb,op->getLoc(),type,allocation,offsets,sizes,strides);
    }
    if (op != product) privateBuffers.push_back(allocation);
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
  if (!schema || !schema.getType().isInteger(64) ||
      (schema.getInt() != 1 && schema.getInt() != 2))
    return module.emitError("native tensor frame requires module schema 1 or 2");
  bool bounded = schema.getInt() == 2;
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
  int64_t rowBound = bounded ? module->getAttrOfType<IntegerAttr>("tessera.native.tensor_program_row_bound").getInt() : 0;
  auto capacityType = [&](Type raw) -> MemRefType {
    return bounded ? tessera::native_tensor_program::capacityType(raw,rowBound) : cast<MemRefType>(raw);
  };
  SmallVector<Type> storage;
  for (Type type : host.getArgumentTypes()) storage.push_back(capacityType(type));
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
    int64_t id = buffers.size();
    ids[value] = id;
    auto type = capacityType(value.getType());
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
      addBuffer(allocation, allocation.getResult() ==
          tessera::native_tensor_program::capacityBase(ret.getOperand(0)) ? "output" : "private");
    if (auto view = dyn_cast<memref::SubViewOp>(operation))
      ids[view.getResult()] = ids.lookup(view.getSource());
    if (auto *invoke = &operation; native_tensor_program::isInvocation(invoke)) {
      llvm::json::Array inputs, scalars, geometry;
      for (Value buffer : invoke->getOperands().drop_back()) inputs.push_back(ids.lookup(buffer));
      for (int64_t value : invoke->getAttrOfType<DenseI64ArrayAttr>("scalars").asArrayRef())
        scalars.push_back(value);
      for (int64_t value : invoke->getAttrOfType<DenseI64ArrayAttr>("geometry").asArrayRef())
        geometry.push_back(value);
      auto symbol = invoke->getAttrOfType<SymbolRefAttr>("callee");
      llvm::json::Object stage{
        {"member",symbol.getRootReference().getValue().str()},
        {"entry",symbol.getNestedReferences().back().getValue().str()},
        {"inputs",std::move(inputs)},{"output",ids.lookup(invoke->getOperands().back())},
        {"scalars",std::move(scalars)},{"geometry",std::move(geometry)}};
      if (bounded) {
        for (StringRef key : {"scalar_row_multipliers", "grid_row_multipliers", "grid_row_divisors"}) {
          llvm::json::Array values;
          for (int64_t value : invoke->getAttrOfType<DenseI64ArrayAttr>(key).asArrayRef())
            values.push_back(value);
          stage[key.str()] = std::move(values);
        }
      }
      stages.push_back(std::move(stage));
    }
  }
  llvm::json::Object frame{
    {"schema", bounded ? "tessera.native.tensor_program_frame.v2" : "tessera.native.tensor_program_frame.v1"},
    {"target","nvidia_sm120"},{"architecture","sm_120"},
    {"function",host.getName().str()},{"buffers",std::move(buffers)},
    {"roots",std::move(roots)},{"stages",std::move(stages)},
    {"output",ids.lookup(ret.getOperand(0))}};
  if (bounded) {
    frame["row_bound"] = module->getAttrOfType<IntegerAttr>("tessera.native.tensor_program_row_bound").getInt();
    frame["row_root"] = module->getAttrOfType<IntegerAttr>("tessera.native.tensor_program_row_root").getInt();
    llvm::json::Array active;
    for (int64_t value : module->getAttrOfType<DenseI64ArrayAttr>("tessera.native.tensor_program_row_buffers").asArrayRef())
      active.push_back(value);
    frame["row_buffers"] = std::move(active);
  }
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << llvm::json::Value(std::move(frame)); stream.flush();
  module->setAttr("tessera.native.tensor_program_frame_json",
      StringAttr::get(module.getContext(),llvm::encodeBase64(text)));
  return success();
}
} // namespace tensor_program
