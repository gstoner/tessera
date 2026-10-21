// Typed whole-program ownership for actual static SM120 tensor producers.
// Numerical kernels remain the existing independently verified leaf lowerings.
#pragma once
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "NativeTensorShapePlan.h"

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
  auto function = operation->getParentOfType<func::FuncOp>();
  if (member->hasAttr("tessera.native.tensor_program_shape_profile") && function &&
      function->hasAttr("tessera.native.tensor_shape_capacity"))
    if (auto capacities = member->getAttrOfType<ArrayAttr>("tessera.native.tensor_program_capacity_types"))
      return capacities;
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
inline mlir::LogicalResult outline(mlir::ModuleOp module, bool &selected,
                                   int64_t projectedRowBound = 0,
    const tessera::native_tensor_program::TensorShapeProfile *shapeProfile = nullptr) {
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
  // The existing matmul epilogue lowering owns matrix-result consumers.
  // Do not replace a fused native image with separate program launches.
  bool hasMatmul = llvm::any_of(operations, [](Operation *op) {
    return op->getName().getStringRef() == "tessera.matmul";
  });
  if (hasMatmul && product->getName().getStringRef() != "tessera.matmul")
    return success();
  // Claim only the semantic families this native owner can lower. Other
  // source-program families retain their registered consumers.
  if (!hasMatmul && !llvm::all_of(operations, [](Operation *op) {
        auto name = op->getName().getStringRef();
        return name == "tessera.add" || name == "tessera.relu" ||
               name == "tessera.rmsnorm" || name == "tessera.layer_norm" ||
               name == "tessera.softmax";
      })) return success();
  if (!llvm::any_of(product->getOperands(), [](Value input) {
        return input.getDefiningOp() != nullptr;
      })) return success();
  auto reject = [&]() {
    return root.emitError("Graph->Schedule native tensor program requires a compact typed "
                          "producer DAG with verified static or bounded leading rows");
  };
  OpBuilder builder(module.getContext());
  if (module->hasAttr("tessera.native.tensor_program_shape_bounds_request")) {
    auto bounds = module->getAttrOfType<DictionaryAttr>("tessera.native.tensor_program_shape_bounds_request");
    if (!bounds) return reject();
    std::string source;
    llvm::raw_string_ostream stream(source); module.print(stream); stream.flush();
    OwningOpRef<ModuleOp> projected(cast<ModuleOp>(module->clone()));
    projected->getOperation()->removeAttr("tessera.native.tensor_program_shape_bounds_request");
    auto function = *projected->getOps<func::FuncOp>().begin();
    auto profile = tessera::native_tensor_program::TensorShapeProfile::analyze(function, bounds);
    if (failed(profile)) return reject();
    for (Operation &operation : function.getBody().front().without_terminator()) {
      if (operation.getName().getStringRef() != "tessera.matmul") continue;
      auto left = profile->capacities.find(operation.getOperand(0))->second;
      auto right = profile->capacities.find(operation.getOperand(1))->second;
      auto a = operation.getAttrOfType<BoolAttr>("transposeA");
      auto b = operation.getAttrOfType<BoolAttr>("transposeB");
      unsigned m = a && a.getValue() ? 1 : 0, n = b && b.getValue() ? 0 : 1;
      auto wanted = builder.getI64ArrayAttr({left[m], right[n], left[1-m]});
      if (Attribute prior = operation.getAttr("shape_bounds"); prior && prior != wanted)
        return reject();
      operation.setAttr("shape_bounds", wanted);
    }
    profile->projectTypes(function);
    bool admitted = false;
    if (failed(outline(*projected, admitted, 0, &*profile)) || !admitted) return reject();
    projected->getOperation()->setAttr("tessera.native.tensor_program_source", builder.getStringAttr(StringRef(source).trim()));
    projected->getOperation()->setAttr("tessera.native.tensor_program_source_sha256",
        builder.getStringAttr(llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(StringRef(source).trim())),true)));
    module->setAttrs(projected->getOperation()->getAttrs());
    module.getBodyRegion().takeBody(projected->getRegion());
    selected = true;
    return success();
  }
  for (Operation *owner : {module.getOperation(), root.getOperation()})
    for (NamedAttribute attribute : owner->getAttrs()) {
      StringRef name = attribute.getName().getValue();
      if (name == "tessera.autodiff" || name.starts_with("tessera.autodiff."))
        return root.emitError("pending differentiation must be transformed before primal tensor program projection");
    }
  auto ret = dyn_cast<func::ReturnOp>(root.getBody().front().back());
  if (root.getNumArguments() < 1 || root.getNumResults() != 1 || !ret ||
      ret.getNumOperands() != 1 ||
      product->getNumResults() != 1 || ret.getOperand(0) != product->getResult(0))
    return reject();
  // Classify the varying row component from actual SSA relationships. Only
  // same-shape semantic edges and matmul LHS/result rows are equal; equal
  // numeric extents (including trace-at-one and M == K) imply no relation.
  auto rowComponent = [&]() -> FailureOr<llvm::DenseSet<Value>> {
    tessera::native_tensor_program::TensorShapeRelations relations;
    for (Value argument : root.getArguments())
      if (failed(relations.addValue(argument))) return failure();
    for (Operation *op : operations)
      if (failed(relations.addOperation(op))) return failure();
    Value output = product->getResult(0);
    llvm::DenseSet<Value> varying;
    for (auto dimension : relations.getDimensions()) {
      if (!relations.equivalent(dimension.tensor, dimension.axis, output, 0)) continue;
      // The v2 row profile keeps all nonleading dimensions fixed. A general
      // shape profile will bind independent components through native shape IR.
      if (dimension.axis != 0) return failure();
      varying.insert(dimension.tensor);
    }
    for (Operation *op : operations)
      if (op->getName().getStringRef() == "tessera.matmul" &&
          varying.contains(op->getOperand(1)))
        return failure(); // Preserve the existing M-only fixed-K contract.
    return varying;
  };
  if (module->hasAttr("tessera.native.tensor_program_row_bound_request")) {
    auto request = module->getAttrOfType<IntegerAttr>("tessera.native.tensor_program_row_bound_request");
    auto component = rowComponent();
    auto out = dyn_cast<RankedTensorType>(product->getResult(0).getType());
    if (!request || !request.getType().isInteger(64) || request.getInt() <= 0 ||
        request.getInt() > INT32_MAX || failed(component) || !out ||
        out.getRank() != 2 || out.isDynamicDim(1) ||
        (!out.isDynamicDim(0) &&
         (out.getDimSize(0) <= 0 || out.getDimSize(0) > request.getInt())))
      return reject();
    for (Value value : *component) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || type.getRank() != 2 || type.isDynamicDim(1) ||
          type.getEncoding() || type.getDimSize(0) != out.getDimSize(0)) return reject();
    }
    std::string source;
    llvm::raw_string_ostream stream(source); module.print(stream); stream.flush();
    OwningOpRef<ModuleOp> projected(cast<ModuleOp>(module->clone()));
    projected->getOperation()->removeAttr("tessera.native.tensor_program_row_bound_request");
    auto function = *projected->getOps<func::FuncOp>().begin();
    auto &body = function.getBody().front();
    // The clone preserves argument/result positions; bind the checked source
    // component to cloned SSA values without matching tensor shapes by number.
    llvm::DenseMap<Value, Value> values;
    for (auto pair : llvm::zip(root.getArguments(), function.getArguments()))
      values[std::get<0>(pair)] = std::get<1>(pair);
    SmallVector<Operation *> clonedOperations;
    for (auto &op : body.without_terminator()) clonedOperations.push_back(&op);
    for (auto pair : llvm::zip(operations, clonedOperations))
      values[std::get<0>(pair)->getResult(0)] = std::get<1>(pair)->getResult(0);
    for (Value original : *component) {
      auto type = cast<RankedTensorType>(original.getType());
      values.lookup(original).setType(RankedTensorType::get(
          {ShapedType::kDynamic, type.getDimSize(1)}, type.getElementType()));
    }
    for (Operation *op : clonedOperations) {
      if (op->getName().getStringRef() != "tessera.matmul") continue;
      auto lhs = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      auto rhs = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
      if (!lhs || !rhs || lhs.getRank() != 2 || rhs.getRank() != 2) return reject();
      if (!lhs.isDynamicDim(0)) continue;
      auto wanted = builder.getI64ArrayAttr({request.getInt(),rhs.getDimSize(1),lhs.getDimSize(1)});
      if (Attribute existing = op->getAttr("shape_bounds"); existing && existing != wanted)
        return reject();
      op->setAttr("shape_bounds",wanted);
    }
    SmallVector<Type> arguments(function.getArgumentTypes());
    for (auto [index,arg] : llvm::enumerate(body.getArguments())) arguments[index]=arg.getType();
    function.setType(builder.getFunctionType(arguments,{clonedOperations.back()->getResult(0).getType()}));
    bool admitted=false;
    if (failed(outline(*projected,admitted,request.getInt())) || !admitted) return reject();
    projected->getOperation()->setAttr("tessera.native.tensor_program_source",builder.getStringAttr(StringRef(source).trim()));
    projected->getOperation()->setAttr("tessera.native.tensor_program_source_sha256",
        builder.getStringAttr(llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(StringRef(source).trim())),true)));
    module->setAttrs(projected->getOperation()->getAttrs());
    module.getBody()->clear();
    module.getBody()->getOperations().splice(module.getBody()->end(),projected->getBody()->getOperations());
    selected=true;
    return success();
  }
  int64_t rowBound = projectedRowBound, rowRoot = -1;
  bool dynamic = llvm::any_of(root.getArgumentTypes(), [](Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    return tensor && !tensor.hasStaticShape();
  });
  if (dynamic && !shapeProfile) {
    auto component = rowComponent();
    if (failed(component)) return reject();
    if (!rowBound) {
      auto bounds = product->getAttrOfType<ArrayAttr>("shape_bounds");
      if (!bounds || bounds.size() != 3) return reject();
      auto capacity = dyn_cast<IntegerAttr>(bounds[0]);
      if (!capacity || !capacity.getType().isInteger(64)) return reject();
      rowBound = capacity.getInt();
    }
    if (rowBound <= 0 || rowBound > INT32_MAX) return reject();
    auto checkType = [&](Value value) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      return type && type.getRank() == 2 && !type.getEncoding() &&
          type.isDynamicDim(0) == component->contains(value) &&
          !type.isDynamicDim(1) && type.getDimSize(1) > 0 &&
          (component->contains(value) || type.getDimSize(0) > 0);
    };
    for (auto argument : root.getArguments()) {
      if (!checkType(argument)) return reject();
      if (rowRoot < 0 && component->contains(argument)) rowRoot = argument.getArgNumber();
    }
    if (rowRoot < 0) return reject();
    for (Operation *op : operations) {
      if (!checkType(op->getResult(0))) return reject();
      if (op->getName().getStringRef() != "tessera.matmul") continue;
      auto lhs = cast<RankedTensorType>(op->getOperand(0).getType());
      auto rhs = cast<RankedTensorType>(op->getOperand(1).getType());
      if (!lhs.isDynamicDim(0)) continue;
      auto wanted = builder.getI64ArrayAttr({rowBound,rhs.getDimSize(1),lhs.getDimSize(1)});
      if (op->getAttr("shape_bounds") != wanted) return reject();
    }
  }
  auto bufferType = [&](Value value) -> MemRefType {
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!tensor || tensor.getRank() != 2 ||
        (!tensor.hasStaticShape() && !shapeProfile && (!rowBound || !tensor.isDynamicDim(0) || tensor.isDynamicDim(1))) ||
        tensor.getEncoding() ||
        (!tensor.getElementType().isF16() && !tensor.getElementType().isBF16() &&
         !tensor.getElementType().isF32())) return {};
    SmallVector<int64_t> shape(tensor.getShape());
    if (shapeProfile) {
      auto capacity = shapeProfile->capacities.find(value);
      if (capacity == shapeProfile->capacities.end()) return {};
      shape.assign(capacity->second.begin(), capacity->second.end());
    } else if (ShapedType::isDynamic(shape[0])) shape[0] = rowBound;
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
    auto type = bufferType(arg);
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
    if (op->getNumResults() != 1 || !bufferType(op->getResult(0)) ||
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
  for (Value arg : root.getArguments()) storage.push_back(bufferType(arg));
  for (Operation *op : operations)
    storage.push_back(bufferType(op->getResult(0)));
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
    if (rowBound && cast<RankedTensorType>(op->getResult(0).getType()).isDynamicDim(0))
      function->setAttr("tessera.native.tensor_row_bound", builder.getI64IntegerAttr(rowBound));
    if (shapeProfile) {
      child->setAttr("tessera.native.tensor_program_shape_profile", builder.getUnitAttr());
      SmallVector<Attribute> capacities;
      SmallVector<int64_t> bindings;
      for (Value value : op->getOperands()) {
        capacities.push_back(TypeAttr::get(bufferType(value)));
        llvm::append_range(bindings, shapeProfile->symbols.find(value)->second);
      }
      capacities.push_back(TypeAttr::get(bufferType(op->getResult(0))));
      llvm::append_range(bindings, shapeProfile->symbols.find(op->getResult(0))->second);
      child->setAttr("tessera.native.tensor_program_capacity_types", builder.getArrayAttr(capacities));
      child->setAttr("tessera.native.tensor_program_dimension_bindings", builder.getDenseI64ArrayAttr(bindings));
      function->setAttr("tessera.native.tensor_shape_capacity",
          builder.getDenseI64ArrayAttr(shapeProfile->capacities.find(op->getResult(0))->second));
      function->setAttr("tessera.native.tensor_dimension_bindings", builder.getDenseI64ArrayAttr(bindings));
    }
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
  const int64_t version = shapeProfile ? 3 : rowBound ? 2 : 1;
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
  SmallVector<Value> activeDimensions;
  if (shapeProfile) {
    SmallVector<int64_t> variables;
    SmallVector<Attribute> capacityTypes, bindings;
    for (auto variable : shapeProfile->variables) {
      Value extent = memref::DimOp::create(hb, root.getLoc(), body->getArgument(variable.argument), variable.axis);
      auto zero = arith::ConstantIndexOp::create(hb, root.getLoc(), 0);
      auto capacity = arith::ConstantIndexOp::create(hb, root.getLoc(), variable.capacity);
      auto positive = arith::CmpIOp::create(hb, root.getLoc(), arith::CmpIPredicate::ugt, extent, zero);
      auto fits = arith::CmpIOp::create(hb, root.getLoc(), arith::CmpIPredicate::ule, extent, capacity);
      auto valid = arith::AndIOp::create(hb, root.getLoc(), positive, fits);
      cf::AssertOp::create(hb, root.getLoc(), valid, hb.getStringAttr("native tensor shape extent violates positive capacity"));
      activeDimensions.push_back(extent);
      variables.push_back(variable.argument);
      variables.push_back(variable.axis);
      variables.push_back(variable.capacity);
    }
    for (auto [index, argument] : llvm::enumerate(root.getArguments())) {
      auto bindings = shapeProfile->symbols.find(argument)->second;
      for (unsigned axis = 0; axis < 2; ++axis) {
        int64_t symbol = bindings[axis];
        if (symbol < 0) continue;
        auto origin = shapeProfile->variables[symbol];
        if (origin.argument == index && origin.axis == axis) continue;
        Value actual = memref::DimOp::create(hb, root.getLoc(), body->getArgument(index), axis);
        auto same = arith::CmpIOp::create(hb, root.getLoc(), arith::CmpIPredicate::eq,
                                        actual, activeDimensions[symbol]);
        cf::AssertOp::create(hb, root.getLoc(), same,
                            hb.getStringAttr("native tensor semantic dimension equality violated"));
      }
    }
    for (Value value : shapeProfile->values) {
      capacityTypes.push_back(TypeAttr::get(bufferType(value)));
      bindings.push_back(builder.getDenseI64ArrayAttr(shapeProfile->symbols.find(value)->second));
    }
    module->setAttr("tessera.native.tensor_program_shape_variables", builder.getDenseI64ArrayAttr(variables));
    module->setAttr("tessera.native.tensor_program_capacity_types", builder.getArrayAttr(capacityTypes));
    module->setAttr("tessera.native.tensor_program_shape_bindings", builder.getArrayAttr(bindings));
  }
  SmallVector<Value> privateBuffers;
  for (auto [index, op] : llvm::enumerate(operations)) {
    Value allocation = memref::AllocOp::create(hb, op->getLoc(),
        bufferType(op->getResult(0)));
    Value output = allocation;
    auto type = activeType(op->getResult(0).getType());
    if (shapeProfile && !type.hasStaticShape()) {
      auto symbols = shapeProfile->symbols.find(op->getResult(0))->second;
      auto capacity = shapeProfile->capacities.find(op->getResult(0))->second;
      SmallVector<OpFoldResult> sizes;
      SmallVector<Value> dims;
      for (unsigned axis = 0; axis < 2; ++axis) {
        Value dim = symbols[axis] >= 0 ? activeDimensions[symbols[axis]] :
            arith::ConstantIndexOp::create(hb, op->getLoc(), capacity[axis]).getResult();
        dims.push_back(dim);
        sizes.push_back(symbols[axis] >= 0 ? OpFoldResult(dim) : OpFoldResult(hb.getIndexAttr(capacity[axis])));
      }
      Value count = arith::MulIOp::create(hb, op->getLoc(), dims[0], dims[1]);
      Value limit = arith::ConstantIndexOp::create(hb, op->getLoc(), capacity[0] * capacity[1]);
      auto fits = arith::CmpIOp::create(hb, op->getLoc(), arith::CmpIPredicate::ule, count, limit);
      cf::AssertOp::create(hb, op->getLoc(), fits, hb.getStringAttr("native tensor active view exceeds allocation capacity"));
      SmallVector<OpFoldResult> strides{sizes[1], hb.getIndexAttr(1)};
      output = memref::ReinterpretCastOp::create(hb, op->getLoc(), type, allocation,
          hb.getIndexAttr(0), sizes, strides);
    } else if (rowBound && type.isDynamicDim(0)) {
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
      (schema.getInt() != 1 && schema.getInt() != 2 && schema.getInt() != 3))
    return module.emitError("native tensor frame requires module schema 1, 2 or 3");
  bool bounded = schema.getInt() == 2;
  bool shaped = schema.getInt() == 3;
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
  tessera::native_tensor_program::ShapeMemoryContract memory;
  if (shaped) {
    Operation *first = nullptr;
    for (Operation &op : host.getBody().front())
      if (tessera::native_tensor_program::isInvocation(&op)) { first = &op; break; }
    if (!first || failed(memory.verify(module, host, first))) return failure();
  }
  int64_t rowBound = bounded ? module->getAttrOfType<IntegerAttr>("tessera.native.tensor_program_row_bound").getInt() : 0;
  auto capacityType = [&](Type raw) -> MemRefType {
    return bounded ? tessera::native_tensor_program::capacityType(raw,rowBound) : cast<MemRefType>(raw);
  };
  SmallVector<Type> storage;
  for (Value value : host.getArguments()) storage.push_back(shaped ? memory.capacity(value) : capacityType(value.getType()));
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
    auto type = shaped ? memory.capacity(value) : capacityType(value.getType());
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
    if (auto view = dyn_cast<memref::ReinterpretCastOp>(operation))
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
    {"schema", shaped ? "tessera.native.tensor_program_frame.v3" : bounded ? "tessera.native.tensor_program_frame.v2" : "tessera.native.tensor_program_frame.v1"},
    {"target","nvidia_sm120"},{"architecture","sm_120"},
    {"function",host.getName().str()},{"buffers",std::move(buffers)},
    {"roots",std::move(roots)},{"stages",std::move(stages)},
    {"output",ids.lookup(ret.getOperand(0))}};
  if (shaped) {
    auto program = exportShapeProgram(module, host, ids, stageCount);
    if (failed(program)) return failure();
    frame["shape_program"] = std::move(*program);
    llvm::json::Array bindings, variables;
    for (auto axes : memory.bindings) {
      llvm::json::Array pair;
      for (int64_t axis : axes) pair.push_back(axis);
      bindings.push_back(std::move(pair));
    }
    auto source = module->getAttrOfType<DenseI64ArrayAttr>("tessera.native.tensor_program_shape_variables");
    for (size_t i = 0; i < source.size(); i += 3)
      variables.push_back(llvm::json::Array{source[i], source[i + 1], source[i + 2]});
    frame["shape_bindings"] = std::move(bindings);
    frame["shape_variables"] = std::move(variables);
  }
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
