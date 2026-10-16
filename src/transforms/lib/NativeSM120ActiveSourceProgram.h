// Native SSA outlining only. Included inside namespace tessera by the tensor
// exporter. Physical Schedule/Tile row-body lowering remains a separate gate.
static std::string nativeSM120RowBodyHash(mlir::func::FuncOp function,
                                        bool normalizeParentAD = false) {
  using namespace mlir;
  auto copy = cast<func::FuncOp>(function->clone());
  SmallVector<StringAttr> remove;
  for (NamedAttribute attr : copy->getAttrs()) {
    auto key = attr.getName().getValue();
    if ((normalizeParentAD && key.starts_with("tessera.autodiff")) ||
        key == "tessera.native.sm120_tensor_member" ||
        key == "tessera.native.sm120_row_producer")
      remove.push_back(attr.getName());
  }
  for (auto key : remove) copy->removeAttr(key);
  copy.setSymName("__native_row_body");
  copy.setPrivate();
  std::string text;
  llvm::raw_string_ostream stream(text);
  copy.print(stream, OpPrintingFlags().useLocalScope());
  stream.flush();
  copy->destroy();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}

static mlir::LogicalResult verifyNativeSM120RowOutline(mlir::func::FuncOp function) {
  using namespace mlir;
  auto module = function->getParentOfType<ModuleOp>();
  auto target = module ? module->getAttrOfType<StringAttr>("tessera.target") : StringAttr{};
  auto arch = module ? module->getAttrOfType<StringAttr>("tessera.arch") : StringAttr{};
  if (!target || target.getValue() != "nvidia_sm120" || !arch || arch.getValue() != "sm_120")
    return function.emitError("SM120 row outline requires its owning target");
  auto c = function->getAttrOfType<DictionaryAttr>("tessera.native.sm120_row_producer");
  auto schema = c ? c.getAs<IntegerAttr>("schema") : IntegerAttr{};
  auto rows = c ? c.getAs<IntegerAttr>("rows") : IntegerAttr{};
  auto columns = c ? c.getAs<IntegerAttr>("columns") : IntegerAttr{};
  auto storage = c ? c.getAs<StringAttr>("storage") : StringAttr{};
  auto accum = c ? c.getAs<StringAttr>("accum") : StringAttr{};
  auto hash = c ? c.getAs<StringAttr>("body_hash") : StringAttr{};
  auto reject = [&]() {
    return function.emitError("SM120 row outline differs from its typed body seal");
  };
  if (!c || c.size() != 6 || !schema || !schema.getType().isInteger(64) ||
      schema.getInt() != 1 || !rows || !rows.getType().isInteger(64) ||
      rows.getInt() <= 0 || rows.getInt() > 65535LL*16 ||
      !columns || !columns.getType().isInteger(64) ||
      columns.getInt() <= 0 || columns.getInt() > INT32_MAX ||
      !storage || (storage.getValue() != "f16" && storage.getValue() != "bf16") ||
      !accum || accum.getValue() != "f32" || !hash || hash.getValue().size() != 64 ||
      function.getNumArguments() != 2 || function.getNumResults() != 2 ||
      !function.getBody().hasOneBlock())
    return reject();
  auto type = dyn_cast<RankedTensorType>(function.getArgument(0).getType());
  if (!type || type.getRank() != 2 || !type.hasStaticShape() || type.getEncoding() ||
      type.getDimSize(0) != rows.getInt() || type.getDimSize(1) != columns.getInt() ||
      (storage.getValue() == "f16" ? !type.getElementType().isF16() : !type.getElementType().isBF16()) ||
      function.getArgument(1).getType() != type ||
      function.getResultTypes()[0] != type || function.getResultTypes()[1] != type ||
      nativeSM120RowBodyHash(function) != hash.getValue())
    return reject();
  return success();
}

static mlir::LogicalResult emitNativeSM120ActiveSourceOutline(
    mlir::ModuleOp module, mlir::func::FuncOp root,
    mlir::func::FuncOp parent, llvm::StringRef originalSource) {
  using namespace mlir;
  auto reject = [&]() {
    return root.emitError("SM120 active-source outline requires the native static explicit-fp32 RMSNorm/ReLU JVP");
  };
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  auto target = module->getAttrOfType<StringAttr>("tessera.target");
  if (!arch || arch.getValue() != "sm_120" || !target ||
      target.getValue() != "nvidia_sm120" ||
      module->hasAttr("tessera.native.sm120_tensor_bounds") ||
      !parent || !parent.getBody().hasOneBlock() || parent.getNumArguments() != 2 ||
      parent.getNumResults() != 1 || parent.getBody().front().getOperations().size() != 4 ||
      root.getNumArguments() != 3 || root.getNumResults() != 2 ||
      !root.getBody().hasOneBlock() || originalSource.empty())
    return reject();
  SmallVector<Operation *> primal;
  for (auto &op : parent.getBody().front().without_terminator()) primal.push_back(&op);
  if (primal[0]->getName().getStringRef() != "tessera.rmsnorm" ||
      primal[1]->getName().getStringRef() != "tessera.relu" ||
      primal[2]->getName().getStringRef() != "tessera.matmul" ||
      primal[0]->getNumOperands() != 1 || primal[0]->getNumResults() != 1 ||
      primal[1]->getNumOperands() != 1 || primal[1]->getNumResults() != 1 ||
      primal[2]->getNumOperands() != 2 || primal[2]->getNumResults() != 1 ||
      primal[1]->getOperand(0) != primal[0]->getResult(0) ||
      primal[2]->getOperand(0) != primal[1]->getResult(0))
    return reject();
  // Keep this first tensor case plain: fused/transposed consumers require
  // their own derivative and physical operand contract.
  // Native AD need not repeat default-valued metadata on tangent matmuls.
  // Validate the same complete plain semantic envelope on every product.
  auto plainMatmulAttrs = [&](Operation *product) {
  for (NamedAttribute field : product->getAttrs()) {
    StringRef key = field.getName().getValue();
    if (key == "tessera.effect_kind") {
      auto value = dyn_cast<StringAttr>(field.getValue());
      if (!value || value.getValue() != "pure") return false;
    } else if (key == "transposeA" || key == "transposeB") {
      auto value = dyn_cast<BoolAttr>(field.getValue());
      if (!value || value.getValue()) return false;
    } else if (key == "output_dtype") {
      auto value = dyn_cast<StringAttr>(field.getValue());
      if (!value || value.getValue() != "fp32") return false;
    } else if (key == "activation" || key == "rhs_storage_order") {
      auto value = dyn_cast<StringAttr>(field.getValue());
      if (!value || value.getValue() != (key == "activation" ? "none" : "row_major"))
        return false;
    } else return false;
  }
    return true;
  };
  if (!plainMatmulAttrs(primal[2])) return reject();
  for (NamedAttribute field : primal[0]->getAttrs()) {
    StringRef key = field.getName().getValue();
    if (key != "eps" && key != "axis" && key != "numeric_policy" &&
        key != "tessera.effect_kind") return reject();
  }
  if (auto axis = primal[0]->getAttrOfType<IntegerAttr>("axis");
      axis && axis.getInt() != -1 && axis.getInt() != 1) return reject();
  auto epsilon = primal[0]->getAttrOfType<FloatAttr>("eps");
  float physicalEpsilon = float(epsilon ? epsilon.getValueAsDouble() : 1e-5);
  if (!std::isfinite(physicalEpsilon) || physicalEpsilon <= 0) return reject();
  auto sourceArg = dyn_cast<BlockArgument>(primal[0]->getOperand(0));
  auto rhsArg = dyn_cast<BlockArgument>(primal[2]->getOperand(1));
  auto wrt = parent->getAttrOfType<ArrayAttr>("tessera.autodiff.wrt_indices");
  auto role = wrt && wrt.size() == 1 ? dyn_cast<IntegerAttr>(wrt[0]) : IntegerAttr{};
  if (!sourceArg || !rhsArg || sourceArg.getOwner() != &parent.getBody().front() ||
      rhsArg.getOwner() != &parent.getBody().front() ||
      sourceArg.getArgNumber() == rhsArg.getArgNumber() || !role ||
      role.getInt() != sourceArg.getArgNumber())
    return reject();
  auto source = dyn_cast<RankedTensorType>(sourceArg.getType());
  auto rhs = dyn_cast<RankedTensorType>(rhsArg.getType());
  auto result = dyn_cast<RankedTensorType>(parent.getResultTypes()[0]);
  if (!source || !rhs || !result || source.getRank() != 2 || rhs.getRank() != 2 ||
      result.getRank() != 2 || !source.hasStaticShape() || !rhs.hasStaticShape() ||
      !result.hasStaticShape() || source.getEncoding() || rhs.getEncoding() ||
      result.getEncoding() || (!source.getElementType().isF16() &&
      !source.getElementType().isBF16()) ||
      rhs.getElementType() != source.getElementType() || !result.getElementType().isF32() ||
      rhs.getDimSize(0) != source.getDimSize(1) ||
      result.getDimSize(0) != source.getDimSize(0) ||
      result.getDimSize(1) != rhs.getDimSize(1))
    return reject();
  int64_t m = source.getDimSize(0), k = source.getDimSize(1), n = rhs.getDimSize(1);
  if (m <= 0 || k <= 0 || n <= 0 || m > 65535LL*16 ||
      k > INT32_MAX || n > INT32_MAX ||
      m > INT64_MAX/k/2 || k > INT64_MAX/n/2 || m > INT64_MAX/n/4)
    return reject();
  auto hasAttrs = [](DictionaryAttr attrs) { return attrs && !attrs.empty(); };
  if (primal[0]->getResult(0).getType() != source ||
      primal[1]->getResult(0).getType() != source) return reject();
  for (unsigned i = 0; i < 2; ++i)
    if (root.getArgument(i).getType() != parent.getArgument(i).getType() ||
        hasAttrs(parent.getArgAttrDict(i)) || hasAttrs(root.getArgAttrDict(i)))
      return reject();
  if (root.getArgument(2).getType() != source ||
      hasAttrs(root.getArgAttrDict(2)) ||
      root.getResultTypes()[0] != result || root.getResultTypes()[1] != result)
    return reject();
  auto policy = primal[0]->getAttrOfType<DictionaryAttr>("numeric_policy");
  auto accum = policy ? policy.getAs<StringAttr>("accum") : StringAttr{};
  auto storage = policy ? policy.getAs<StringAttr>("storage") : StringAttr{};
  StringRef storageName = source.getElementType().isF16() ? "fp16" : "bf16";
  if (!policy || !accum || accum.getValue() != "fp32" ||
      (policy.get("storage") && (!storage || storage.getValue() != storageName)))
    return reject();
  for (auto field : policy)
    if (field.getName().getValue() != "storage" &&
        field.getName().getValue() != "accum")
      return reject();
  auto original = parseSourceString<ModuleOp>(originalSource, module.getContext());
  SmallVector<func::FuncOp> witnesses;
  if (original) for (auto fn : original->getOps<func::FuncOp>()) witnesses.push_back(fn);
  if (witnesses.size() != 1 || failed(verify(*original)) ||
      nativeSM120RowBodyHash(witnesses[0], true) != nativeSM120RowBodyHash(parent, true))
    return reject();
  SmallVector<Operation *> ops;
  for (auto &op : root.getBody().front().without_terminator()) ops.push_back(&op);
  if (ops.size() < 4 || ops.size() > 64) return reject();
  auto first = ops[ops.size()-2], second = ops.back();
  if (first->getNumResults() != 1 || second->getNumResults() != 1) return reject();
  auto ret = dyn_cast<func::ReturnOp>(root.getBody().front().getTerminator());
  if (!ret || ret.getNumOperands() != 2 ||
      ret.getOperand(0) != first->getResult(0) ||
      ret.getOperand(1) != second->getResult(0))
    return reject();
  Value x = root.getArgument(sourceArg.getArgNumber());
  Value b = root.getArgument(rhsArg.getArgNumber());
  Value dx = root.getArgument(2);
  for (Operation *product : {first, second})
    if (product->getName().getStringRef() != "tessera.matmul" ||
        product->getNumOperands() != 2 || product->getNumResults() != 1 ||
        product->getOperand(1) != b || product->getResult(0).getType() != result ||
        !plainMatmulAttrs(product))
      return reject();
  auto relu = first->getOperand(0).getDefiningOp();
  auto norm = relu && relu->getNumOperands() == 1 ? relu->getOperand(0).getDefiningOp() : nullptr;
  if (!relu || !norm || relu->getName() != primal[1]->getName() ||
      norm->getName() != primal[0]->getName() || norm->getNumOperands() != 1 ||
      norm->getOperand(0) != x || norm->getAttrs() != primal[0]->getAttrs() ||
      relu->getAttrs() != primal[1]->getAttrs() ||
      first->getOperand(0).getType() != source ||
      second->getOperand(0).getType() != source)
    return reject();

  llvm::SmallPtrSet<Operation *,32> live;
  llvm::SmallDenseSet<Value> roots;
  std::function<bool(Value)> visit = [&](Value value) {
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg.getOwner() != &root.getBody().front() || (value != x && value != dx))
        return false;
      roots.insert(value); return true;
    }
    auto producer = value.getDefiningOp();
    if (!producer || producer == first || producer == second ||
        producer->getNumRegions() || !isMemoryEffectFree(producer))
      return false;
    if (!live.insert(producer).second) return true;
    auto name = producer->getName().getStringRef();
    if (name != "tessera.rmsnorm" && name != "tessera.relu" &&
        name != "tessera.cast" && name != "tessera.normalization_stats" &&
        name != "tessera.broadcast_in_dim" && name != "tessera.mul" &&
        name != "tessera.reduce" && name != "tessera.sub" &&
        name != "tessera.add" && name != "arith.constant" &&
        name != "tessera.compare_scalar" && name != "tessera.masked_fill")
      return false;
    for (auto type : producer->getResultTypes()) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      if (name == "arith.constant") {
        auto dense = producer->getAttrOfType<DenseElementsAttr>("value");
        if (!tensor || tensor.getRank() != 0 || tensor.getEncoding() ||
            !tensor.getElementType().isF32() || producer->getNumOperands() ||
            !dense || !dense.isSplat() || !dense.getSplatValue<APFloat>().isFinite())
          return false;
        continue;
      }
      if (!tensor || !tensor.hasStaticShape() || tensor.getEncoding() ||
          (tensor.getRank() != 1 && tensor.getRank() != 2) ||
          tensor.getDimSize(0) != m ||
          (tensor.getRank() == 2 && tensor.getDimSize(1) != k) ||
          (tensor.getElementType() != source.getElementType() &&
           !tensor.getElementType().isF32() && !tensor.getElementType().isInteger(1)))
        return false;
    }
    for (Value operand : producer->getOperands()) if (!visit(operand)) return false;
    return true;
  };
  if (!visit(first->getOperand(0)) || !visit(second->getOperand(0)) ||
      !roots.contains(x) || !roots.contains(dx) || live.size() != ops.size()-2)
    return reject();
  for (unsigned i=0;i<3;++i)
    if (SymbolTable::lookupSymbolIn(module,(root.getSymName()+"__tensor_member_"+llvm::Twine(i)).str()))
      return reject();
  std::string paired;
  llvm::raw_string_ostream pairedStream(paired); module.print(pairedStream); pairedStream.flush();
  OpBuilder builder(module.getContext());
  SmallVector<Attribute> symbols;
  llvm::json::Array graphs, steps;
  for (unsigned i=0;i<3;++i) {
    builder.setInsertionPointToEnd(module.getBody());
    auto symbol=(root.getSymName()+"__tensor_member_"+llvm::Twine(i)).str();
    SmallVector<Type> ins = i==0 ? SmallVector<Type>{source,source} : SmallVector<Type>{source,rhs};
    SmallVector<Type> outs = i==0 ? SmallVector<Type>{source,source} : SmallVector<Type>{result};
    auto member=func::FuncOp::create(builder,root.getLoc(),symbol,builder.getFunctionType(ins,outs));
    member.setPrivate();
    member->setAttr("tessera.native.sm120_tensor_member",builder.getI64IntegerAttr(i));
    auto block=member.addEntryBlock();
    OpBuilder body(block,block->begin()); IRMapping mapping;
    if (i==0) {
      mapping.map(x,block->getArgument(0)); mapping.map(dx,block->getArgument(1));
      for (auto *op : ArrayRef<Operation *>(ops).drop_back(2)) body.clone(*op,mapping);
      func::ReturnOp::create(body,root.getLoc(),ValueRange{
          mapping.lookup(first->getOperand(0)),mapping.lookup(second->getOperand(0))});
      member->setAttr("tessera.native.sm120_row_producer",builder.getDictionaryAttr({
        builder.getNamedAttr("schema",builder.getI64IntegerAttr(1)),
        builder.getNamedAttr("rows",builder.getI64IntegerAttr(m)),
        builder.getNamedAttr("columns",builder.getI64IntegerAttr(k)),
        builder.getNamedAttr("storage",builder.getStringAttr(source.getElementType().isF16()?"f16":"bf16")),
        builder.getNamedAttr("accum",builder.getStringAttr("f32")),
        builder.getNamedAttr("body_hash",builder.getStringAttr(nativeSM120RowBodyHash(member)))}));
    } else {
      auto product=i==1?first:second;
      mapping.map(product->getOperand(0),block->getArgument(0));
      mapping.map(b,block->getArgument(1));
      auto cloned=body.clone(*product,mapping);
      // Program roots and private edges are compact row-major buffers.
      cloned->setAttr("rhs_storage_order",builder.getStringAttr("row_major"));
      func::ReturnOp::create(body,root.getLoc(),cloned->getResults());
    }
    if (i == 0 && failed(verifyNativeSM120RowOutline(member))) return failure();
    symbols.push_back(FlatSymbolRefAttr::get(member));
    auto projected=ModuleOp::create(module.getLoc());projected->setAttrs(module->getAttrs());
    projected.getBody()->push_back(cloneNativeSM120TensorMember(member,i));
    std::string text;llvm::raw_string_ostream stream(text);projected.print(stream);stream.flush();
    graphs.push_back(text+"\n");projected.erase();
    llvm::json::Array inputs = i==0 ? llvm::json::Array{int64_t(sourceArg.getArgNumber()),int64_t(2)} :
        llvm::json::Array{int64_t(i==1?3:4),int64_t(rhsArg.getArgNumber())};
    llvm::json::Array outputs = i==0 ? llvm::json::Array{int64_t(3),int64_t(4)} :
        llvm::json::Array{int64_t(i==1?5:6)};
    steps.push_back(llvm::json::Object{{"step",int64_t(i)},{"member",symbol},
        {"operation",i==0?"native_row_region":"tessera.matmul"},
        {"inputs",std::move(inputs)},{"outputs",std::move(outputs)}});
  }
  SmallVector<Type> types(root.getArgumentTypes());types.append({source,source,result,result});
  llvm::json::Array buffers;
  for (unsigned i=0;i<7;++i) {
    auto type=cast<RankedTensorType>(types[i]);
    uint64_t bytes=type.getElementTypeBitWidth()/8;
    llvm::json::Array shape;
    for (int64_t dim : type.getShape()) {bytes*=dim;shape.push_back(dim);}
    int64_t write=i<3?-1:i<5?0:i==5?1:2;
    int64_t read=i<3?(i==rhsArg.getArgNumber()?2:0):i==3?1:i==4?2:3;
    std::string storage;llvm::raw_string_ostream stream(storage);type.getElementType().print(stream);stream.flush();
    buffers.push_back(llvm::json::Object{{"id",int64_t(i)},{"shape",std::move(shape)},
        {"storage",storage},{"bytes",int64_t(bytes)},
        {"ownership",i<3?"readonly_input":i<5?"private_scratch":"returned_output"},
        {"first_write",write},{"last_read",read}});
  }
  llvm::json::Object plan{{"schema","tessera.native.sm120_active_source_outline.v1"},
    {"execution_state","artifact_only"},{"kind","paired_active_source_jvp"},
    {"source_graph_ir",originalSource.str()},{"paired_graph_ir",paired},
    {"root",root.getSymName().str()},
    {"role_indices",llvm::json::Array{int64_t(sourceArg.getArgNumber()),int64_t(rhsArg.getArgNumber()),int64_t(2)}},
    {"buffers",std::move(buffers)},{"steps",std::move(steps)},
    {"outputs",llvm::json::Array{int64_t(5),int64_t(6)}},{"member_graphs",std::move(graphs)}};
  std::string json;llvm::raw_string_ostream stream(json);stream<<llvm::json::Value(std::move(plan));stream.flush();
  module->setAttr("tessera.native.sm120_tensor_program_json",builder.getStringAttr(llvm::encodeBase64(json)));
  module->setAttr("tessera.native.sm120_tensor_program",builder.getDictionaryAttr({
    builder.getNamedAttr("members",builder.getArrayAttr(symbols))}));
  return success();
}
