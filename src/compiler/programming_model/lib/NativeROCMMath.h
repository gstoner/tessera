// E2E-REAL-6: native static ROCm math recipes. Admission is explicitly
// marked by host bindings. The original Graph remains in the Schedule;
// replay recomputes its full contract before materializing launch-level Tile.
namespace {
static StringRef rocmMathKind(Operation *op) {
  return llvm::StringSwitch<StringRef>(op->getName().getStringRef())
      .Case("tessera.sqrt", "sqrt").Case("tessera.exp", "exp")
      .Case("tessera.add", "add").Case("tessera.div", "div")
      .Case("tessera.cumsum", "sum").Case("tessera.cummax", "max")
      .Default("");
}
static bool rocmMathScan(Operation *op) {
  auto n = op->getName().getStringRef();
  return n == "tessera.cumsum" || n == "tessera.cummax";
}
static bool rocmMathBinary(Operation *op) {
  auto n = op->getName().getStringRef();
  return n == "tessera.add" || n == "tessera.div";
}
static FailureOr<DictionaryAttr> rocmMathContract(Operation *op) {
  auto mod = op->getParentOfType<ModuleOp>();
  auto fn = op->getParentOfType<func::FuncOp>();
  auto arch = mod->getAttrOfType<StringAttr>("tessera.arch");
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  auto names = mod->getAttrOfType<ArrayAttr>("tessera.launch_bindings");
  unsigned inputs = rocmMathBinary(op) ? 2 : 1;
  if (!fn || !fn.getBody().hasOneBlock() || !target ||
      target.getValue() != "rocm" || !arch ||
      (arch.getValue() != "gfx1151" && arch.getValue() != "gfx1201") ||
      fn.getNumArguments() != inputs || fn.getNumResults() != 1 ||
      op->getNumOperands() != inputs || op->getNumResults() != 1 ||
      !names || names.size() != inputs + 1)
    return op->emitError("ROCm math requires isolated operands, target and launch bindings"), failure();
  auto ty = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!ty || !ty.hasStaticShape() || ty.getRank() < 1 ||
      !ty.getElementType().isF32() || ty.getEncoding() ||
      op->getResult(0).getType() != ty || fn.getResultTypes()[0] != ty)
    return op->emitError("ROCm math requires plain static shape-preserving f32 tensors"), failure();
  llvm::StringSet<> distinct;
  for (auto name : names) {
    auto s = dyn_cast<StringAttr>(name);
    if (!s || s.getValue().empty() || !distinct.insert(s.getValue()).second)
      return op->emitError("ROCm math launch aliases must be nonempty and distinct"), failure();
  }
  SmallVector<Attribute> roles;
  SmallVector<Operation *> casts;
  OpBuilder b(op);
  auto inputTy = dyn_cast<RankedTensorType>(fn.getArgument(0).getType());
  if (!inputTy || inputTy.getShape() != ty.getShape() || inputTy.getEncoding() ||
      (!inputTy.getElementType().isF32() && !inputTy.getElementType().isF16() &&
       !inputTy.getElementType().isBF16()))
    return op->emitError("ROCm math requires f16/bf16/f32 storage and f32 computation"), failure();
  for (Value operand : op->getOperands()) {
    Value source = operand;
    if (auto *cast = operand.getDefiningOp()) {
      if (cast->getName().getStringRef() != "tessera.cast" ||
          cast->getNumOperands() != 1 || cast->getNumResults() != 1 ||
          cast->getResult(0).getType() != ty || !cast->getResult(0).hasOneUse())
        return op->emitError("ROCm math supports only single-use exact widening casts"), failure();
      for (auto attr : cast->getAttrs()) {
        if (attr.getName() == "tessera.effect_kind" && attr.getValue() == b.getStringAttr("pure")) continue;
        auto dtype = dyn_cast<StringAttr>(attr.getValue());
        if (attr.getName() == "dtype" && dtype &&
            (dtype.getValue() == "fp32" || dtype.getValue() == "f32" || dtype.getValue() == "float32")) continue;
        return op->emitError("ROCm math widening cast policy must be default and match f32"), failure();
      }
      casts.push_back(cast);
      source = cast->getOperand(0);
    }
    auto arg = dyn_cast<BlockArgument>(source);
    if (!arg || arg.getOwner() != &fn.getBody().front() ||
        source.getType() != inputTy || operand.getType() != ty ||
        (inputTy != ty && source == operand))
      return op->emitError("ROCm math operands require exact f32 widening of same-shaped entry storage"), failure();
    roles.push_back(b.getI64IntegerAttr(arg.getArgNumber()));
  }
  for (Operation &candidate : fn.getBody().front())
    if (&candidate != op && !isa<func::ReturnOp>(candidate) &&
        candidate.getName().getStringRef() != "schedule.artifact" &&
        !llvm::is_contained(casts, &candidate))
      return op->emitError("ROCm math entry contains an unsupported producer"), failure();
  for (unsigned i = 0; i < inputs; ++i) {
    if (fn.getArgument(i).getType() != inputTy)
      return op->emitError("ROCm math entry argument storage/shape disagrees"), failure();
    if (auto attrs = fn.getArgAttrDict(i))
      for (NamedAttribute attr : attrs) {
        if (attr.getName() == "tessera.layout" &&
            attr.getValue() == b.getStringAttr("row_major")) continue;
        auto dims = dyn_cast<ArrayAttr>(attr.getValue());
        if (attr.getName() != "tessera.dim_names" || !dims ||
            dims.size() != static_cast<size_t>(ty.getRank()))
          return op->emitError("ROCm math argument layout policy is unsupported"), failure();
        for (auto [axis, dim] : llvm::enumerate(dims)) {
          auto name = dyn_cast<StringAttr>(dim);
          int64_t extent;
          if (!name || name.getValue().empty() ||
              (!name.getValue().getAsInteger(10, extent) && extent != ty.getDimSize(axis)))
            return op->emitError("ROCm math dimension names disagree with static shape"), failure();
        }
      }
  }
  int64_t count = 1;
  for (int64_t extent : ty.getShape()) {
    if (extent <= 0 || count > std::numeric_limits<int64_t>::max() / extent)
      return op->emitError("ROCm math shape exceeds its signed runtime ABI"), failure();
    count *= extent;
  }
  for (NamedAttribute attr : op->getAttrs()) {
    if (attr.getName() == "schedule.artifact_hash") continue;
    if (attr.getName() == "tessera.effect_kind" && attr.getValue() == b.getStringAttr("pure")) continue;
    auto axis = dyn_cast<IntegerAttr>(attr.getValue());
    if (rocmMathScan(op) && attr.getName() == "axis" && axis &&
        (axis.getInt() == -1 || axis.getInt() == ty.getRank() - 1)) continue;
    return op->emitError("ROCm math has an unsupported numerical or axis policy"), failure();
  }
  return b.getDictionaryAttr({
      b.getNamedAttr("shape", b.getDenseI64ArrayAttr(ty.getShape())),
      b.getNamedAttr("bindings", names), b.getNamedAttr("roles", b.getArrayAttr(roles)),
      b.getNamedAttr("architecture", arch), b.getNamedAttr("kind", b.getStringAttr(rocmMathKind(op))),
      b.getNamedAttr("family", b.getStringAttr(rocmMathScan(op) ? "scan" : rocmMathBinary(op) ? "binary" : "unary")),
      b.getNamedAttr("storage", b.getStringAttr(inputTy.getElementType().isF16() ? "f16" : inputTy.getElementType().isBF16() ? "bf16" : "f32")),
      b.getNamedAttr("output_storage", b.getStringAttr("f32")),
      b.getNamedAttr("elements", b.getI64IntegerAttr(count)),
      b.getNamedAttr("rows", b.getI64IntegerAttr(count / ty.getShape().back())),
      b.getNamedAttr("columns", b.getI64IntegerAttr(ty.getShape().back())),
      b.getNamedAttr("numeric_policy", b.getStringAttr(rocmMathScan(op) ? "f32_inclusive_scan" : "f32_compute"))});
}
static std::string rocmMathHash(DictionaryAttr c) {
  std::string text; llvm::raw_string_ostream os(text); c.print(os); os.flush();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}
static LogicalResult scheduleNativeROCMMath(ModuleOp mod) {
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  if (!target || target.getValue() != "rocm" || !mod->hasAttr("tessera.launch_bindings")) return success();
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) { if (!rocmMathKind(op).empty()) ops.push_back(op); });
  for (auto *op : ops) {
    auto c = rocmMathContract(op); if (failed(c)) return failure();
    auto fn = op->getParentOfType<func::FuncOp>();
    auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
    if (!ret || ret.getOperands() != op->getResults())
      return op->emitError("ROCm math must return its isolated Graph operation");
    OpBuilder b(op); b.setInsertionPointAfter(op);
    auto hash = b.getStringAttr(rocmMathHash(*c)); op->setAttr("schedule.artifact_hash", hash);
    OperationState state(op->getLoc(), "schedule.artifact");
    state.addAttribute("hash", hash); state.addAttribute("arch", c->get("architecture"));
    state.addAttribute("shape_key", b.getStringAttr("family=rocm_math"));
    state.addAttribute("contract", *c); b.create(state);
  }
  return success();
}
static LogicalResult lowerNativeROCMMath(ModuleOp mod) {
  SmallVector<schedule::ArtifactOp> records;
  mod.walk([&](schedule::ArtifactOp op) { if (op.getShapeKey() == "family=rocm_math") records.push_back(op); });
  for (auto record : records) {
    auto fn = record->getParentOfType<func::FuncOp>();
    if (!fn || !fn.getBody().hasOneBlock())
      return record.emitError("ROCm math Schedule lost its isolated Graph parent");
    Operation *op = nullptr;
    for (Operation &candidate : fn.getBody().front())
      if (!rocmMathKind(&candidate).empty()) {
        if (op) return record.emitError("ROCm math Schedule has multiple consumers");
        op = &candidate;
      }
    if (!op) return record.emitError("ROCm math Schedule lost its Graph operation");
    auto c = rocmMathContract(op);
    auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
    if (failed(c) || !ret || ret.getOperands() != op->getResults() ||
        record->getAttr("contract") != *c || record.getHash() != rocmMathHash(*c) ||
        record->getAttr("arch") != c->get("architecture") ||
        op->getAttr("schedule.artifact_hash") != record->getAttr("hash"))
      return record.emitError("ROCm math Schedule contract was altered");
    OpBuilder b(mod.getBody(), mod.getBody()->end());
    auto ptr = LLVM::LLVMPointerType::get(mod.getContext());
    SmallVector<Type> params(fn.getNumArguments() + 1, ptr);
    params.push_back(b.getI64Type()); if (rocmMathScan(op)) params.push_back(b.getI64Type());
    auto type = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(mod.getContext()), params, false);
    auto symbol = (Twine("tessera_tile_rocm_math_") + rocmMathKind(op) + "_" + record.getHash().take_front(12)).str();
    if (SymbolTable::lookupSymbolIn(mod, symbol)) return record.emitError("ROCm math symbol already exists");
    auto kernel = LLVM::LLVMFuncOp::create(b, op->getLoc(), symbol, type);
    kernel->setAttr("tessera.rocm_math_contract", *c);
    kernel->setAttr("tessera.schedule_hash", record->getAttr("hash"));
    Block *entry = kernel.addEntryBlock(b); b.setInsertionPointToStart(entry);
    SmallVector<Value> args;
    for (auto role : cast<ArrayAttr>(c->get("roles"))) args.push_back(entry->getArgument(cast<IntegerAttr>(role).getInt()));
    args.push_back(entry->getArgument(fn.getNumArguments()));
    args.push_back(entry->getArgument(fn.getNumArguments()+1));
    if (rocmMathScan(op)) args.push_back(entry->getArgument(fn.getNumArguments()+2));
    OperationState tile(op->getLoc(), rocmMathScan(op) ? "tile.scan_kernel" : "tile.elementwise_kernel");
    tile.addOperands(args); tile.addAttribute("kind", c->get("kind"));
    tile.addAttribute("storage", c->get("storage"));
    tile.addAttribute("tessera.schedule_hash", record->getAttr("hash"));
    tile.addAttribute("output_storage", c->get("output_storage"));
    if (rocmMathScan(op)) tile.addAttribute("inclusive", b.getBoolAttr(true));
    else {
      tile.addAttribute("family", b.getStringAttr(op->getName().getStringRef() == "tessera.exp" ? "transcendental" : rocmMathBinary(op) ? "binary" : "unary"));
    }
    b.create(tile); LLVM::ReturnOp::create(b, op->getLoc(), ValueRange{}); fn.erase();
  }
  return success();
}
} // namespace
