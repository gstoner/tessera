// Bounded absolute ownership through the existing durable Schedule record.
namespace {
static FailureOr<DictionaryAttr> absoluteContract(Operation *op) {
  auto fn = op->getParentOfType<func::FuncOp>();
  auto mod = op->getParentOfType<ModuleOp>();
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch = mod->getAttrOfType<StringAttr>("tessera.arch");
  if (!fn || !fn.getBody().hasOneBlock() || fn.getNumArguments() != 1 || fn.getNumResults() != 1 ||
      op->getNumOperands() != 1 || op->getNumResults() != 1 || op->getOperand(0) != fn.getArgument(0) ||
      !target || target.getValue() != "x86" || !arch || arch.getValue() != "zen5-avx512")
    return op->emitError("absolute requires an isolated x86 f32 entry"), failure();
  auto ty = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!ty || !ty.hasStaticShape() || ty.getRank() < 1 || !ty.getElementType().isF32() ||
      ty.getEncoding() || op->getResult(0).getType() != ty || fn.getResultTypes()[0] != ty)
    return op->emitError("absolute requires plain static shape-preserving f32 tensors"), failure();
  if (auto attrs = fn.getArgAttrDict(0)) for (NamedAttribute attr : attrs) {
    if (attr.getName() == "tessera.layout") {
      auto layout = dyn_cast<StringAttr>(attr.getValue());
      if (!layout || layout.getValue() != "row_major")
        return op->emitError("absolute requires row_major argument layout"), failure();
      continue;
    }
    auto names = dyn_cast<ArrayAttr>(attr.getValue());
    if (attr.getName() != "tessera.dim_names" || !names || names.size() != static_cast<size_t>(ty.getRank()))
      return op->emitError("absolute argument policy is unsupported"), failure();
    for (int64_t axis = 0; axis < ty.getRank(); ++axis) {
      auto name = dyn_cast<StringAttr>(names[axis]);
      int64_t extent;
      if (!name || name.getValue().empty() ||
          (!name.getValue().getAsInteger(10, extent) && extent != ty.getDimSize(axis)))
        return op->emitError("absolute dimension names disagree with static shape"), failure();
    }
  }
  int64_t elements = 1;
  for (int64_t dim : ty.getShape()) {
    if (dim <= 0 || elements > std::numeric_limits<int64_t>::max() / dim)
      return op->emitError("absolute shape exceeds its runtime ABI"), failure();
    elements *= dim;
  }
  for (NamedAttribute attr : op->getAttrs())
    if (attr.getName() != "schedule.artifact_hash" &&
        !(attr.getName() == "tessera.effect_kind" && attr.getValue() == StringAttr::get(op->getContext(), "pure")))
      return op->emitError("absolute has an unsupported policy attribute"), failure();
  auto names = mod->getAttrOfType<ArrayAttr>("tessera.launch_bindings");
  if (!names || names.size() != 2 || !isa<StringAttr>(names[0]) || !isa<StringAttr>(names[1]) || names[0] == names[1])
    return op->emitError("absolute requires distinct input/output binding names"), failure();
  OpBuilder b(op);
  return b.getDictionaryAttr({b.getNamedAttr("shape", b.getDenseI64ArrayAttr(ty.getShape())),
      b.getNamedAttr("bindings", names), b.getNamedAttr("kind", b.getStringAttr("abs")),
      b.getNamedAttr("storage", b.getStringAttr("f32")), b.getNamedAttr("layout", b.getStringAttr("row_major")),
      b.getNamedAttr("numeric_policy", b.getStringAttr("ieee_abs_clear_sign")),
      b.getNamedAttr("elements", b.getI64IntegerAttr(elements))});
}
static std::string absoluteHash(DictionaryAttr contract) {
  std::string text; llvm::raw_string_ostream os(text); contract.print(os); os.flush();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}
static LogicalResult scheduleNativeAbsolute(ModuleOp mod) {
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  if (!target || target.getValue() != "x86") return success();
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) { if (op->getName().getStringRef() == "tessera.absolute" || op->getName().getStringRef() == "tessera.abs") ops.push_back(op); });
  for (auto *op : ops) {
    auto contract = absoluteContract(op); if (failed(contract)) return failure();
    auto fn = op->getParentOfType<func::FuncOp>();
    auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
    if (fn.getBody().front().getOperations().size() != 2 || !ret || ret.getOperands() != op->getResults())
      return op->emitError("absolute must return its only operation");
    OpBuilder b(op); b.setInsertionPointAfter(op);
    auto hash = b.getStringAttr(absoluteHash(*contract));
    op->setAttr("schedule.artifact_hash", hash);
    OperationState state(op->getLoc(), "schedule.artifact");
    state.addAttribute("hash", hash); state.addAttribute("arch", b.getStringAttr("zen5-avx512"));
    state.addAttribute("shape_key", b.getStringAttr("family=absolute"));
    state.addAttribute("contract", *contract);
    b.create(state);
  }
  return success();
}
static LogicalResult lowerNativeAbsolute(ModuleOp mod) {
  SmallVector<schedule::ArtifactOp> records;
  mod.walk([&](schedule::ArtifactOp op) { if (op.getShapeKey() == "family=absolute") records.push_back(op); });
  for (auto record : records) {
    auto fn = record->getParentOfType<func::FuncOp>();
    if (!fn || !fn.getBody().hasOneBlock() || fn.getBody().front().getOperations().size() != 3)
      return record.emitError("absolute Schedule requires its isolated Graph parent");
    auto *op = &fn.getBody().front().front();
    if (op->getName().getStringRef() != "tessera.absolute" && op->getName().getStringRef() != "tessera.abs")
      return record.emitError("absolute Schedule lost its Graph operation");
    auto contract = absoluteContract(op);
    auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
    if (failed(contract) || !ret || ret.getOperands() != op->getResults() ||
        record->getAttr("contract") != *contract || record.getHash() != absoluteHash(*contract) ||
        record.getArch() != "zen5-avx512" || op->getAttr("schedule.artifact_hash") != record->getAttr("hash"))
      return record.emitError("absolute Schedule contract was altered");
    OpBuilder b(mod.getBody(), mod.getBody()->end());
    auto ptr = LLVM::LLVMPointerType::get(mod.getContext());
    auto type = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(mod.getContext()), {ptr, ptr, b.getI64Type()}, false);
    auto kernel = LLVM::LLVMFuncOp::create(b, op->getLoc(), "tessera_tile_x86_unary_abs", type);
    kernel->setAttr("tessera.absolute_contract", *contract);
    kernel->setAttr("tessera.schedule_hash", record->getAttr("hash"));
    Block *entry = kernel.addEntryBlock(b); b.setInsertionPointToStart(entry);
    OperationState tile(op->getLoc(), "tile.elementwise_kernel");
    tile.addOperands(entry->getArguments());
    tile.addAttribute("family", b.getStringAttr("unary")); tile.addAttribute("kind", b.getStringAttr("abs"));
    tile.addAttribute("storage", b.getStringAttr("f32")); tile.addAttribute("output_storage", b.getStringAttr("f32"));
    b.create(tile); LLVM::ReturnOp::create(b, op->getLoc(), ValueRange{});
    fn.erase();
  }
  return success();
}
} // namespace
