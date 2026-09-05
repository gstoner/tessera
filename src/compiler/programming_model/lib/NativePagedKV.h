// Native bounded paged-read ownership. Included inside namespace tessera.
namespace {
struct NativePagedKV {
  func::FuncOp function;
  DictionaryAttr contract;
  std::string hash;
};
static FailureOr<NativePagedKV> pagedKVContract(Operation *graph) {
  auto fn = graph->getParentOfType<func::FuncOp>();
  auto mod = graph->getParentOfType<ModuleOp>();
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch = mod->getAttrOfType<StringAttr>("tessera.arch");
  if (!fn || !llvm::hasSingleElement(fn.getBody()) || fn.getNumArguments() != 2 ||
      fn.getNumResults() != 1 || graph->getNumOperands() != 2 || graph->getNumResults() != 1 ||
      !target || target.getValue() != "nvidia_sm120" || !arch || arch.getValue() != "sm_120" ||
      graph->getOperand(0) != fn.getArgument(0) || graph->getOperand(1) != fn.getArgument(1) ||
      graph->getResultTypes() != fn.getResultTypes())
    return graph->emitError("paged read requires an isolated SM120 tensor entry"), failure();
  for (unsigned i = 0; i < fn.getNumArguments(); ++i)
    if (fn.getArgAttr(i, "tessera.layout"))
      return graph->emitError("paged read layout overrides are unsupported"), failure();
  for (NamedAttribute attr : graph->getAttrs())
    if (attr.getName() != "start" && attr.getName() != "end" && attr.getName() != "schedule.artifact_hash")
      return graph->emitError("paged read has an unsupported policy attribute"), failure();
  auto pages = dyn_cast<RankedTensorType>(graph->getOperand(0).getType());
  auto table = dyn_cast<RankedTensorType>(graph->getOperand(1).getType());
  auto start = graph->getAttrOfType<IntegerAttr>("start");
  auto end = graph->getAttrOfType<IntegerAttr>("end");
  // Registered Graph verification has already established shapes and bounds.
  if (!pages || pages.getRank() != 4 || !table || table.getRank() != 1 || !start || !end)
    return graph->emitError("paged read lost its tensor contract"), failure();
  auto names = fn->getAttrOfType<ArrayAttr>("tessera.bindings");
  if (!names || names.size() != 3)
    return graph->emitError("paged read requires three binding names"), failure();
  llvm::SmallDenseSet<StringRef> seen;
  for (Attribute attr : names) {
    auto name = dyn_cast<StringAttr>(attr);
    if (!name || name.getValue().empty() || !seen.insert(name.getValue()).second)
      return graph->emitError("paged read binding names must be unique"), failure();
  }
  OpBuilder builder(graph->getContext());
  auto dims = builder.getDenseI64ArrayAttr({pages.getDimSize(0), table.getDimSize(0),
      pages.getDimSize(1), pages.getDimSize(2), pages.getDimSize(3), start.getInt(), end.getInt()-start.getInt()});
  auto contract = builder.getDictionaryAttr({
      builder.getNamedAttr("shape", dims), builder.getNamedAttr("bindings", names),
      builder.getNamedAttr("target", target), builder.getNamedAttr("arch", arch),
      builder.getNamedAttr("page_ownership", builder.getStringAttr("read_only_borrow")),
      builder.getNamedAttr("table_bounds", builder.getStringAttr("runtime_checked_physical_page_indices")),
      builder.getNamedAttr("layout", builder.getStringAttr("row_major"))});
  std::string text; llvm::raw_string_ostream os(text); contract.print(os); os.flush();
  return NativePagedKV{fn, contract, llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true)};
}
static LogicalResult scheduleNativePagedKV(ModuleOp mod) {
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) { if (op->getName().getStringRef() == "tessera.paged_kv_read") ops.push_back(op); });
  for (Operation *graph : ops) {
    auto c = pagedKVContract(graph); if (failed(c)) return failure();
    auto ret = dyn_cast<func::ReturnOp>(c->function.getBody().front().back());
    if (c->function.getBody().front().getOperations().size() != 2 || !ret || ret.getOperands() != graph->getResults())
      return graph->emitError("paged read must return its sole tensor result");
    OpBuilder builder(graph); builder.setInsertionPointAfter(graph);
    graph->setAttr("schedule.artifact_hash", builder.getStringAttr(c->hash));
    OperationState state(graph->getLoc(), "schedule.paged_kv_read");
    state.addOperands(graph->getResults()); state.addTypes(graph->getResultTypes());
    state.addAttribute("artifact_hash", builder.getStringAttr(c->hash)); state.addAttribute("contract", c->contract);
    auto scheduled = builder.create(state);
    graph->getResult(0).replaceAllUsesExcept(scheduled->getResult(0), scheduled);
  }
  return success();
}
static LogicalResult lowerNativePagedKV(ModuleOp mod) {
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) { if (op->getName().getStringRef() == "schedule.paged_kv_read") ops.push_back(op); });
  for (Operation *scheduled : ops) {
    auto graph = scheduled->getOperand(0).getDefiningOp();
    if (!graph || graph->getName().getStringRef() != "tessera.paged_kv_read")
      return scheduled->emitError("paged schedule requires retained native producer");
    auto c = pagedKVContract(graph); if (failed(c)) return failure();
    auto hash = scheduled->getAttrOfType<StringAttr>("artifact_hash");
    auto ret = dyn_cast<func::ReturnOp>(c->function.getBody().front().back());
    if (!hash || hash.getValue() != c->hash || graph->getAttr("schedule.artifact_hash") != hash ||
        scheduled->getAttr("contract") != c->contract || scheduled->getAttrs().size() != 2 ||
        scheduled->getResultTypes() != graph->getResultTypes() || !ret || ret.getOperands() != scheduled->getResults() ||
        c->function.getBody().front().getOperations().size() != 3)
      return scheduled->emitError("paged Schedule contract changed after hashing");
    StringRef entry = "tessera_tile_paged_kv_read_f32_direct";
    if (SymbolTable::lookupSymbolIn(mod, entry)) return scheduled->emitError("paged entry collision");
    OpBuilder builder(mod.getContext()); builder.setInsertionPointToEnd(mod.getBody());
    SmallVector<Type> args(3, LLVM::LLVMPointerType::get(mod.getContext())); args.append(7, builder.getI64Type());
    auto fn = LLVM::LLVMFuncOp::create(builder, scheduled->getLoc(), entry,
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(mod.getContext()), args, false));
    fn->setAttr("nvvm.kernel", builder.getUnitAttr()); fn->setAttr("tessera.schedule_hash", hash);
    fn->setAttr("tessera.native_contract", c->contract);
    auto block = fn.addEntryBlock(builder); builder.setInsertionPointToStart(block);
    OperationState kernel(scheduled->getLoc(), "tile.paged_kv_read_kernel"); kernel.addOperands(block->getArguments());
    kernel.addAttribute("storage", builder.getStringAttr("f32")); kernel.addAttribute("table_storage", builder.getStringAttr("i32"));
    kernel.addAttribute("route", builder.getStringAttr("direct")); builder.create(kernel);
    LLVM::ReturnOp::create(builder, scheduled->getLoc(), ValueRange{}); c->function.erase();
  }
  return success();
}
} // namespace
