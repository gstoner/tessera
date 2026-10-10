// Bounded gfx1151 token-gather Schedule owner. Included in namespace tessera.
namespace {
struct NativeMoeDispatch {
  func::FuncOp function;
  DictionaryAttr contract;
  std::string hash;
};

static FailureOr<NativeMoeDispatch> moeDispatchContract(Operation *graph) {
  auto fn = graph->getParentOfType<func::FuncOp>();
  auto mod = graph->getParentOfType<ModuleOp>();
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch = mod->getAttrOfType<StringAttr>("tessera.arch");
  if (!fn || !llvm::hasSingleElement(fn.getBody()) || fn.getNumArguments() != 2 ||
      fn.getNumResults() != 1 || graph->getNumOperands() != 2 ||
      graph->getNumResults() != 1 || !target || !arch ||
      target.getValue() != "rocm_gfx1151" || arch.getValue() != "gfx1151" ||
      !((graph->getOperand(0) == fn.getArgument(0) &&
         graph->getOperand(1) == fn.getArgument(1)) ||
        (graph->getOperand(0) == fn.getArgument(1) &&
         graph->getOperand(1) == fn.getArgument(0))) ||
      graph->getResultTypes() != fn.getResultTypes())
    return graph->emitError("MoE token gather needs isolated gfx1151 Graph entry"), failure();
  for (unsigned i = 0; i < fn.getNumArguments(); ++i)
    if (fn.getArgAttr(i, "tessera.layout"))
      return graph->emitError("MoE token gather does not accept layout overrides"), failure();
  for (NamedAttribute attr : graph->getAttrs())
    if (attr.getName() != "schedule.artifact_hash" &&
        !(attr.getName() == "tessera.effect_kind" &&
          attr.getValue() == StringAttr::get(graph->getContext(), "collective")))
      return graph->emitError("MoE token gather has unsupported policy attributes"), failure();
  auto x = dyn_cast<RankedTensorType>(graph->getOperand(0).getType());
  auto indices = dyn_cast<RankedTensorType>(graph->getOperand(1).getType());
  auto result = dyn_cast<RankedTensorType>(graph->getResult(0).getType());
  if (!x || !indices || !result || !x.hasStaticShape() ||
      !indices.hasStaticShape() || !result.hasStaticShape() ||
      x.getRank() != 2 || indices.getRank() != 1 || result.getRank() != 2 ||
      !x.getElementType().isF32() ||
      !indices.getElementType().isInteger(32) ||
      !result.getElementType().isF32() ||
      x.getDimSize(0) <= 0 || x.getDimSize(1) <= 0 ||
      indices.getDimSize(0) <= 0 ||
      result.getDimSize(0) != indices.getDimSize(0) ||
      result.getDimSize(1) != x.getDimSize(1))
    return graph->emitError("MoE token gather requires f32[T,H], i32[S] -> f32[S,H]"), failure();
  auto names = fn->getAttrOfType<ArrayAttr>("tessera.bindings");
  if (!names || names.size() != 3)
    return graph->emitError("MoE token gather needs three bindings"), failure();
  llvm::SmallDenseSet<StringRef> seen;
  for (Attribute attr : names) {
    auto name = dyn_cast<StringAttr>(attr);
    if (!name || name.getValue().empty() || !seen.insert(name.getValue()).second)
      return graph->emitError("MoE token gather bindings must be unique"), failure();
  }
  OpBuilder builder(graph->getContext());
  auto contract = builder.getDictionaryAttr({
      builder.getNamedAttr("shape", builder.getDenseI64ArrayAttr(
          {x.getDimSize(0), indices.getDimSize(0), x.getDimSize(1)})),
      builder.getNamedAttr("bindings", names),
      builder.getNamedAttr("target", target),
      builder.getNamedAttr("arch", arch),
      builder.getNamedAttr("route", builder.getStringAttr("direct_gather")),
      builder.getNamedAttr("layout", builder.getStringAttr("row_major"))});
  std::string text;
  llvm::raw_string_ostream os(text);
  contract.print(os);
  os.flush();
  return NativeMoeDispatch{fn, contract,
      llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true)};
}

static LogicalResult scheduleNativeMoeDispatch(ModuleOp mod) {
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "tessera.moe_dispatch") ops.push_back(op);
  });
  for (Operation *graph : ops) {
    auto c = moeDispatchContract(graph);
    if (failed(c)) return failure();
    auto ret = dyn_cast<func::ReturnOp>(c->function.getBody().front().back());
    if (c->function.getBody().front().getOperations().size() != 2 || !ret ||
        ret.getOperands() != graph->getResults())
      return graph->emitError("MoE token gather must return its sole result");
    OpBuilder builder(graph);
    builder.setInsertionPointAfter(graph);
    graph->setAttr("schedule.artifact_hash", builder.getStringAttr(c->hash));
    OperationState state(graph->getLoc(), "schedule.moe_dispatch");
    state.addOperands(graph->getResults());
    state.addTypes(graph->getResultTypes());
    state.addAttribute("artifact_hash", builder.getStringAttr(c->hash));
    state.addAttribute("contract", c->contract);
    auto scheduled = builder.create(state);
    graph->getResult(0).replaceAllUsesExcept(scheduled->getResult(0), scheduled);
  }
  return success();
}

static LogicalResult lowerNativeMoeDispatch(ModuleOp mod) {
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "schedule.moe_dispatch") ops.push_back(op);
  });
  for (Operation *scheduled : ops) {
    auto graph = scheduled->getOperand(0).getDefiningOp();
    if (!graph || graph->getName().getStringRef() != "tessera.moe_dispatch")
      return scheduled->emitError("MoE Schedule needs retained Graph producer");
    auto c = moeDispatchContract(graph);
    if (failed(c)) return failure();
    auto hash = scheduled->getAttrOfType<StringAttr>("artifact_hash");
    auto ret = dyn_cast<func::ReturnOp>(c->function.getBody().front().back());
    if (!hash || hash.getValue() != c->hash ||
        graph->getAttr("schedule.artifact_hash") != hash ||
        scheduled->getAttr("contract") != c->contract ||
        scheduled->getAttrs().size() != 2 ||
        scheduled->getResultTypes() != graph->getResultTypes() ||
        !ret || ret.getOperands() != scheduled->getResults() ||
        c->function.getBody().front().getOperations().size() != 3)
      return scheduled->emitError("MoE Schedule contract changed after hashing");
    StringRef entry = "tessera_tile_moe_dispatch_f32_direct";
    if (SymbolTable::lookupSymbolIn(mod, entry))
      return scheduled->emitError("MoE token gather entry collision");
    OpBuilder builder(mod.getContext());
    builder.setInsertionPointToEnd(mod.getBody());
    SmallVector<Type> args(3, LLVM::LLVMPointerType::get(mod.getContext()));
    args.append(3, builder.getI64Type());
    auto fn = LLVM::LLVMFuncOp::create(builder, scheduled->getLoc(), entry,
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(mod.getContext()), args, false));
    fn->setAttr("tessera.schedule_hash", hash);
    fn->setAttr("tessera.native_contract", c->contract);
    auto block = fn.addEntryBlock(builder);
    builder.setInsertionPointToStart(block);
    OperationState kernel(scheduled->getLoc(), "tile.moe_dispatch_kernel");
    kernel.addOperands(block->getArguments());
    kernel.addAttribute("storage", builder.getStringAttr("f32"));
    kernel.addAttribute("index_storage", builder.getStringAttr("i32"));
    builder.create(kernel);
    LLVM::ReturnOp::create(builder, scheduled->getLoc(), ValueRange{});
    c->function.erase();
  }
  return success();
}
} // namespace
