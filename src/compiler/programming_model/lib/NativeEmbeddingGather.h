// AR-DECODE-PROGRAM-1 S1c: native embedding row gather on sm_120. Included
// inside namespace tessera.
//
// One isolated `tessera.gather %table, %tokens {axis = 0}` over an f32[V, H]
// table and i32[B] tokens in a module that names
// `tessera.target = "nvidia_sm120"` -> a content-addressed
// `schedule.embedding_gather` -> `tile.embedding_gather_kernel` (table,
// tokens, out, status, V, H, B), which the sm_120 consumer materializes as one
// 128-thread CTA per token row.
//
// Out-of-range tokens are a semantic choice, so the contract states it
// (Decision #21a): the kernel checks every token against [0, V) on the device,
// leaves that row unwritten and sets status[0] = 1, and the launcher then
// refuses the whole call and copies nothing back. NumPy's `take` would raise;
// a negative token is never wrapped. Ownership is by target: every other
// module passes through untouched (x86 and Apple have their own gather
// contracts); inside an sm_120 module anything outside this envelope fails
// closed.
namespace {
struct NativeEmbeddingGather {
  func::FuncOp function;
  DictionaryAttr contract;
  std::string hash;
};

static bool ownsNativeEmbeddingGather(Operation *graph) {
  if (graph->getName().getStringRef() != "tessera.gather") return false;
  auto mod = graph->getParentOfType<ModuleOp>();
  auto target = mod ? mod->getAttrOfType<StringAttr>("tessera.target") : StringAttr();
  // A tensor-program member is refused by that program's own admission check.
  return target && target.getValue() == "nvidia_sm120" &&
         !mod->hasAttr("tessera.native.tensor_program_member");
}

static FailureOr<NativeEmbeddingGather> embeddingGatherContract(Operation *graph) {
  auto fn = graph->getParentOfType<func::FuncOp>();
  auto mod = graph->getParentOfType<ModuleOp>();
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch = mod->getAttrOfType<StringAttr>("tessera.arch");
  if (!arch || arch.getValue() != "sm_120")
    return graph->emitError("sm_120 embedding gather requires tessera.arch = \"sm_120\""), failure();
  if (!fn || !llvm::hasSingleElement(fn.getBody()) || fn.getNumArguments() != 2 ||
      fn.getNumResults() != 1 || graph->getNumOperands() != 2 ||
      graph->getNumResults() != 1 || graph->getOperand(0) != fn.getArgument(0) ||
      graph->getOperand(1) != fn.getArgument(1) ||
      graph->getResultTypes() != fn.getResultTypes())
    return graph->emitError("sm_120 embedding gather requires an isolated (table, tokens) entry"), failure();
  for (unsigned i = 0; i < 2; ++i)
    if (Attribute raw = fn.getArgAttr(i, "tessera.layout")) {
      auto layout = dyn_cast<StringAttr>(raw);
      if (!layout || layout.getValue() != "row_major")
        return graph->emitError("sm_120 embedding gather requires row_major operands"), failure();
    }
  for (NamedAttribute attr : graph->getAttrs()) {
    StringRef key = attr.getName().getValue();
    if (key == "axis" || key == "schedule.artifact_hash") continue;
    if (key == "tessera.effect_kind" &&
        attr.getValue() == StringAttr::get(graph->getContext(), "pure")) continue;
    return graph->emitError("sm_120 embedding gather has an unsupported policy attribute '")
               << key << "'", failure();
  }
  // The ODS default is 0; an explicit axis must name the table's row axis.
  auto axis = graph->getAttrOfType<IntegerAttr>("axis");
  if (graph->hasAttr("axis") && (!axis || (axis.getInt() != 0 && axis.getInt() != -2)))
    return graph->emitError("sm_120 embedding gather gathers only table rows (axis 0)"), failure();
  auto table = dyn_cast<RankedTensorType>(graph->getOperand(0).getType());
  auto tokens = dyn_cast<RankedTensorType>(graph->getOperand(1).getType());
  auto result = dyn_cast<RankedTensorType>(graph->getResult(0).getType());
  if (!table || !table.hasStaticShape() || table.getRank() != 2 ||
      !table.getElementType().isF32())
    return graph->emitError("sm_120 embedding gather requires a static f32[V, H] table"), failure();
  if (!tokens || !tokens.hasStaticShape() || tokens.getRank() != 1 ||
      !tokens.getElementType().isInteger(32))
    return graph->emitError("sm_120 embedding gather requires static i32[B] tokens"), failure();
  int64_t vocab = table.getDimSize(0), hidden = table.getDimSize(1), batch = tokens.getDimSize(0);
  if (!result || !result.hasStaticShape() || !result.getElementType().isF32() ||
      result.getShape() != ArrayRef<int64_t>({batch, hidden}))
    return graph->emitError("sm_120 embedding gather result must be f32[B, H]"), failure();
  // One CTA per token row: B is the grid's x extent, and a token is an i32.
  if (vocab <= 0 || hidden <= 0 || batch <= 0 || vocab > INT32_MAX ||
      batch > INT32_MAX || hidden > INT64_MAX / vocab || hidden > INT64_MAX / batch)
    return graph->emitError("sm_120 embedding gather extents exceed its launch ABI"), failure();
  auto names = fn->getAttrOfType<ArrayAttr>("tessera.bindings");
  if (!names || names.size() != 3)
    return graph->emitError("sm_120 embedding gather requires three binding names"), failure();
  llvm::SmallDenseSet<StringRef> seen;
  for (Attribute attr : names) {
    auto binding = dyn_cast<StringAttr>(attr);
    // `status` is the launch-owned refusal word, not a caller tensor.
    if (!binding || binding.getValue().empty() || binding.getValue() == "status" ||
        !seen.insert(binding.getValue()).second)
      return graph->emitError("sm_120 embedding gather binding names must be unique and not 'status'"), failure();
  }
  OpBuilder builder(graph->getContext());
  auto contract = builder.getDictionaryAttr({
      builder.getNamedAttr("shape", builder.getDenseI64ArrayAttr({vocab, hidden, batch})),
      builder.getNamedAttr("bindings", names),
      builder.getNamedAttr("target", target), builder.getNamedAttr("arch", arch),
      builder.getNamedAttr("storage", builder.getStringAttr("f32")),
      builder.getNamedAttr("index_storage", builder.getStringAttr("i32")),
      builder.getNamedAttr("index_bounds", builder.getStringAttr("kernel_checked_status_word")),
      builder.getNamedAttr("schedule", builder.getStringAttr("cta_128_per_token_row"))});
  std::string text; llvm::raw_string_ostream os(text); contract.print(os); os.flush();
  return NativeEmbeddingGather{fn, contract,
      llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true)};
}

static LogicalResult scheduleNativeEmbeddingGather(ModuleOp mod) {
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) { if (ownsNativeEmbeddingGather(op)) ops.push_back(op); });
  for (Operation *graph : ops) {
    auto c = embeddingGatherContract(graph); if (failed(c)) return failure();
    auto ret = dyn_cast<func::ReturnOp>(c->function.getBody().front().back());
    if (c->function.getBody().front().getOperations().size() != 2 || !ret ||
        ret.getOperands() != graph->getResults())
      return graph->emitError("sm_120 embedding gather must return its sole gathered rows");
    OpBuilder builder(graph); builder.setInsertionPointAfter(graph);
    graph->setAttr("schedule.artifact_hash", builder.getStringAttr(c->hash));
    OperationState state(graph->getLoc(), "schedule.embedding_gather");
    state.addOperands(graph->getResults()); state.addTypes(graph->getResultTypes());
    state.addAttribute("artifact_hash", builder.getStringAttr(c->hash));
    state.addAttribute("contract", c->contract);
    auto scheduled = builder.create(state);
    graph->getResult(0).replaceAllUsesExcept(scheduled->getResult(0), scheduled);
  }
  return success();
}

static LogicalResult lowerNativeEmbeddingGather(ModuleOp mod) {
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "schedule.embedding_gather") ops.push_back(op);
  });
  for (Operation *scheduled : ops) {
    Operation *graph = scheduled->getOperand(0).getDefiningOp();
    if (!graph || !ownsNativeEmbeddingGather(graph))
      return scheduled->emitError("embedding gather schedule requires its retained sm_120 Graph producer");
    auto c = embeddingGatherContract(graph); if (failed(c)) return failure();
    auto hash = scheduled->getAttrOfType<StringAttr>("artifact_hash");
    auto ret = dyn_cast<func::ReturnOp>(c->function.getBody().front().back());
    if (!hash || hash.getValue() != c->hash || graph->getAttr("schedule.artifact_hash") != hash ||
        scheduled->getAttr("contract") != c->contract || scheduled->getAttrs().size() != 2 ||
        scheduled->getResultTypes() != graph->getResultTypes() || !ret ||
        ret.getOperands() != scheduled->getResults() ||
        c->function.getBody().front().getOperations().size() != 3)
      return scheduled->emitError("embedding gather Schedule contract changed after hashing");
    std::string entry = (Twine("tessera_tile_embedding_gather_f32_") +
                         hash.getValue().take_front(10)).str();
    if (SymbolTable::lookupSymbolIn(mod, entry))
      return scheduled->emitError("embedding gather entry collision");
    OpBuilder builder(mod.getContext()); builder.setInsertionPointToEnd(mod.getBody());
    SmallVector<Type> args(4, LLVM::LLVMPointerType::get(mod.getContext()));
    args.append(3, builder.getI64Type());
    auto fn = LLVM::LLVMFuncOp::create(builder, scheduled->getLoc(), entry,
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(mod.getContext()), args, false));
    fn->setAttr("nvvm.kernel", builder.getUnitAttr());
    fn->setAttr("nvvm.reqntid", builder.getDenseI32ArrayAttr({128, 1, 1}));
    fn->setAttr("tessera.schedule_hash", hash);
    fn->setAttr("tessera.native_contract", c->contract);
    auto block = fn.addEntryBlock(builder); builder.setInsertionPointToStart(block);
    OperationState kernel(scheduled->getLoc(), "tile.embedding_gather_kernel");
    kernel.addOperands(block->getArguments());
    for (StringRef key : {"storage", "index_storage", "index_bounds", "schedule"})
      kernel.addAttribute(key, c->contract.get(key));
    builder.create(kernel);
    LLVM::ReturnOp::create(builder, scheduled->getLoc(), ValueRange{});
    c->function.erase();
  }
  return success();
}
} // namespace
