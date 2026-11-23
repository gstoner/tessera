// AR-DECODE-PROGRAM-1 S1b: native row argmax/argmin on sm_120. Included
// inside namespace tessera.
//
// One isolated `tessera.argmax|argmin` over an f32 tensor in a module that
// names `tessera.target = "nvidia_sm120"` -> a content-addressed
// `schedule.argreduce` -> the shared `tile.argreduce_kernel` launch envelope
// (source, i32 destination, rows, columns), which the sm_120 consumer
// materializes as one 128-thread CTA per row with a warp-shuffle combine.
//
// The reduction is over the last axis, or over the flattened tensor when the
// Graph op has no axis (NumPy `axis=None`). Ties take the first occurrence and
// a NaN wins over every number (the first NaN), which is NumPy's `argmax`
// semantics. Ownership is by target: every other module passes through
// untouched (x86 has its own contract in NativeX86Kernel.h); inside an sm_120
// module anything outside this envelope fails closed (Decision #21a).
namespace {
struct NativeArgReduce {
  func::FuncOp function;
  DictionaryAttr contract;
  std::string hash;
  int64_t rows = 0;
  int64_t columns = 0;
};

static bool ownsNativeArgReduce(Operation *graph) {
  auto mod = graph->getParentOfType<ModuleOp>();
  auto target = mod ? mod->getAttrOfType<StringAttr>("tessera.target") : StringAttr();
  // A tensor-program member is refused by that program's own admission check.
  return target && target.getValue() == "nvidia_sm120" &&
         !mod->hasAttr("tessera.native.tensor_program_member");
}

static FailureOr<NativeArgReduce> argReduceContract(Operation *graph) {
  auto fn = graph->getParentOfType<func::FuncOp>();
  auto mod = graph->getParentOfType<ModuleOp>();
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch = mod->getAttrOfType<StringAttr>("tessera.arch");
  StringRef name = graph->getName().getStringRef();
  if (!arch || arch.getValue() != "sm_120")
    return graph->emitError("sm_120 argreduce requires tessera.arch = \"sm_120\""), failure();
  if (!fn || !llvm::hasSingleElement(fn.getBody()) || fn.getNumArguments() != 1 ||
      fn.getNumResults() != 1 || graph->getNumOperands() != 1 ||
      graph->getNumResults() != 1 || graph->getOperand(0) != fn.getArgument(0) ||
      graph->getResultTypes() != fn.getResultTypes())
    return graph->emitError("sm_120 argreduce requires an isolated one-argument tensor entry"), failure();
  if (Attribute raw = fn.getArgAttr(0, "tessera.layout")) {
    auto layout = dyn_cast<StringAttr>(raw);
    if (!layout || layout.getValue() != "row_major")
      return graph->emitError("sm_120 argreduce requires a row_major operand"), failure();
  }
  for (NamedAttribute attr : graph->getAttrs()) {
    StringRef key = attr.getName().getValue();
    if (key == "axis" || key == "keepdims" || key == "schedule.artifact_hash") continue;
    if (key == "tessera.effect_kind" &&
        attr.getValue() == StringAttr::get(graph->getContext(), "pure")) continue;
    return graph->emitError("sm_120 argreduce has an unsupported policy attribute '")
               << key << "'", failure();
  }
  auto input = dyn_cast<RankedTensorType>(graph->getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(graph->getResult(0).getType());
  if (!input || !input.hasStaticShape() || input.getRank() < 1 ||
      !input.getElementType().isF32())
    return graph->emitError("sm_120 argreduce requires a static f32 operand of rank >= 1"), failure();
  if (!result || !result.hasStaticShape() || !result.getElementType().isInteger(32))
    return graph->emitError("sm_120 argreduce result must be a static i32 index tensor"), failure();
  auto axisAttr = graph->getAttrOfType<IntegerAttr>("axis");
  if (graph->hasAttr("axis") && !axisAttr)
    return graph->emitError("sm_120 argreduce axis must be an integer"), failure();
  bool flatten = !axisAttr;
  if (axisAttr && axisAttr.getInt() != -1 && axisAttr.getInt() != input.getRank() - 1)
    return graph->emitError("sm_120 argreduce reduces only the last axis"), failure();
  auto keepAttr = graph->getAttrOfType<BoolAttr>("keepdims");
  if (graph->hasAttr("keepdims") && !keepAttr)
    return graph->emitError("sm_120 argreduce keepdims must be a boolean"), failure();
  int64_t elements = 1;
  for (int64_t dim : input.getShape()) {
    if (dim <= 0 || elements > INT32_MAX / dim)
      return graph->emitError("sm_120 argreduce shape exceeds its i32 index ABI"), failure();
    elements *= dim;
  }
  int64_t columns = flatten ? elements : input.getShape().back();
  int64_t rows = elements / columns;
  if (result.getNumElements() != rows)
    return graph->emitError("sm_120 argreduce result must hold one index per row"), failure();
  auto names = fn->getAttrOfType<ArrayAttr>("tessera.bindings");
  if (!names || names.size() != 2)
    return graph->emitError("sm_120 argreduce requires two binding names"), failure();
  llvm::SmallDenseSet<StringRef> seen;
  for (Attribute attr : names) {
    auto binding = dyn_cast<StringAttr>(attr);
    if (!binding || binding.getValue().empty() || !seen.insert(binding.getValue()).second)
      return graph->emitError("sm_120 argreduce binding names must be unique"), failure();
  }
  OpBuilder builder(graph->getContext());
  auto contract = builder.getDictionaryAttr({
      builder.getNamedAttr("kind", builder.getStringAttr(name == "tessera.argmax" ? "argmax" : "argmin")),
      builder.getNamedAttr("shape", builder.getDenseI64ArrayAttr({rows, columns})),
      builder.getNamedAttr("logical_shape", builder.getDenseI64ArrayAttr(input.getShape())),
      builder.getNamedAttr("result_shape", builder.getDenseI64ArrayAttr(result.getShape())),
      builder.getNamedAttr("axis_mode", builder.getStringAttr(flatten ? "flatten" : "last")),
      builder.getNamedAttr("keepdims", builder.getBoolAttr(keepAttr && keepAttr.getValue())),
      builder.getNamedAttr("bindings", names),
      builder.getNamedAttr("target", target), builder.getNamedAttr("arch", arch),
      builder.getNamedAttr("storage", builder.getStringAttr("f32")),
      builder.getNamedAttr("output_storage", builder.getStringAttr("i32")),
      builder.getNamedAttr("tie_break", builder.getStringAttr("first")),
      builder.getNamedAttr("nan_mode", builder.getStringAttr("first_nan")),
      builder.getNamedAttr("schedule", builder.getStringAttr("cta_128_warp_shuffle"))});
  std::string text; llvm::raw_string_ostream os(text); contract.print(os); os.flush();
  return NativeArgReduce{fn, contract,
      llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true), rows, columns};
}

static bool isArgReduceGraphOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "tessera.argmax" || name == "tessera.argmin";
}

static LogicalResult scheduleNativeArgReduce(ModuleOp mod) {
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) {
    if (isArgReduceGraphOp(op) && ownsNativeArgReduce(op)) ops.push_back(op);
  });
  for (Operation *graph : ops) {
    auto c = argReduceContract(graph); if (failed(c)) return failure();
    auto ret = dyn_cast<func::ReturnOp>(c->function.getBody().front().back());
    if (c->function.getBody().front().getOperations().size() != 2 || !ret ||
        ret.getOperands() != graph->getResults())
      return graph->emitError("sm_120 argreduce must return its sole index result");
    OpBuilder builder(graph); builder.setInsertionPointAfter(graph);
    graph->setAttr("schedule.artifact_hash", builder.getStringAttr(c->hash));
    OperationState state(graph->getLoc(), "schedule.argreduce");
    state.addOperands(graph->getResults()); state.addTypes(graph->getResultTypes());
    state.addAttribute("artifact_hash", builder.getStringAttr(c->hash));
    state.addAttribute("contract", c->contract);
    auto scheduled = builder.create(state);
    graph->getResult(0).replaceAllUsesExcept(scheduled->getResult(0), scheduled);
  }
  return success();
}

static LogicalResult lowerNativeArgReduce(ModuleOp mod) {
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "schedule.argreduce") ops.push_back(op);
  });
  for (Operation *scheduled : ops) {
    Operation *graph = scheduled->getOperand(0).getDefiningOp();
    if (!graph || !isArgReduceGraphOp(graph) || !ownsNativeArgReduce(graph))
      return scheduled->emitError("argreduce schedule requires its retained sm_120 Graph producer");
    auto c = argReduceContract(graph); if (failed(c)) return failure();
    auto hash = scheduled->getAttrOfType<StringAttr>("artifact_hash");
    auto ret = dyn_cast<func::ReturnOp>(c->function.getBody().front().back());
    if (!hash || hash.getValue() != c->hash || graph->getAttr("schedule.artifact_hash") != hash ||
        scheduled->getAttr("contract") != c->contract || scheduled->getAttrs().size() != 2 ||
        scheduled->getResultTypes() != graph->getResultTypes() || !ret ||
        ret.getOperands() != scheduled->getResults() ||
        c->function.getBody().front().getOperations().size() != 3)
      return scheduled->emitError("argreduce Schedule contract changed after hashing");
    StringRef kind = c->contract.getAs<StringAttr>("kind").getValue();
    std::string entry = (Twine("tessera_tile_argreduce_") + kind + "_f32_" +
                         hash.getValue().take_front(10)).str();
    if (SymbolTable::lookupSymbolIn(mod, entry))
      return scheduled->emitError("argreduce entry collision");
    OpBuilder builder(mod.getContext()); builder.setInsertionPointToEnd(mod.getBody());
    SmallVector<Type> args(2, LLVM::LLVMPointerType::get(mod.getContext()));
    args.append(2, builder.getI64Type());
    auto fn = LLVM::LLVMFuncOp::create(builder, scheduled->getLoc(), entry,
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(mod.getContext()), args, false));
    fn->setAttr("nvvm.kernel", builder.getUnitAttr());
    fn->setAttr("nvvm.reqntid", builder.getDenseI32ArrayAttr({128, 1, 1}));
    fn->setAttr("tessera.schedule_hash", hash);
    fn->setAttr("tessera.native_contract", c->contract);
    auto block = fn.addEntryBlock(builder); builder.setInsertionPointToStart(block);
    OperationState kernel(scheduled->getLoc(), "tile.argreduce_kernel");
    kernel.addOperands(block->getArguments());
    for (StringRef key : {"kind", "storage", "output_storage", "tie_break", "nan_mode", "schedule"})
      kernel.addAttribute(key, c->contract.get(key));
    builder.create(kernel);
    LLVM::ReturnOp::create(builder, scheduled->getLoc(), ValueRange{});
    c->function.erase();
  }
  return success();
}
} // namespace
