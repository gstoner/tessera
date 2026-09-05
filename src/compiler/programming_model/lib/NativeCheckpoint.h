// Native checkpoint boundaries, owned by GraphToSchedule / ScheduleToTile.
// Included after the common PM dialect/MLIR headers inside namespace tessera.
namespace {
struct NativeCheckpoint {
  func::FuncOp function;
  Operation *graph;
  bool backward;
  SmallVector<int64_t> dims;
  DictionaryAttr contract;
  std::string hash;
};

static FailureOr<NativeCheckpoint> checkpointContract(Operation *graph) {
  bool backward = graph->getName().getStringRef() == "tessera_attn.checkpoint_backward";
  auto fn = graph->getParentOfType<func::FuncOp>();
  auto mod = graph->getParentOfType<ModuleOp>();
  auto target = mod ? mod->getAttrOfType<StringAttr>("tessera.target") : StringAttr();
  auto arch = mod ? mod->getAttrOfType<StringAttr>("tessera.arch") : StringAttr();
  if (!fn || !target || target.getValue() != "nvidia_sm120" ||
      !arch || arch.getValue() != "sm_120" || !llvm::hasSingleElement(fn.getBody()) ||
      fn.getNumArguments() != (backward ? 5 : 3) ||
      fn.getNumResults() != (backward ? 3 : 2) ||
      graph->getNumOperands() != fn.getNumArguments() ||
      graph->getResultTypes() != fn.getResultTypes())
    return graph->emitError("checkpoint requires an isolated SM120 f32 tensor entry"), failure();
  for (unsigned i = 0; i < fn.getNumArguments(); ++i)
    if (graph->getOperand(i) != fn.getArgument(i))
      return graph->emitError("checkpoint operands must preserve function argument order"), failure();
  for (unsigned i = 0; i < fn.getNumArguments(); ++i)
    if (fn.getArgAttr(i, "tessera.layout"))
      return graph->emitError("checkpoint layout overrides are unsupported"), failure();
  SmallVector<RankedTensorType> types;
  for (Type type : fn.getArgumentTypes()) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || tensor.getEncoding() || !tensor.hasStaticShape() || !tensor.getElementType().isF32() ||
        llvm::any_of(tensor.getShape(), [](int64_t d) { return d <= 0; }))
      return graph->emitError("checkpoint requires positive static f32 shapes"), failure();
    types.push_back(tensor);
  }
  unsigned base = backward ? 1 : 0;
  auto q = types[base], k = types[base + 1], v = types[base + 2];
  if (q.getRank() != 4 || k.getRank() != 4 || v.getRank() != 4)
    return graph->emitError("checkpoint Q/K/V must have rank four"), failure();
  int64_t b=q.getDimSize(0), hq=q.getDimSize(1), sq=q.getDimSize(2), d=q.getDimSize(3);
  int64_t hkv=k.getDimSize(1), sk=k.getDimSize(2), dv=v.getDimSize(3);
  auto tensor = [&](ArrayRef<int64_t> shape) { return RankedTensorType::get(shape, q.getElementType()); };
  auto output = tensor({b,hq,sq,dv}), lse = tensor({b,hq,sq});
  if (k != tensor({b,hkv,sk,d}) || v != tensor({b,hkv,sk,dv}) || hq % hkv ||
      (!backward && (fn.getResultTypes()[0] != output || fn.getResultTypes()[1] != lse)) ||
      (backward && (types[0] != output || types[4] != lse ||
                    fn.getResultTypes()[0] != q || fn.getResultTypes()[1] != k || fn.getResultTypes()[2] != v)))
    return graph->emitError("checkpoint shapes or output roles disagree"), failure();
  auto scale = graph->getAttrOfType<FloatAttr>("scale");
  auto causal = graph->getAttrOfType<BoolAttr>("causal");
  if (!scale || !scale.getType().isF32() || !std::isfinite(scale.getValueAsDouble()) ||
      scale.getValueAsDouble() <= 0 || !causal)
    return graph->emitError("checkpoint requires positive finite f32 scale and boolean causal"), failure();
  for (NamedAttribute attr : graph->getAttrs())
    if (attr.getName() != "scale" && attr.getName() != "causal" && attr.getName() != "schedule.artifact_hash")
      return graph->emitError("checkpoint has an unsupported policy attribute"), failure();
  auto args = fn->getAttrOfType<ArrayAttr>("tessera.argument_bindings");
  auto results = fn->getAttrOfType<ArrayAttr>("tessera.result_bindings");
  llvm::SmallDenseSet<StringRef> seen;
  auto validNames = [&](ArrayAttr names, unsigned count) {
    if (!names || names.size() != count) return false;
    for (Attribute attr : names) {
      auto name = dyn_cast<StringAttr>(attr);
      if (!name || name.getValue().empty() || !seen.insert(name.getValue()).second) return false;
    }
    return true;
  };
  if (!validNames(args, fn.getNumArguments()) || !validNames(results, fn.getNumResults()))
    return graph->emitError("checkpoint requires unique argument and result binding names"), failure();
  OpBuilder builder(graph->getContext());
  SmallVector<int64_t> dims{b,hq,hkv,sq,sk,d,dv};
  auto contract = builder.getDictionaryAttr({
      builder.getNamedAttr("family", builder.getStringAttr(backward ? "attention_checkpoint_backward" : "attention_checkpoint_forward")),
      builder.getNamedAttr("shape", builder.getDenseI64ArrayAttr(dims)),
      builder.getNamedAttr("scale", scale), builder.getNamedAttr("causal", causal),
      builder.getNamedAttr("arguments", args), builder.getNamedAttr("results", results),
      builder.getNamedAttr("mask_alignment", builder.getStringAttr("end_aligned_v1")),
      builder.getNamedAttr("target", target), builder.getNamedAttr("arch", arch)});
  std::string text; llvm::raw_string_ostream os(text); contract.print(os); os.flush();
  auto hash = llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
  return NativeCheckpoint{fn,graph,backward,dims,contract,hash};
}

static LogicalResult scheduleNativeCheckpoints(ModuleOp mod) {
  SmallVector<Operation *> graphs;
  mod.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "tessera_attn.checkpoint_forward" ||
        op->getName().getStringRef() == "tessera_attn.checkpoint_backward") graphs.push_back(op);
  });
  for (Operation *graph : graphs) {
    auto c = checkpointContract(graph);
    if (failed(c)) return failure();
    if (c->function.getBody().front().getOperations().size() != 2)
      return graph->emitError("checkpoint entry must contain only its producer and return");
    auto ret = dyn_cast<func::ReturnOp>(c->function.getBody().front().back());
    if (!ret || ret.getOperands() != graph->getResults())
      return graph->emitError("checkpoint return must preserve all result roles");
    OpBuilder builder(graph); builder.setInsertionPointAfter(graph);
    graph->setAttr("schedule.artifact_hash", builder.getStringAttr(c->hash));
    OperationState state(graph->getLoc(), "schedule.attention_checkpoint");
    state.addOperands(graph->getResults()); state.addTypes(graph->getResultTypes());
    state.addAttribute("artifact_hash", builder.getStringAttr(c->hash));
    state.addAttribute("contract", c->contract);
    auto scheduled = builder.create(state);
    for (auto [oldValue,newValue] : llvm::zip(graph->getResults(), scheduled->getResults()))
      oldValue.replaceAllUsesExcept(newValue, scheduled);
  }
  return success();
}

static LogicalResult lowerNativeCheckpoints(ModuleOp mod) {
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) { if (op->getName().getStringRef() == "schedule.attention_checkpoint") ops.push_back(op); });
  for (Operation *scheduled : ops) {
    Operation *graph = scheduled->getNumOperands() ? scheduled->getOperand(0).getDefiningOp() : nullptr;
    if (!graph || (graph->getName().getStringRef() != "tessera_attn.checkpoint_forward" &&
                   graph->getName().getStringRef() != "tessera_attn.checkpoint_backward"))
      return scheduled->emitError("checkpoint schedule requires its retained tensor producer");
    auto c = checkpointContract(graph); if (failed(c)) return failure();
    auto hash = scheduled->getAttrOfType<StringAttr>("artifact_hash");
    if (!hash || hash.getValue() != c->hash || graph->getAttr("schedule.artifact_hash") != hash ||
        scheduled->getAttr("contract") != c->contract || scheduled->getAttrs().size() != 2 ||
        scheduled->getOperands() != graph->getResults() || scheduled->getResultTypes() != graph->getResultTypes() ||
        c->function.getBody().front().getOperations().size() != 3)
      return scheduled->emitError("checkpoint Schedule contract changed after hashing");
    auto ret = dyn_cast<func::ReturnOp>(c->function.getBody().front().back());
    if (!ret || ret.getOperands() != scheduled->getResults())
      return scheduled->emitError("checkpoint Schedule return roles disagree");
    OpBuilder builder(mod.getContext()); auto ptr = LLVM::LLVMPointerType::get(mod.getContext());
    SmallVector<Type> types(c->backward ? 8 : 5, ptr); types.append(7, builder.getI64Type());
    std::string entry = (Twine("tessera_tile_attention_") + (c->backward ? "backward_lse_" : "lse_") + c->hash.substr(0,10)).str();
    if (SymbolTable::lookupSymbolIn(mod, entry)) return scheduled->emitError("checkpoint entry symbol collision");
    builder.setInsertionPointToEnd(mod.getBody());
    auto fn = LLVM::LLVMFuncOp::create(builder, scheduled->getLoc(), entry,
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(mod.getContext()), types, false));
    fn->setAttr("nvvm.kernel", builder.getUnitAttr());
    fn->setAttr("tessera.native_contract", c->contract);
    fn->setAttr("tessera.schedule_hash", hash);
    auto block = fn.addEntryBlock(builder); builder.setInsertionPointToStart(block);
    OperationState kernel(scheduled->getLoc(), c->backward ? "tile.attention_backward_kernel" : "tile.attention_kernel");
    kernel.addOperands(block->getArguments());
    kernel.addAttribute("storage",builder.getStringAttr("f32"));
    kernel.addAttribute("accum",builder.getStringAttr("f32"));
    kernel.addAttribute("scale",graph->getAttr("scale")); kernel.addAttribute("causal",graph->getAttr("causal"));
    kernel.addAttribute("bias",builder.getBoolAttr(false));
    kernel.addAttribute("window_left",builder.getI64IntegerAttr(-1)); kernel.addAttribute("window_right",builder.getI64IntegerAttr(-1));
    kernel.addAttribute("softcap",builder.getF32FloatAttr(0)); kernel.addAttribute("dropout_p",builder.getF32FloatAttr(0));
    kernel.addAttribute("dropout_seed",builder.getI64IntegerAttr(0));
    kernel.addAttribute("lse_checkpoint",builder.getStringAttr("saved"));
    if (c->backward) {
      kernel.addAttribute("route",builder.getStringAttr("deterministic_direct"));
      kernel.addAttribute("deterministic",builder.getBoolAttr(true));
      kernel.addAttribute("workspace_bytes",builder.getI64IntegerAttr(0));
      kernel.addAttribute("workspace_owner",builder.getStringAttr("output_element"));
    }
    builder.create(kernel); LLVM::ReturnOp::create(builder, scheduled->getLoc(), ValueRange{});
    c->function.erase();
  }
  return success();
}
} // namespace
