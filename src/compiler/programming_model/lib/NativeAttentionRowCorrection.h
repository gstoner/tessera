// Native AD tensor residual -> sealed Schedule -> typed launch-level Tile.
// Included inside namespace tessera after the common PM/MLIR headers.
namespace {
struct NativeAttentionRowCorrection {
  func::FuncOp function;
  Operation *graph;
  DictionaryAttr contract;
  std::string hash;
  bool backward;
};

static FailureOr<NativeAttentionRowCorrection>
attentionRowCorrectionContract(Operation *graph) {
  bool backward = graph->getName().getStringRef() == "tessera_attn.checkpoint_backward_rows";
  if ((!backward && graph->getName().getStringRef() != "tessera_attn.checkpoint_row_correction") ||
      failed(mlir::verify(graph)))
    return failure();
  auto fn = graph->getParentOfType<func::FuncOp>();
  auto mod = graph->getParentOfType<ModuleOp>();
  auto target = mod ? mod->getAttrOfType<StringAttr>("tessera.target") : StringAttr();
  auto arch = mod ? mod->getAttrOfType<StringAttr>("tessera.arch") : StringAttr();
  if (!fn || !target || target.getValue() != "nvidia_sm120" || !arch ||
      arch.getValue() != "sm_120" || !llvm::hasSingleElement(fn.getBody()) ||
      fn.getNumArguments() != graph->getNumOperands() ||
      !llvm::equal(fn.getResultTypes(), graph->getResultTypes()))
    return graph->emitError("row correction requires one isolated SM120 tensor entry"), failure();
  SmallVector<unsigned> roles;
  llvm::SmallDenseSet<unsigned> distinct;
  for (Value operand : graph->getOperands()) {
    auto arg = dyn_cast<BlockArgument>(operand);
    if (!arg || arg.getOwner() != &fn.getBody().front() ||
        !distinct.insert(arg.getArgNumber()).second)
      return graph->emitError("row correction inputs require distinct direct entry arguments"), failure();
    roles.push_back(arg.getArgNumber());
  }
  for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
    if (auto attr = fn.getArgAttr(i, "tessera.layout")) {
      auto layout = dyn_cast<StringAttr>(attr);
      if (!layout || layout.getValue() != "row_major")
        return graph->emitError("row correction Schedule currently requires compact row-major argument layouts"), failure();
    }
  }
  // A physical launch may only index byte spans representable by its signed
  // i64 address arithmetic. Check every role before outlining opaque pointers.
  auto span = [&](RankedTensorType type) -> FailureOr<int64_t> {
    int64_t bytes = type.getElementTypeBitWidth() / 8;
    for (int64_t dim : type.getShape()) {
      if (dim <= 0 || bytes > std::numeric_limits<int64_t>::max() / dim)
        return failure();
      bytes *= dim;
    }
    return bytes;
  };
  for (Type type : graph->getOperandTypes())
    if (failed(span(cast<RankedTensorType>(type))))
      return graph->emitError("row correction physical input byte span overflows i64"), failure();
  bool bias = graph->getAttrOfType<BoolAttr>("bias").getValue();
  auto rowBytes = span(cast<RankedTensorType>(graph->getOperand(4 + unsigned(bias)).getType()));
  if (failed(rowBytes) || *rowBytes / 4 > int64_t(INT32_MAX) * 32)
    return graph->emitError("row correction result exceeds byte or launch-grid capacity"), failure();
  auto q = cast<RankedTensorType>(graph->getOperand(1).getType());
  auto k = cast<RankedTensorType>(graph->getOperand(2).getType());
  auto v = cast<RankedTensorType>(graph->getOperand(3).getType());
  auto dout = cast<RankedTensorType>(graph->getOperand(0).getType());
  SmallVector<int64_t> dims{q.getDimSize(0), q.getDimSize(1), k.getDimSize(1),
      q.getDimSize(2), k.getDimSize(2), q.getDimSize(3), v.getDimSize(3)};
  auto rawNames = fn->getAttrOfType<ArrayAttr>("tessera.argument_bindings");
  auto results = fn->getAttrOfType<ArrayAttr>("tessera.result_bindings");
  if (!rawNames || rawNames.size() != roles.size() || !results || results.size() != graph->getNumResults())
    return graph->emitError("row correction requires complete argument/result binding names"), failure();
  SmallVector<Attribute> names;
  llvm::SmallDenseSet<StringRef> seen;
  auto addName = [&](Attribute attr) {
    auto name = dyn_cast<StringAttr>(attr);
    return name && !name.getValue().empty() && seen.insert(name.getValue()).second;
  };
  for (unsigned role : roles) {
    if (!addName(rawNames[role]))
      return graph->emitError("row correction binding names must be unique and nonempty"), failure();
    names.push_back(rawNames[role]);
  }
  for (Attribute result : results)
    if (!addName(result))
      return graph->emitError("row correction result bindings must be distinct"), failure();
  OpBuilder b(graph->getContext());
  NamedAttrList fields;
  fields.set("family", b.getStringAttr(backward ? "attention_backward_rows" : "attention_row_correction"));
  fields.set("shape", b.getDenseI64ArrayAttr(dims));
  fields.set("storage", b.getStringAttr(checkpointStorageSpelling(q.getElementType())));
  fields.set("output_storage", b.getStringAttr(checkpointStorageSpelling(dout.getElementType())));
  fields.set("row_storage", b.getStringAttr("f32"));
  fields.set("accum", b.getStringAttr("f32"));
  fields.set("row_correction", b.getStringAttr("probability_dot_v_f32_v1"));
  fields.set("mask_alignment", b.getStringAttr("end_aligned_v1"));
  fields.set("row_bytes", b.getI64IntegerAttr(*rowBytes));
  fields.set("block_threads", b.getI64IntegerAttr(backward ? 128 : 32));
  fields.set("arguments", b.getArrayAttr(names));
  fields.set("results", results);
  fields.set("target", target); fields.set("arch", arch);
  for (StringRef name : {"scale", "causal", "bias", "lse_cotangent"})
    fields.set(name, graph->getAttr(name));
  if (bias)
    fields.set("bias_shape", b.getDenseI64ArrayAttr(
        cast<RankedTensorType>(graph->getOperand(4).getType()).getShape()));
  if (auto mappingAttr = mod->getAttr("tessera.attention_argument_indices")) {
    auto mapping = dyn_cast<DenseI64ArrayAttr>(mappingAttr);
    llvm::SmallDenseSet<int64_t> unique;
    if (!mapping || mapping.size() != 3 + unsigned(bias))
      return graph->emitError("row correction frontend mapping must cover all primal roles"), failure();
    for (int64_t index : mapping.asArrayRef())
      if (index < 0 || index >= mapping.size() || !unique.insert(index).second)
        return graph->emitError("row correction frontend mapping must be a permutation"), failure();
    fields.set("frontend_argument_indices", mapping);
  }
  if (auto activityAttr = fn->getAttr(backward ? "tessera.checkpoint_gradient_activity" : "tessera.row_correction_gradient_activity")) {
    auto activity = dyn_cast<DenseI64ArrayAttr>(activityAttr);
    if (!activity || activity.size() != 3 + unsigned(bias) ||
        llvm::any_of(activity.asArrayRef(), [](int64_t x) { return x != 0 && x != 1; }) ||
        llvm::none_of(activity.asArrayRef(), [](int64_t x) { return x == 1; }))
      return graph->emitError("row correction gradient activity must retain nonempty binary primal roles"), failure();
    fields.set("gradient_activity", activity);
  }
  if (backward) {
    fields.set("row_correction_usage", b.getStringAttr("external_input_v1"));
    fields.set("bias_gradient", b.getBoolAttr(bias));
    if (auto compactAttr = fn->getAttr("tessera.checkpoint_gradient_output")) {
      auto compact = dyn_cast<StringAttr>(compactAttr);
      auto launch = fn->getAttrOfType<StringAttr>("tessera.checkpoint_gradient_launch");
      auto threads = fn->getAttrOfType<IntegerAttr>("tessera.checkpoint_gradient_threads");
      if (!compact || compact.getValue() != "compact_v1" || !fields.get("gradient_activity") ||
          !launch || (launch.getValue() != "packed_v1" && launch.getValue() != "logical_v1") ||
          !threads || !threads.getType().isInteger(64) || (threads.getInt() != 64 && threads.getInt() != 128))
        return graph->emitError("row-corrected backward compact outputs require activity, launch policy and 64/128 threads"), failure();
      fields.set("gradient_output", compact); fields.set("gradient_launch", launch);
      fields.set("block_threads", threads);
    } else if (fn->hasAttr("tessera.checkpoint_gradient_launch") || fn->hasAttr("tessera.checkpoint_gradient_threads")) {
      return graph->emitError("row-corrected backward launch overrides require compact output"), failure();
    }
    int64_t count = 0;
    for (Type type : graph->getResultTypes()) {
      auto bytes = span(cast<RankedTensorType>(type));
      if (failed(bytes))
        return graph->emitError("row-corrected backward result byte span overflows i64"), failure();
      int64_t elements = *bytes / (cast<RankedTensorType>(type).getElementTypeBitWidth() / 8);
      if (elements > int64_t(INT32_MAX) * cast<IntegerAttr>(fields.get("block_threads")).getInt() - count)
        return graph->emitError("row-corrected backward output launch exceeds grid capacity"), failure();
      count += elements;
    }
  }
  if (auto pairAttr = mod->getAttr("tessera.attention_ad_pair")) {
    auto pair = dyn_cast<StringAttr>(pairAttr);
    if (!pair || pair.getValue().empty())
      return graph->emitError("row correction paired lineage must be nonempty text"), failure();
    fields.set("source_ad_pair_digest", b.getStringAttr(llvm::toHex(
        llvm::SHA256::hash(llvm::arrayRefFromStringRef(pair.getValue())), true)));
  }
  auto contract = b.getDictionaryAttr(fields);
  std::string text; llvm::raw_string_ostream os(text); contract.print(os); os.flush();
  auto hash = llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
  return NativeAttentionRowCorrection{fn, graph, contract, hash, backward};
}

static LogicalResult scheduleNativeRowCorrections(ModuleOp mod, bool &selected) {
  SmallVector<Operation *> graphs;
  mod.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "tessera_attn.checkpoint_row_correction" ||
        op->getName().getStringRef() == "tessera_attn.checkpoint_backward_rows") graphs.push_back(op);
  });
  selected = !graphs.empty();
  if (graphs.empty()) return success();
  if (graphs.size() != 1 || mod.getBody()->getOperations().size() != 1)
    return mod.emitError("row correction Schedule requires one isolated product");
  auto c = attentionRowCorrectionContract(graphs[0]);
  if (failed(c)) return failure();
  auto ret = dyn_cast<func::ReturnOp>(c->function.getBody().front().back());
  if (c->function.getBody().front().getOperations().size() != 2 || !ret ||
      ret.getOperands() != c->graph->getResults())
    return c->graph->emitError("row correction Schedule must retain the complete tensor result");
  OpBuilder b(c->graph); b.setInsertionPointAfter(c->graph);
  if (auto prior = c->graph->getAttr("schedule.artifact_hash"))
    if (prior != b.getStringAttr(c->hash))
      return c->graph->emitError("row correction retained Graph has a different Schedule hash");
  c->graph->setAttr("schedule.artifact_hash", b.getStringAttr(c->hash));
  OperationState state(c->graph->getLoc(), c->backward ? "schedule.attention_backward_rows" : "schedule.attention_row_correction");
  state.addOperands(c->graph->getResults()); state.addTypes(c->graph->getResultTypes());
  state.addAttribute("artifact_hash", b.getStringAttr(c->hash));
  state.addAttribute("contract", c->contract);
  auto *scheduled = b.create(state);
  for (auto [source, target] : llvm::zip(c->graph->getResults(), scheduled->getResults()))
    source.replaceAllUsesExcept(target, scheduled);
  return success();
}

static LogicalResult lowerNativeRowCorrections(ModuleOp mod, bool &selected) {
  SmallVector<Operation *> schedules;
  mod.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "schedule.attention_row_correction" ||
        op->getName().getStringRef() == "schedule.attention_backward_rows") schedules.push_back(op);
  });
  selected = !schedules.empty();
  if (selected && (schedules.size() != 1 || mod.getBody()->getOperations().size() != 1))
    return mod.emitError("row correction Tile lowering requires one isolated product");
  for (Operation *scheduled : schedules) {
    if (scheduled->getNumOperands() == 0 || scheduled->getNumOperands() != scheduled->getNumResults())
      return scheduled->emitError("row correction Schedule requires one tensor result");
    auto *graph = scheduled->getOperand(0).getDefiningOp();
    if (!graph || (graph->getName().getStringRef() != "tessera_attn.checkpoint_row_correction" &&
                   graph->getName().getStringRef() != "tessera_attn.checkpoint_backward_rows"))
      return scheduled->emitError("row correction Schedule lost its tensor producer");
    auto c = attentionRowCorrectionContract(graph); if (failed(c)) return failure();
    auto hash = scheduled->getAttrOfType<StringAttr>("artifact_hash");
    auto ret = dyn_cast<func::ReturnOp>(c->function.getBody().front().back());
    StringRef expectedName = c->backward ? "schedule.attention_backward_rows" : "schedule.attention_row_correction";
    if (scheduled->getName().getStringRef() != expectedName ||
        !llvm::equal(scheduled->getOperands(), graph->getResults()) ||
        !hash || hash.getValue() != c->hash || graph->getAttr("schedule.artifact_hash") != hash ||
        scheduled->getAttr("contract") != c->contract || scheduled->getAttrs().size() != 2 ||
        scheduled->getResultTypes() != graph->getResultTypes() ||
        c->function.getBody().front().getOperations().size() != 3 || !ret ||
        ret.getOperands() != scheduled->getResults())
      return scheduled->emitError("row correction Schedule contract changed after hashing");
    OpBuilder b(mod.getContext()); b.setInsertionPointToEnd(mod.getBody());
    auto ptr = LLVM::LLVMPointerType::get(mod.getContext());
    unsigned outputs = graph->getNumResults();
    if (c->backward && c->contract.get("gradient_output")) {
      outputs = 0;
      for (int64_t bit : cast<DenseI64ArrayAttr>(c->contract.get("gradient_activity")).asArrayRef()) outputs += bit;
    }
    SmallVector<Type> types(graph->getNumOperands() + outputs, ptr);
    types.append(7, b.getI64Type());
    std::string entry = std::string(c->backward ? "tessera_tile_attention_backward_rows_" : "tessera_tile_attention_row_correction_") + c->hash.substr(0, 10);
    if (SymbolTable::lookupSymbolIn(mod, entry))
      return scheduled->emitError("row correction entry symbol collision");
    auto fn = LLVM::LLVMFuncOp::create(b, graph->getLoc(), entry,
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(mod.getContext()), types, false));
    fn->setAttr("nvvm.kernel", b.getUnitAttr());
    fn->setAttr("tessera.native_contract", c->contract);
    fn->setAttr("tessera.schedule_hash", hash);
    auto *block = fn.addEntryBlock(b); b.setInsertionPointToStart(block);
    OperationState kernel(graph->getLoc(), c->backward ? "tile.attention_backward_rows_kernel" : "tile.attention_row_correction_kernel");
    kernel.addOperands(block->getArguments());
    for (StringRef name : {"shape", "storage", "output_storage", "row_storage", "accum",
                          "row_correction", "scale", "causal", "bias", "lse_cotangent",
                          "row_bytes", "block_threads", "bias_shape", "bias_gradient",
                          "gradient_activity", "gradient_output", "gradient_launch", "row_correction_usage"})
      if (auto attr = c->contract.get(name)) kernel.addAttribute(name, attr);
    b.create(kernel); LLVM::ReturnOp::create(b, graph->getLoc(), ValueRange{});
    c->function.erase();
  }
  return success();
}
} // namespace
