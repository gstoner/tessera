// Native checkpoint boundaries, owned by GraphToSchedule / ScheduleToTile.
// Included after the common PM dialect/MLIR headers inside namespace tessera.
namespace {
struct NativeCheckpoint {
  func::FuncOp function;
  Operation *graph;
  bool backward;
  bool bias;
  bool biasGradient;
  bool lseCotangent;
  SmallVector<int64_t> dims;
  SmallVector<int64_t> biasShape;
  DictionaryAttr contract;
  std::string hash;
};

static FailureOr<NativeCheckpoint> checkpointContract(Operation *graph) {
  bool backward = graph->getName().getStringRef() == "tessera_attn.checkpoint_backward";
  auto seedAttr = graph->getAttrOfType<BoolAttr>("lse_cotangent");
  bool lseCotangent = seedAttr && seedAttr.getValue();
  if (graph->hasAttr("lse_cotangent") && (!seedAttr || !backward))
    return graph->emitError("LSE cotangent requires a boolean backward policy"), failure();
  bool bias = graph->getNumOperands() == (backward ? 7u : 4u) + unsigned(lseCotangent);
  bool biasGradient = backward && graph->getNumResults() == 4;
  auto fn = graph->getParentOfType<func::FuncOp>();
  auto mod = graph->getParentOfType<ModuleOp>();
  auto target = mod ? mod->getAttrOfType<StringAttr>("tessera.target") : StringAttr();
  auto arch = mod ? mod->getAttrOfType<StringAttr>("tessera.arch") : StringAttr();
  if (!fn || !target || target.getValue() != "nvidia_sm120" ||
      !arch || arch.getValue() != "sm_120" || !llvm::hasSingleElement(fn.getBody()) ||
      fn.getNumArguments() != (backward ? 6 : 3) + unsigned(bias) + unsigned(lseCotangent) ||
      fn.getNumResults() != (backward ? 3 + unsigned(biasGradient) : 2) ||
      graph->getNumOperands() != fn.getNumArguments() ||
      graph->getResultTypes() != fn.getResultTypes())
    return graph->emitError("checkpoint requires an isolated SM120 f32 tensor entry"), failure();
  SmallVector<unsigned> argumentRoles;
  llvm::SmallDenseSet<unsigned> distinctRoles;
  for (Value operand : graph->getOperands()) {
    auto argument = dyn_cast<BlockArgument>(operand);
    if (!argument || argument.getOwner() != &fn.getBody().front() ||
        !distinctRoles.insert(argument.getArgNumber()).second)
      return graph->emitError("checkpoint operands require distinct function argument roles"), failure();
    argumentRoles.push_back(argument.getArgNumber());
  }
  for (unsigned i = 0; i < fn.getNumArguments(); ++i)
    if (fn.getArgAttr(i, "tessera.layout"))
      return graph->emitError("checkpoint layout overrides are unsupported"), failure();
  SmallVector<RankedTensorType> types;
  for (Type type : graph->getOperandTypes()) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || tensor.getEncoding() || !tensor.getElementType().isF32() ||
        llvm::any_of(tensor.getShape(), [](int64_t d) { return !ShapedType::isDynamic(d) && d <= 0; }))
      return graph->emitError("checkpoint requires positive or bounded f32 shapes"), failure();
    types.push_back(tensor);
  }
  unsigned base = backward ? 1 : 0;
  auto q = types[base], k = types[base + 1], v = types[base + 2];
  if (q.getRank() != 4 || k.getRank() != 4 || v.getRank() != 4)
    return graph->emitError("checkpoint Q/K/V must have rank four"), failure();
  int64_t b=q.getDimSize(0), hq=q.getDimSize(1), sq=q.getDimSize(2), d=q.getDimSize(3);
  int64_t hkv=k.getDimSize(1), sk=k.getDimSize(2), dv=v.getDimSize(3);

  SmallVector<int64_t> symbolic{b,hq,hkv,sq,sk,d,dv};
  auto shape = resolveNativeAttentionShape(graph, symbolic);
  if (failed(shape)) return failure();
  auto bounds = shape->bounds;

  auto tensor = [&](ArrayRef<int64_t> shape) { return RankedTensorType::get(shape, q.getElementType()); };
  auto output = tensor({b,hq,sq,dv}), lse = tensor({b,hq,sq});
  unsigned lseIndex = 5 + unsigned(bias);
  SmallVector<int64_t> biasShape;
  if (bias) {
    auto biasType = types[backward ? 5 : 3];
    if (biasType.getRank() != 4)
      return graph->emitError("checkpoint bias requires rank-four f32 shape"), failure();
    SmallVector<int64_t, 4> scores{b,hq,sq,sk};
    for (unsigned axis = 0; axis < 4; ++axis)
      if (biasType.getDimSize(axis) != 1 && biasType.getDimSize(axis) != scores[axis])
        return graph->emitError("checkpoint bias axes must be one or match [B,Hq,Sq,Sk]"), failure();
    if (biasGradient && fn.getResultTypes()[3] != biasType)
      return graph->emitError("checkpoint bias gradient must match physical bias shape"), failure();
    if (biasType.getShape() != ArrayRef<int64_t>(scores))
      biasShape.assign(biasType.getShape().begin(), biasType.getShape().end());
  } else if (biasGradient) {
    return graph->emitError("checkpoint bias gradient requires a bias operand"), failure();
  }
  if (k != tensor({b,hkv,sk,d}) || v != tensor({b,hkv,sk,dv}) || hq % hkv ||
      (!backward && (fn.getResultTypes()[0] != output || fn.getResultTypes()[1] != lse)) ||
      (backward && (types[0] != output || types[4] != output || types[lseIndex] != lse ||
                    fn.getResultTypes()[0] != q || fn.getResultTypes()[1] != k || fn.getResultTypes()[2] != v)))
    return graph->emitError("checkpoint shapes or output roles disagree"), failure();
  if (lseCotangent && types.back() != lse)
    return graph->emitError("LSE cotangent must match saved row LSE"), failure();
  auto scale = graph->getAttrOfType<FloatAttr>("scale");
  auto causal = graph->getAttrOfType<BoolAttr>("causal");
  if (!scale || !scale.getType().isF32() || !std::isfinite(scale.getValueAsDouble()) ||
      scale.getValueAsDouble() <= 0 || !causal)
    return graph->emitError("checkpoint requires positive finite f32 scale and boolean causal"), failure();
  for (NamedAttribute attr : graph->getAttrs())
    if (attr.getName() != "scale" && attr.getName() != "causal" && attr.getName() != "lse_cotangent" && attr.getName() != "schedule.artifact_hash")
      return graph->emitError("checkpoint has an unsupported policy attribute"), failure();
  auto args = fn->getAttrOfType<ArrayAttr>("tessera.argument_bindings");
  if (args && args.size() == fn.getNumArguments()) {
    SmallVector<Attribute> ordered;
    for (unsigned role : argumentRoles) ordered.push_back(args[role]);
    args = ArrayAttr::get(graph->getContext(), ordered);
  }
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
      builder.getNamedAttr("bias", builder.getBoolAttr(bias)),
      builder.getNamedAttr("scale", scale), builder.getNamedAttr("causal", causal),
      builder.getNamedAttr("arguments", args), builder.getNamedAttr("results", results),
      builder.getNamedAttr("mask_alignment", builder.getStringAttr("end_aligned_v1")),
      builder.getNamedAttr("target", target), builder.getNamedAttr("arch", arch)});
  if (bounds) {
    NamedAttrList fields(contract);
    fields.set("shape_bounds", bounds);
    fields.set("shape_policy", builder.getStringAttr("bounded_sequences_v1"));
    contract = builder.getDictionaryAttr(fields);
  }
  if (lseCotangent) {
    NamedAttrList fields(contract);
    fields.set("lse_cotangent", builder.getBoolAttr(true));
    contract = builder.getDictionaryAttr(fields);
  }
  if (auto activityAttr = fn->getAttr("tessera.checkpoint_gradient_activity")) {
    auto activity = dyn_cast<DenseI64ArrayAttr>(activityAttr);
    if (!backward || !activity || activity.size() != graph->getNumResults() ||
        llvm::any_of(activity.asArrayRef(), [](int64_t x) { return x != 0 && x != 1; }) ||
        llvm::none_of(activity.asArrayRef(), [](int64_t x) { return x == 1; }))
      return graph->emitError("checkpoint gradient activity requires nonempty binary backward result roles"), failure();
    NamedAttrList fields(contract);
    fields.set("gradient_activity", activity);
    fields.set("inactive_gradient", builder.getStringAttr("zero_fill_v1"));
    contract = builder.getDictionaryAttr(fields);
  }
  if (auto outputAttr = fn->getAttr("tessera.checkpoint_gradient_output")) {
    auto output = dyn_cast<StringAttr>(outputAttr);
    if (!backward || !contract.get("gradient_activity") || !output || output.getValue() != "compact_v1")
      return graph->emitError("compact checkpoint outputs require verified backward gradient activity"), failure();
    auto launch = fn->getAttrOfType<StringAttr>("tessera.checkpoint_gradient_launch");
    if (!launch || (launch.getValue() != "packed_v1" && launch.getValue() != "logical_v1"))
      return graph->emitError("compact checkpoint launch requires packed_v1 or logical_v1"), failure();
    auto threads = fn->getAttrOfType<IntegerAttr>("tessera.checkpoint_gradient_threads");
    if (!threads || !threads.getType().isInteger(64) || (threads.getInt() != 64 && threads.getInt() != 128))
      return graph->emitError("compact checkpoint threads require 64 or 128"), failure();
    auto activity = cast<DenseI64ArrayAttr>(contract.get("gradient_activity"));
    SmallVector<Attribute> physical;
    for (unsigned i = 0; i < results.size(); ++i)
      if (activity[i]) physical.push_back(results[i]);
    NamedAttrList fields(contract);
    fields.set("inactive_gradient", builder.getStringAttr("absent_v1"));
    fields.set("gradient_output", output);
    fields.set("gradient_launch", launch);
    fields.set("gradient_block_threads", threads);
    fields.set("physical_results", builder.getArrayAttr(physical));
    contract = builder.getDictionaryAttr(fields);
  }
  if (!biasShape.empty()) {
    NamedAttrList fields(contract);
    fields.set("bias_shape", builder.getDenseI64ArrayAttr(biasShape));
    fields.set("bias_gradient_reduction", builder.getStringAttr("physical_owner_lexicographic_bhqk_v1"));
    contract = builder.getDictionaryAttr(fields);
  }
  if (auto mappingAttr = mod->getAttr("tessera.attention_argument_indices")) {
    auto mapping = dyn_cast<DenseI64ArrayAttr>(mappingAttr);
    unsigned count = 3 + unsigned(bias);
    llvm::SmallDenseSet<int64_t> unique;
    if (!mapping || mapping.size() != count)
      return graph->emitError("checkpoint frontend input mapping requires all input roles"), failure();
    for (int64_t index : mapping.asArrayRef())
      if (index < 0 || index >= count || !unique.insert(index).second)
        return graph->emitError("checkpoint frontend input mapping must be a permutation"), failure();
    NamedAttrList fields(contract);
    fields.set("frontend_argument_indices", mapping);
    contract = builder.getDictionaryAttr(fields);
  }
  std::string text; llvm::raw_string_ostream os(text); contract.print(os); os.flush();
  auto hash = llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
  return NativeCheckpoint{fn,graph,backward,bias,biasGradient,lseCotangent,dims,biasShape,contract,hash};
}

// Preserve directly authored saved-LSE Graph semantics in the native pipeline.
// The frontend supplies names only; this pass owns checkpoint conversion.
static LogicalResult importSavedAttentionGraphs(ModuleOp mod) {
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch = mod->getAttrOfType<StringAttr>("tessera.arch");
  if (!target || target.getValue() != "nvidia_sm120" ||
      !arch || arch.getValue() != "sm_120") return success();
  SmallVector<Operation *> graphs;
  mod.walk([&](Operation *op) {
    auto name = op->getName().getStringRef();
    auto saved = op->getAttrOfType<StringAttr>("lse_checkpoint");
    if ((name == "tessera.flash_attn" || name == "tessera.flash_attn_bwd") &&
        saved && saved.getValue() == "saved") graphs.push_back(op);
  });
  for (Operation *graph : graphs) {
    bool backward = graph->getName().getStringRef() == "tessera.flash_attn_bwd";
    unsigned qIndex = backward ? 1 : 0;
    if (graph->getNumOperands() <= qIndex)
      return graph->emitError("saved attention Graph has no query operand");
    auto query = dyn_cast<RankedTensorType>(graph->getOperand(qIndex).getType());
    if (!query || query.getRank() != 4 ||
        query.getDimSize(3) <= 0)
      return graph->emitError("saved attention requires rank-four query with fixed positive head width");
    for (NamedAttribute attr : graph->getAttrs()) {
      StringRef name = attr.getName().strref();
      if (name == "scale" || name == "causal" || name == "lse_checkpoint" ||
          name == "operandSegmentSizes" || name == "tessera.effect_kind") continue;
      if (backward && name == "lse_cotangent" && isa<BoolAttr>(attr.getValue())) continue;
      if (name == "head_dim") {
        auto dim = dyn_cast<IntegerAttr>(attr.getValue());
        if (dim && dim.getInt() == query.getDimSize(3)) continue;
      } else if (name == "window_left" || name == "window_right" || name == "window") {
        auto integer = dyn_cast<IntegerAttr>(attr.getValue());
        auto array = dyn_cast<ArrayAttr>(attr.getValue());
        if ((integer && integer.getInt() == -1) ||
            (name == "window" && array && array.size() == 2 &&
             llvm::all_of(array, [](Attribute a) {
               auto value = dyn_cast<IntegerAttr>(a);
               return value && value.getInt() == -1;
             }))) continue;
      } else if (name == "softcap" || name == "logit_softcap" ||
                 name == "dropout" || name == "dropout_p") {
        auto number = dyn_cast<FloatAttr>(attr.getValue());
        auto integer = dyn_cast<IntegerAttr>(attr.getValue());
        if ((number && number.getValueAsDouble() == 0) ||
            (integer && !isa<BoolAttr>(attr.getValue()) && integer.getInt() == 0)) continue;
      } else if (name == "bias") {
        auto disabled = dyn_cast<BoolAttr>(attr.getValue());
        if (disabled && !disabled.getValue()) continue;
      } else if (backward && name == "route") {
        auto route = dyn_cast<StringAttr>(attr.getValue());
        if (route && route.getValue() == "deterministic_direct") continue;
      } else if (backward && name == "deterministic") {
        auto deterministic = dyn_cast<BoolAttr>(attr.getValue());
        if (deterministic && deterministic.getValue()) continue;
      } else if (backward && name == "workspace_limit_bytes") {
        auto bytes = dyn_cast<IntegerAttr>(attr.getValue());
        if (bytes && bytes.getInt() == 0) continue;
      }
      return graph->emitError("saved attention Graph has an unsupported policy attribute: ") << name;
    }
    OpBuilder builder(graph);
    auto scale = graph->getAttrOfType<FloatAttr>("scale");
    auto integerScale = graph->getAttrOfType<IntegerAttr>("scale");
    if (graph->hasAttr("scale") && !scale &&
        (!integerScale || isa<BoolAttr>(integerScale)))
      return graph->emitError("saved attention scale must be numeric");
    double scaleValue = scale ? scale.getValueAsDouble() :
        integerScale ? double(integerScale.getInt()) :
        1.0 / std::sqrt(double(query.getDimSize(3)));
    auto causal = graph->getAttrOfType<BoolAttr>("causal");
    if (graph->hasAttr("causal") && !causal)
      return graph->emitError("saved attention causal must be boolean");
    OperationState state(graph->getLoc(), backward ?
        "tessera_attn.checkpoint_backward" : "tessera_attn.checkpoint_forward");
    state.addOperands(graph->getOperands());
    state.addTypes(graph->getResultTypes());
    state.addAttribute("scale", builder.getF32FloatAttr(scaleValue));
    state.addAttribute("causal", builder.getBoolAttr(causal && causal.getValue()));
    if (backward && graph->hasAttr("lse_cotangent"))
      state.addAttribute("lse_cotangent", graph->getAttr("lse_cotangent"));
    Operation *checkpoint = builder.create(state);
    auto checked = checkpointContract(checkpoint);
    if (failed(checked)) {
      checkpoint->erase();
      return failure();
    }
    for (auto [before, after] : llvm::zip(graph->getResults(), checkpoint->getResults()))
      before.replaceAllUsesWith(after);
    graph->erase();
  }
  return success();
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
    bool compact = bool(c->contract.get("gradient_output"));
    unsigned outputCount = c->backward ? 3 + unsigned(c->biasGradient) : 2;
    unsigned mask = 0;
    if (compact) {
      auto activity = cast<DenseI64ArrayAttr>(c->contract.get("gradient_activity"));
      outputCount = 0;
      for (unsigned i = 0; i < activity.size(); ++i)
        if (activity[i]) { ++outputCount; mask |= 1u << i; }
    }
    SmallVector<Type> types((c->backward ? 6 : 3) + unsigned(c->bias) + unsigned(c->lseCotangent) + outputCount, ptr);
    bool dynamicBias = llvm::any_of(c->biasShape, [](int64_t extent) {
      return ShapedType::isDynamic(extent);
    });
    unsigned scalarCount = (compact || dynamicBias) && !c->biasShape.empty() ? 11 : 7;
    types.append(scalarCount, builder.getI64Type());
    std::string entry = (Twine("tessera_tile_attention_") +
        (c->backward ? (c->biasGradient ? "backward_lse_output_bias_gradient_" : "backward_lse_output_") : "lse_") +
        c->hash.substr(0,10)).str();
    if (c->lseCotangent)
      entry = std::string(c->biasGradient ? "tessera_tile_attention_backward_lse_output_bias_gradient_cotangent_" : "tessera_tile_attention_backward_lse_output_cotangent_") + c->hash.substr(0,10);
    if (compact) {
      entry = (Twine("tessera_tile_attention_backward_lse_output_compact_m") +
          Twine(mask) + "_b" + Twine(unsigned(c->bias)) + "_g" +
          Twine(unsigned(c->biasGradient)) + "_l" +
          Twine(unsigned(cast<StringAttr>(c->contract.get("gradient_launch")).getValue() == "logical_v1")) +
          "_t" + Twine(cast<IntegerAttr>(c->contract.get("gradient_block_threads")).getInt()) +
          "_" + c->hash.substr(0,10)).str();
    }
    if (c->lseCotangent && compact) entry += "_cotangent_";
    if (SymbolTable::lookupSymbolIn(mod, entry)) return scheduled->emitError("checkpoint entry symbol collision");
    builder.setInsertionPointToEnd(mod.getBody());
    auto fn = LLVM::LLVMFuncOp::create(builder, scheduled->getLoc(), entry,
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(mod.getContext()), types, false));
    fn->setAttr("nvvm.kernel", builder.getUnitAttr());
    fn->setAttr("tessera.native_contract", c->contract);
    fn->setAttr("tessera.schedule_hash", hash);
    auto block = fn.addEntryBlock(builder); builder.setInsertionPointToStart(block);
    OperationState kernel(scheduled->getLoc(), c->backward ? "tile.attention_backward_kernel" : "tile.attention_kernel");
    // Symbolic physical bias carries four checked runtime extents after the
    // logical dimensions. Static carriers retain their original operand ABI.
    kernel.addOperands(block->getArguments().take_front(
        types.size() - scalarCount + (dynamicBias ? 11 : 7)));
    kernel.addAttribute("storage",builder.getStringAttr("f32"));
    kernel.addAttribute("accum",builder.getStringAttr("f32"));
    kernel.addAttribute("scale",graph->getAttr("scale")); kernel.addAttribute("causal",graph->getAttr("causal"));
    kernel.addAttribute("bias",builder.getBoolAttr(c->bias));
    if (!c->biasShape.empty())
      kernel.addAttribute("bias_shape", builder.getDenseI64ArrayAttr(c->biasShape));
    kernel.addAttribute("window_left",builder.getI64IntegerAttr(-1)); kernel.addAttribute("window_right",builder.getI64IntegerAttr(-1));
    kernel.addAttribute("softcap",builder.getF32FloatAttr(0)); kernel.addAttribute("dropout_p",builder.getF32FloatAttr(0));
    kernel.addAttribute("dropout_seed",builder.getI64IntegerAttr(0));
    kernel.addAttribute("lse_checkpoint",builder.getStringAttr("saved"));
    if (c->backward) kernel.addAttribute("saved_output", builder.getBoolAttr(true));
    if (c->lseCotangent) kernel.addAttribute("lse_cotangent", builder.getBoolAttr(true));
    if (auto activity = c->contract.get("gradient_activity"))
      kernel.addAttribute("gradient_activity", activity);
    if (compact) {
      kernel.addAttribute("gradient_output", builder.getStringAttr("compact_v1"));
      kernel.addAttribute("gradient_launch", c->contract.get("gradient_launch"));
      kernel.addAttribute("block_threads", c->contract.get("gradient_block_threads"));
    }
    if (c->biasGradient) kernel.addAttribute("bias_gradient", builder.getBoolAttr(true));
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
