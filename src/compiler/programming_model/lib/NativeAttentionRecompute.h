// Original recompute Graph -> sealed Schedule -> native launch Tile.
namespace {
struct NativeAttentionRecompute {
  func::FuncOp function;
  Operation *graph;
  DictionaryAttr contract;
  std::string hash;
};

static FailureOr<NativeAttentionRecompute> attentionRecomputeContract(Operation *op) {
  auto fn = op->getParentOfType<func::FuncOp>();
  auto mod = op->getParentOfType<ModuleOp>();
  auto target = mod ? mod->getAttrOfType<StringAttr>("tessera.target") : StringAttr();
  auto arch = mod ? mod->getAttrOfType<StringAttr>("tessera.arch") : StringAttr();
  unsigned count = op->getNumOperands();
  bool bias = count == 5;
  if (!fn || !target || target.getValue() != "nvidia_sm120" || !arch ||
      arch.getValue() != "sm_120" || !llvm::hasSingleElement(fn.getBody()) ||
      (count != 4 && count != 5) || fn.getNumArguments() != count ||
      op->getNumResults() != 3 || op->getResultTypes() != fn.getResultTypes())
    return op->emitError("recompute attention requires an isolated SM120 tensor entry"), failure();
  auto checkpoint = op->getAttrOfType<StringAttr>("lse_checkpoint");
  if (op->hasAttr("lse_checkpoint") && (!checkpoint || checkpoint.getValue() != "recompute"))
    return op->emitError("recompute attention requires the recompute checkpoint policy"), failure();
  SmallVector<unsigned> roles;
  llvm::SmallDenseSet<unsigned> unique;
  SmallVector<RankedTensorType> tensors;
  for (Value value : op->getOperands()) {
    auto arg = dyn_cast<BlockArgument>(value);
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!arg || arg.getOwner() != &fn.getBody().front() ||
        !unique.insert(arg.getArgNumber()).second || fn.getArgAttr(arg.getArgNumber(), "tessera.layout") ||
        !tensor || tensor.getRank() != 4 || tensor.getEncoding() || !tensor.hasStaticShape() ||
        llvm::any_of(tensor.getShape(), [](int64_t d) { return d <= 0; }))
      return op->emitError("recompute attention requires distinct static rank-four row-major arguments"), failure();
    roles.push_back(arg.getArgNumber()); tensors.push_back(tensor);
  }
  auto q=tensors[1], k=tensors[2], v=tensors[3], dout=tensors[0];
  Type element=q.getElementType();
  if ((!element.isF16() && !element.isBF16() && !element.isF32()) ||
      k.getElementType()!=element || v.getElementType()!=element || dout.getElementType()!=element)
    return op->emitError("recompute attention requires matching f16/bf16/f32 storage"), failure();
  int64_t b=q.getDimSize(0), hq=q.getDimSize(1), sq=q.getDimSize(2), d=q.getDimSize(3);
  int64_t hkv=k.getDimSize(1), sk=k.getDimSize(2), dv=v.getDimSize(3);
  if (k.getDimSize(0)!=b || v.getDimSize(0)!=b || v.getDimSize(1)!=hkv ||
      v.getDimSize(2)!=sk || k.getDimSize(3)!=d || hq%hkv ||
      dout.getShape()!=ArrayRef<int64_t>{b,hq,sq,dv})
    return op->emitError("recompute attention has inconsistent batch/head/sequence/output dimensions"), failure();
  for (unsigned i=0;i<3;++i)
    if (op->getResult(i).getType()!=tensors[i+1])
      return op->emitError("recompute gradient types must match Q/K/V"), failure();
  if (bias && (!tensors[4].getElementType().isF32() ||
               tensors[4].getShape()!=ArrayRef<int64_t>{b,hq,sq,sk}))
    return op->emitError("recompute attention bias requires full f32 [B,Hq,Sq,Sk]"), failure();
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name=attr.getName().strref();
    if (name=="scale" || name=="causal" || name=="window" || name=="window_left" ||
        name=="window_right" || name=="softcap" || name=="logit_softcap" ||
        name=="dropout_p" || name=="dropout" || name=="dropout_seed" || name=="seed" ||
        name=="route" || name=="deterministic" || name=="workspace_limit_bytes" ||
        name=="lse_checkpoint" || name=="tessera.effect_kind" || name=="schedule.artifact_hash" || name=="head_dim") continue;
    return op->emitError("recompute attention has an unsupported policy attribute: ") << name, failure();
  }
  if (op->hasAttr("head_dim")) {
    auto width=op->getAttrOfType<IntegerAttr>("head_dim");
    if (!width || isa<BoolAttr>(width) || width.getInt()!=d)
      return op->emitError("recompute attention head_dim must match query width"), failure();
  }
  bool valid=true;
  auto integer = [&](StringRef name, int64_t fallback) {
    auto attr=op->getAttr(name); if (!attr) return fallback;
    auto value=dyn_cast<IntegerAttr>(attr);
    if (!value || isa<BoolAttr>(attr)) { valid=false; return fallback; }
    return value.getInt();
  };
  auto number = [&](StringRef name, double fallback) {
    auto attr=op->getAttr(name); if (!attr) return fallback;
    if (auto value=dyn_cast<FloatAttr>(attr)) return value.getValueAsDouble();
    if (auto value=dyn_cast<IntegerAttr>(attr); value && !isa<BoolAttr>(attr))
      return double(value.getInt());
    valid=false; return fallback;
  };
  auto boolean = [&](StringRef name, bool fallback) {
    auto attr=op->getAttr(name); if (!attr) return fallback;
    auto value=dyn_cast<BoolAttr>(attr);
    if (!value) { valid=false; return fallback; } return value.getValue();
  };
  int64_t left=integer("window_left",-1), right=integer("window_right",-1);
  if (auto window=op->getAttr("window")) {
    int64_t wl=-1, wr=-1;
    if (auto value=dyn_cast<IntegerAttr>(window); value && !isa<BoolAttr>(window)) wl=wr=value.getInt();
    else if (auto array=dyn_cast<ArrayAttr>(window); array && array.size()==2 &&
             isa<IntegerAttr>(array[0]) && !isa<BoolAttr>(array[0]) &&
             isa<IntegerAttr>(array[1]) && !isa<BoolAttr>(array[1])) {
      wl=cast<IntegerAttr>(array[0]).getInt(); wr=cast<IntegerAttr>(array[1]).getInt();
    } else valid=false;
    if ((op->hasAttr("window_left") && left!=wl) ||
        (op->hasAttr("window_right") && right!=wr)) valid=false;
    left=wl; right=wr;
  }
  double scale=double(float(number("scale",1.0/std::sqrt(double(d)))));
  double softcap=double(float(number("softcap",number("logit_softcap",0))));
  double dropout=double(float(number("dropout_p",number("dropout",0))));
  if ((op->hasAttr("softcap") && op->hasAttr("logit_softcap") &&
       float(number("softcap",0))!=float(number("logit_softcap",0))) ||
      (op->hasAttr("dropout_p") && op->hasAttr("dropout") &&
       float(number("dropout_p",0))!=float(number("dropout",0)))) valid=false;
  int64_t seed=integer("dropout_seed",integer("seed",0));
  if (op->hasAttr("dropout_seed") && op->hasAttr("seed") &&
      integer("dropout_seed",0)!=integer("seed",0)) valid=false;
  auto route=op->getAttrOfType<StringAttr>("route");
  bool causal=boolean("causal",false), deterministic=boolean("deterministic",true);
  int64_t workspace=integer("workspace_limit_bytes",0);
  if (!valid || !std::isfinite(scale) || scale<=0 || !std::isfinite(softcap) || softcap<0 ||
      !std::isfinite(dropout) || dropout<0 || dropout>=1 || left < -1 || right < -1 ||
      !deterministic || workspace<0 || (op->hasAttr("route") && (!route || route.getValue()!="deterministic_direct")))
    return op->emitError("recompute attention has an invalid or conflicting numerical/route policy"), failure();
  auto args=fn->getAttrOfType<ArrayAttr>("tessera.argument_bindings");
  auto results=fn->getAttrOfType<ArrayAttr>("tessera.result_bindings");
  if (!args || args.size()!=count || !results || results.size()!=3)
    return op->emitError("recompute attention requires complete frontend binding metadata"), failure();
  llvm::SmallDenseSet<StringRef> names;
  SmallVector<Attribute> ordered;
  for (unsigned role:roles) ordered.push_back(args[role]);
  SmallVector<Attribute> bindingNames(ordered);
  llvm::append_range(bindingNames,results.getValue());
  for (Attribute name:bindingNames) {
    auto str=dyn_cast<StringAttr>(name);
    if (!str || str.getValue().empty() || !names.insert(str.getValue()).second)
      return op->emitError("recompute attention binding names must be unique"), failure();
  }
  OpBuilder builder(op->getContext());
  auto contract=builder.getDictionaryAttr({
    builder.getNamedAttr("family",builder.getStringAttr("attention_backward_recompute")),
    builder.getNamedAttr("target",target), builder.getNamedAttr("arch",arch),
    builder.getNamedAttr("storage",builder.getStringAttr(element.isF16()?"f16":element.isBF16()?"bf16":"f32")),
    builder.getNamedAttr("shape",builder.getDenseI64ArrayAttr({b,hq,hkv,sq,sk,d,dv})),
    builder.getNamedAttr("arguments",builder.getArrayAttr(ordered)), builder.getNamedAttr("results",results),
    builder.getNamedAttr("bias",builder.getBoolAttr(bias)), builder.getNamedAttr("scale",builder.getF32FloatAttr(scale)),
    builder.getNamedAttr("causal",builder.getBoolAttr(causal)),
    builder.getNamedAttr("window_left",builder.getI64IntegerAttr(left)),
    builder.getNamedAttr("window_right",builder.getI64IntegerAttr(right)),
    builder.getNamedAttr("softcap",builder.getF32FloatAttr(softcap)),
    builder.getNamedAttr("dropout_p",builder.getF32FloatAttr(dropout)),
    builder.getNamedAttr("dropout_seed",builder.getI64IntegerAttr(seed)),
    builder.getNamedAttr("route",builder.getStringAttr("deterministic_direct")),
    builder.getNamedAttr("deterministic",builder.getBoolAttr(true)),
    builder.getNamedAttr("workspace_bytes",builder.getI64IntegerAttr(0)),
    builder.getNamedAttr("workspace_owner",builder.getStringAttr("output_element")),
    builder.getNamedAttr("lse_checkpoint",builder.getStringAttr("recompute")),
    builder.getNamedAttr("mask_alignment",builder.getStringAttr("end_aligned_v1"))});
  std::string text; llvm::raw_string_ostream os(text); contract.print(os); os.flush();
  auto hash=llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)),true);
  return NativeAttentionRecompute{fn,op,contract,hash};
}

static LogicalResult scheduleNativeAttentionRecompute(ModuleOp mod) {
  auto target=mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch=mod->getAttrOfType<StringAttr>("tessera.arch");
  if (!target || target.getValue()!="nvidia_sm120" || !arch || arch.getValue()!="sm_120") return success();
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) {
    if (op->getName().getStringRef()=="tessera.flash_attn_bwd") ops.push_back(op);
  });
  for (Operation *graph:ops) {
    auto c=attentionRecomputeContract(graph); if (failed(c)) return failure();
    auto ret=dyn_cast<func::ReturnOp>(c->function.getBody().front().back());
    if (c->function.getBody().front().getOperations().size()!=2 || !ret ||
        ret.getOperands()!=graph->getResults())
      return graph->emitError("recompute attention requires only its producer and ordered return");
    OpBuilder builder(graph); builder.setInsertionPointAfter(graph);
    graph->setAttr("schedule.artifact_hash",builder.getStringAttr(c->hash));
    OperationState state(graph->getLoc(),"schedule.attention_checkpoint");
    state.addOperands(graph->getResults()); state.addTypes(graph->getResultTypes());
    state.addAttribute("artifact_hash",builder.getStringAttr(c->hash)); state.addAttribute("contract",c->contract);
    auto scheduled=builder.create(state);
    for (auto [oldValue,newValue]:llvm::zip(graph->getResults(),scheduled->getResults()))
      oldValue.replaceAllUsesExcept(newValue,scheduled);
  }
  return success();
}

static LogicalResult lowerNativeAttentionRecompute(ModuleOp mod) {
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) {
    if (op->getName().getStringRef()=="schedule.attention_checkpoint" && op->getNumOperands() &&
        op->getOperand(0).getDefiningOp() &&
        op->getOperand(0).getDefiningOp()->getName().getStringRef()=="tessera.flash_attn_bwd")
      ops.push_back(op);
  });
  for (Operation *scheduled:ops) {
    Operation *graph=scheduled->getOperand(0).getDefiningOp();
    auto c=attentionRecomputeContract(graph); if (failed(c)) return failure();
    auto hash=scheduled->getAttrOfType<StringAttr>("artifact_hash");
    auto ret=dyn_cast<func::ReturnOp>(c->function.getBody().front().back());
    if (!hash || hash.getValue()!=c->hash || graph->getAttr("schedule.artifact_hash")!=hash ||
        scheduled->getAttr("contract")!=c->contract || scheduled->getAttrs().size()!=2 ||
        scheduled->getOperands()!=graph->getResults() || scheduled->getResultTypes()!=graph->getResultTypes() ||
        c->function.getBody().front().getOperations().size()!=3 || !ret ||
        ret.getOperands()!=scheduled->getResults())
      return scheduled->emitError("recompute attention Schedule contract changed after hashing");
    OpBuilder builder(mod.getContext()); auto ptr=LLVM::LLVMPointerType::get(mod.getContext());
    SmallVector<Type> types(graph->getNumOperands()+3,ptr); types.append(7,builder.getI64Type());
    // The existing host/resident ABI derives transfer width from this storage
    // component. Keep it explicit in the native symbol as well as the contract.
    std::string storage=cast<StringAttr>(c->contract.get("storage")).getValue().str();
    std::string entry=(Twine("tessera_tile_attention_backward_")+storage+
                       "_recompute_"+c->hash.substr(0,10)).str();
    if (SymbolTable::lookupSymbolIn(mod,entry)) return scheduled->emitError("recompute entry symbol collision");
    builder.setInsertionPointToEnd(mod.getBody());
    auto fn=LLVM::LLVMFuncOp::create(builder,scheduled->getLoc(),entry,
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(mod.getContext()),types,false));
    fn->setAttr("nvvm.kernel",builder.getUnitAttr());
    fn->setAttr("tessera.native_contract",c->contract); fn->setAttr("tessera.schedule_hash",hash);
    auto block=fn.addEntryBlock(builder); builder.setInsertionPointToStart(block);
    OperationState kernel(scheduled->getLoc(),"tile.attention_backward_kernel");
    kernel.addOperands(block->getArguments());
    for (StringRef name:{"storage","scale","causal","bias","window_left","window_right","softcap","dropout_p",
                         "dropout_seed","route","deterministic","workspace_bytes","workspace_owner","lse_checkpoint"})
      kernel.addAttribute(name,c->contract.get(name));
    kernel.addAttribute("accum",builder.getStringAttr("f32"));
    builder.create(kernel); LLVM::ReturnOp::create(builder,scheduled->getLoc(),ValueRange{});
    c->function.erase();
  }
  return success();
}
} // namespace
