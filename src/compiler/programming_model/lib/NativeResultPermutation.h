// Native Graph -> immutable Schedule -> typed Tile result permutation.
// The original Graph and all semantic attributes remain the replay witness.
namespace {
static FailureOr<DictionaryAttr> resultPermutationContract(Operation *op) {
  auto mod = op->getParentOfType<ModuleOp>();
  auto fn = op->getParentOfType<func::FuncOp>();
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch = mod->getAttrOfType<StringAttr>("tessera.arch");
  auto bindings = mod->getAttrOfType<ArrayAttr>("tessera.launch_bindings");
  auto axes = tessera::transposePermutation(op);
  auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!fn || !fn.getBody().hasOneBlock() || fn.getNumArguments()!=1 ||
      fn.getNumResults()!=1 || !target || target.getValue()!="rocm" ||
      !arch || arch.getValue()!="gfx1201" || !bindings || bindings.size()!=2 ||
      !axes || !input.hasStaticShape() || !output.hasStaticShape() ||
      !input.getElementType().isF32() || input.getEncoding() || output.getEncoding() ||
      op->getOperand(0)!=fn.getArgument(0) || fn.getResultTypes()[0]!=output)
    return op->emitError("result permutation requires isolated static compact f32 gfx1201 storage"), failure();
  auto count=tessera::staticPermutationElements(input.getShape(),*axes);
  if (!count || (*count-1)/256+1 > INT32_MAX)
    return op->emitError("result permutation count exceeds its native launch ABI"), failure();
  llvm::StringSet<> distinct;
  for (auto attr: bindings) {
    auto name=dyn_cast<StringAttr>(attr);
    if (!name || name.getValue().empty() || !distinct.insert(name.getValue()).second)
      return op->emitError("result permutation source and output bindings must be distinct"), failure();
  }
  OpBuilder b(op);
  SmallVector<NamedAttribute> semantic;
  for (auto attr: op->getAttrs()) {
    if (attr.getName()=="schedule.artifact_hash") continue;
    if (attr.getName()!="permutation" &&
        !(attr.getName()=="tessera.effect_kind" && attr.getValue()==b.getStringAttr("pure")) &&
        !(attr.getName()=="tessera.autodiff.activity" &&
          (attr.getValue()==b.getStringAttr("active") || attr.getValue()==b.getStringAttr("inactive"))))
      return op->emitError("result permutation policy requires its owning native consumer"), failure();
    semantic.push_back(attr);
  }
  return b.getDictionaryAttr({
    b.getNamedAttr("source_shape",b.getDenseI64ArrayAttr(input.getShape())),
    b.getNamedAttr("output_shape",b.getDenseI64ArrayAttr(output.getShape())),
    b.getNamedAttr("permutation",b.getDenseI64ArrayAttr(*axes)),
    b.getNamedAttr("elements",b.getI64IntegerAttr(*count)),
    b.getNamedAttr("bindings",bindings),b.getNamedAttr("architecture",arch),
    b.getNamedAttr("semantic_attributes",b.getDictionaryAttr(semantic)),
    b.getNamedAttr("block_size",b.getI64IntegerAttr(256))});
}
static std::string resultPermutationHash(DictionaryAttr contract) {
  std::string text; llvm::raw_string_ostream os(text); contract.print(os); os.flush();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)),true);
}
static LogicalResult scheduleNativeResultPermutation(ModuleOp mod) {
  auto target=mod->getAttrOfType<StringAttr>("tessera.target");
  if (!target || target.getValue()!="rocm" || !mod->hasAttr("tessera.launch_bindings")) return success();
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op){if (op->getName().getStringRef()=="tessera.transpose") ops.push_back(op);});
  for (auto *op: ops) {
    auto contract=resultPermutationContract(op);
    if (failed(contract)) return failure();
    auto fn=op->getParentOfType<func::FuncOp>();
    auto ret=dyn_cast<func::ReturnOp>(fn.getBody().front().back());
    if (fn.getBody().front().getOperations().size()!=2 || !ret || ret.getOperands()!=op->getResults())
      return op->emitError("result permutation must return its isolated Graph operation");
    OpBuilder b(op); b.setInsertionPointAfter(op);
    auto hash=b.getStringAttr(resultPermutationHash(*contract));
    op->setAttr("schedule.artifact_hash",hash);
    OperationState state(op->getLoc(),"schedule.artifact");
    state.addAttribute("hash",hash);state.addAttribute("arch",contract->get("architecture"));
    state.addAttribute("shape_key",b.getStringAttr("family=result_permutation"));
    state.addAttribute("contract",*contract);b.create(state);
  }
  return success();
}
static LogicalResult lowerNativeResultPermutation(ModuleOp mod) {
  SmallVector<schedule::ArtifactOp> records;
  mod.walk([&](schedule::ArtifactOp op){if(op.getShapeKey()=="family=result_permutation") records.push_back(op);});
  for (auto record: records) {
    auto fn=record->getParentOfType<func::FuncOp>();
    if(!fn || !fn.getBody().hasOneBlock() || fn.getBody().front().getOperations().size()!=3)
      return record.emitError("result permutation Schedule lost its isolated Graph parent");
    auto *op=&fn.getBody().front().front();
    if(op->getName().getStringRef()!="tessera.transpose") return record.emitError("result permutation Schedule lost its Graph");
    auto contract=resultPermutationContract(op);
    auto ret=dyn_cast<func::ReturnOp>(fn.getBody().front().back());
    if(failed(contract) || !ret || ret.getOperands()!=op->getResults() ||
       record->getAttr("contract")!=*contract || record.getHash()!=resultPermutationHash(*contract) ||
       record->getAttr("arch")!=contract->get("architecture") ||
       op->getAttr("schedule.artifact_hash")!=record->getAttr("hash"))
      return record.emitError("result permutation Schedule contract was altered");
    OpBuilder b(mod.getBody(),mod.getBody()->end());
    auto ptr=LLVM::LLVMPointerType::get(mod.getContext());
    auto type=LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(mod.getContext()),{ptr,ptr,b.getI64Type()},false);
    auto symbol=(Twine("tessera_tile_result_permutation_")+record.getHash().take_front(16)).str();
    if(SymbolTable::lookupSymbolIn(mod,symbol)) return record.emitError("result permutation symbol already exists");
    auto kernel=LLVM::LLVMFuncOp::create(b,op->getLoc(),symbol,type);
    kernel->setAttr("tessera.result_permutation_contract",*contract);
    kernel->setAttr("tessera.schedule_hash",record->getAttr("hash"));
    auto *entry=kernel.addEntryBlock(b);b.setInsertionPointToStart(entry);
    OperationState tile(op->getLoc(),"tile.transpose_kernel");
    tile.addOperands(entry->getArguments());
    tile.addAttribute("source_shape",contract->get("source_shape"));
    tile.addAttribute("permutation",contract->get("permutation"));
    tile.addAttribute("tessera.schedule_hash",record->getAttr("hash"));
    b.create(tile);LLVM::ReturnOp::create(b,op->getLoc(),ValueRange{});fn.erase();
  }
  return success();
}
} // namespace
