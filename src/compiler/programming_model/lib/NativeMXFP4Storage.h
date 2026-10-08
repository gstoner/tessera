// Typed checkpoint conversion ownership. Included in namespace tessera.
namespace {
static FailureOr<DictionaryAttr> nativeMXFP4StorageContract(Operation *graph) {
  auto fn = graph->getParentOfType<func::FuncOp>();
  auto mod = graph->getParentOfType<ModuleOp>();
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch = mod->getAttrOfType<StringAttr>("tessera.arch");
  if (!fn || !fn.getBody().hasOneBlock() || fn.getNumArguments() != 2 ||
      fn.getNumResults() != 2 || graph->getNumOperands() != 2 ||
      graph->getNumResults() != 2 || !target || !arch ||
      target.getValue() != "rocm_gfx1201" || arch.getValue() != "gfx1201" ||
      graph->getResultTypes() != fn.getResultTypes())
    return graph->emitError("MXFP4Storage ingest needs an isolated gfx1201 checkpoint entry"), failure();
  for (unsigned i = 0; i < 2; ++i) {
    auto attrs = fn.getArgAttrDict(i);
    if (graph->getOperand(i) != fn.getArgument(i) || (attrs && !attrs.empty()))
      return graph->emitError("MXFP4Storage ingest argument lineage/layout changed"), failure();
  }
  for (NamedAttribute attr : graph->getAttrs())
    if (attr.getName() != "storage_contract" &&
        attr.getName() != "schedule.artifact_hash" &&
        !(attr.getName() == "tessera.effect_kind" &&
          attr.getValue() == StringAttr::get(mod.getContext(),"pure")))
      return graph->emitError("MXFP4Storage ingest has an unsupported policy attribute"), failure();
  auto codes = dyn_cast<RankedTensorType>(graph->getOperand(0).getType());
  if (!codes || !codes.hasStaticShape() || codes.getRank() != 2)
    return graph->emitError("MXFP4Storage ingest lost static packed shape"), failure();
  auto names = fn->getAttrOfType<ArrayAttr>("tessera.bindings");
  if (!names || names.size() != 4)
    return graph->emitError("MXFP4Storage ingest requires four launch binding names"), failure();
  llvm::SmallDenseSet<StringRef> seen;
  for (Attribute attr : names) {
    auto name = dyn_cast<StringAttr>(attr);
    if (!name || name.getValue().empty() || !seen.insert(name.getValue()).second)
      return graph->emitError("MXFP4Storage ingest binding names must be unique"), failure();
  }
  OpBuilder b(graph);
  return b.getDictionaryAttr({
      b.getNamedAttr("n",b.getI64IntegerAttr(codes.getDimSize(0))),
      b.getNamedAttr("k",b.getI64IntegerAttr(codes.getDimSize(1)*2)),
      b.getNamedAttr("storage_contract",graph->getAttr("storage_contract")),
      b.getNamedAttr("bindings",names),b.getNamedAttr("target",target),
      b.getNamedAttr("arch",arch),
      b.getNamedAttr("ownership",b.getStringAttr("private_outputs_distinct_readonly_inputs")),
      b.getNamedAttr("layout",b.getStringAttr("row_major")),
      b.getNamedAttr("scale_storage",b.getStringAttr("legacy_e8m0_bits"))});
}
static LogicalResult scheduleNativeMXFP4StorageIngest(ModuleOp mod) {
  SmallVector<Operation *> graphs;
  mod.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "tessera.mxfp4_folded_storage") graphs.push_back(op);
  });
  for (Operation *graph : graphs) {
    auto c = nativeMXFP4StorageContract(graph); if (failed(c)) return failure();
    auto fn = graph->getParentOfType<func::FuncOp>();
    auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
    if (fn.getBody().front().getOperations().size() != 2 || !ret ||
        ret.getOperands() != graph->getResults())
      return graph->emitError("MXFP4Storage ingest must return its sole two-result operation");
    OpBuilder b(graph); b.setInsertionPointAfter(graph);
    auto hash = b.getStringAttr(mlir::tessera_contract::nvfp4ContractHash(*c));
    graph->setAttr("schedule.artifact_hash",hash);
    OperationState state(graph->getLoc(),"schedule.artifact");
    state.addAttribute("hash",hash);state.addAttribute("arch",b.getStringAttr("gfx1201"));
    state.addAttribute("shape_key",b.getStringAttr("family=mxfp4_folded_storage"));
    state.addAttribute("contract",*c);b.create(state);
  }
  return success();
}
static LogicalResult lowerNativeMXFP4StorageIngest(ModuleOp mod) {
  SmallVector<schedule::ArtifactOp> records;
  mod.walk([&](schedule::ArtifactOp op) {
    if (op.getShapeKey() == "family=mxfp4_folded_storage") records.push_back(op);
  });
  for (auto record : records) {
    auto fn = record->getParentOfType<func::FuncOp>();
    if (!fn || !fn.getBody().hasOneBlock() || fn.getBody().front().getOperations().size() != 3)
      return record.emitError("MXFP4Storage Schedule lost its isolated Graph entry");
    Operation *graph = &fn.getBody().front().front();
    if (graph->getName().getStringRef() != "tessera.mxfp4_folded_storage")
      return record.emitError("MXFP4Storage Schedule lost its native producer");
    auto c = nativeMXFP4StorageContract(graph);if (failed(c)) return failure();
    auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
    if (!ret || ret.getOperands() != graph->getResults() ||
        record.getArch() != "gfx1201" ||
        record->getAttr("contract") != *c ||
        record.getHash() != mlir::tessera_contract::nvfp4ContractHash(*c) ||
        graph->getAttr("schedule.artifact_hash") != record->getAttr("hash"))
      return record.emitError("MXFP4Storage Schedule contract changed after hashing");
    OpBuilder b(mod.getContext());b.setInsertionPointToEnd(mod.getBody());
    std::string entry = "tessera_mxfp4_storage_" + record.getHash().take_front(16).str();
    if (SymbolTable::lookupSymbolIn(mod,entry)) return record.emitError("MXFP4Storage entry collision");
    auto bytes = MemRefType::get({ShapedType::kDynamic},b.getI8Type());
    SmallVector<Type> types{bytes,bytes,bytes,bytes};
    auto kernel = func::FuncOp::create(b,graph->getLoc(),entry,b.getFunctionType(types,{}));
    auto block = kernel.addEntryBlock();
    b.setInsertionPointToStart(block);
    OperationState tile(graph->getLoc(),"tile.mxfp4_folded_storage_kernel");
    tile.addOperands(block->getArguments());tile.addAttribute("contract",*c);
    tile.addAttribute("artifact_hash",record->getAttr("hash"));b.create(tile);
    func::ReturnOp::create(b,graph->getLoc(),ValueRange{});
    fn.erase();
  }
  return success();
}
}
