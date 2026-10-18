// Shared native SSA outlining of the RMSNorm/ReLU source pullback.
// No derivative formula or executable kernel is reconstructed here.
#pragma once
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/Base64.h"
#include <functional>

namespace tessera {
static mlir::LogicalResult emitNativeSourceReverseProgram(
    mlir::ModuleOp module, llvm::StringRef originalSource, bool rocm = false) {
  using namespace mlir;
  auto reject = [&]() { return module.emitError(
      "native source reverse outline requires its owning static explicit-fp32 RMSNorm/ReLU pullback"); };
  auto target = module->getAttrOfType<StringAttr>("tessera.target");
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  SmallVector<func::FuncOp> functions;
  func::FuncOp backward;
  for (auto fn : module.getOps<func::FuncOp>()) {
    functions.push_back(fn);
    auto role = fn->getAttrOfType<StringAttr>("tessera.autodiff.role");
    if (role && role.getValue() == "backward") backward = fn;
  }
  bool owningTarget = target && arch && (rocm
      ? ((arch.getValue() == "gfx1151" || arch.getValue() == "gfx1201") &&
         target.getValue() == (llvm::Twine("rocm_") + arch.getValue()).str())
      : (target.getValue() == "nvidia_sm120" && arch.getValue() == "sm_120"));
  if (!owningTarget || functions.size() != 2 || !backward ||
      !backward.getBody().hasOneBlock() || originalSource.empty()) return reject();
  auto parentRef = backward->getAttrOfType<FlatSymbolRefAttr>("tessera.autodiff.forward");
  auto parent = parentRef ? module.lookupSymbol<func::FuncOp>(parentRef.getValue()) : func::FuncOp{};
  if (!parent || parent == backward || !parent.getBody().hasOneBlock() ||
      parent.getNumArguments() != 2 || parent.getNumResults() != 1 ||
      backward.getNumArguments() != 3 || backward.getNumResults() != 2) return reject();
  SmallVector<Operation *> primal;
  for (auto &op : parent.getBody().front().without_terminator()) primal.push_back(&op);
  if (primal.size() != 3 || primal[0]->getName().getStringRef() != "tessera.rmsnorm" ||
      primal[1]->getName().getStringRef() != "tessera.relu" ||
      primal[2]->getName().getStringRef() != "tessera.matmul" ||
      primal[0]->getNumOperands() != 1 || primal[1]->getNumOperands() != 1 ||
      primal[2]->getNumOperands() != 2 || primal[0]->getNumResults() != 1 ||
      primal[1]->getNumResults() != 1 || primal[2]->getNumResults() != 1 ||
      primal[1]->getOperand(0) != primal[0]->getResult(0) ||
      primal[2]->getOperand(0) != primal[1]->getResult(0)) return reject();
  // Preserve the forward numerical/orientation policy instead of silently
  // interpreting a fused, approximate, or transposed primal as this profile.
  for (auto attr : primal[2]->getAttrs()) {
    auto key=attr.getName().getValue();
    if (key=="tessera.effect_kind" || key=="tessera.autodiff.activity") continue;
    if (key=="output_dtype") {
      auto value=dyn_cast<StringAttr>(attr.getValue());
      if (!value || value.getValue()!="fp32") return reject();
    } else if (key=="transposeA" || key=="transposeB") {
      auto value=dyn_cast<BoolAttr>(attr.getValue());
      if (!value || value.getValue()) return reject();
    } else if (key=="activation" || key=="rhs_storage_order") {
      auto value=dyn_cast<StringAttr>(attr.getValue());
      if (!value || value.getValue()!=(key=="activation"?"none":"row_major")) return reject();
    } else if (key=="numeric_policy") {
      auto value=dyn_cast<DictionaryAttr>(attr.getValue());
      auto accum=value ? value.getAs<StringAttr>("accum") : StringAttr{};
      if (!value || value.size()!=1 || !accum || accum.getValue()!="fp32") return reject();
    } else return reject();
  }
  auto x = dyn_cast<BlockArgument>(primal[0]->getOperand(0));
  auto b = dyn_cast<BlockArgument>(primal[2]->getOperand(1));
  if (!x || !b || x.getOwner() != &parent.getBody().front() ||
      b.getOwner() != x.getOwner() || x == b) return reject();
  auto wrt = parent->getAttrOfType<ArrayAttr>("tessera.autodiff.wrt_indices");
  auto role = wrt && wrt.size() == 1 ? dyn_cast<IntegerAttr>(wrt[0]) : IntegerAttr{};
  if (!role || role.getInt() != x.getArgNumber()) return reject();
  auto source = dyn_cast<RankedTensorType>(x.getType());
  auto rhs = dyn_cast<RankedTensorType>(b.getType());
  auto result = dyn_cast<RankedTensorType>(parent.getResultTypes()[0]);
  if (!source || !rhs || !result || source.getRank() != 2 || rhs.getRank() != 2 ||
      result.getRank() != 2 || !source.hasStaticShape() || !rhs.hasStaticShape() ||
      !result.hasStaticShape() || source.getEncoding() || rhs.getEncoding() ||
      result.getEncoding() || (!source.getElementType().isF16() &&
      !source.getElementType().isBF16()) || rhs.getElementType() != source.getElementType() ||
      !result.getElementType().isF32() ||
      primal[0]->getResult(0).getType() != source || primal[1]->getResult(0).getType() != source ||
      rhs.getDimSize(0) != source.getDimSize(1) ||
      result.getShape() != ArrayRef<int64_t>({source.getDimSize(0), rhs.getDimSize(1)}))
    return reject();
  int64_t m = source.getDimSize(0), k = source.getDimSize(1), n = rhs.getDimSize(1);
  if (m <= 0 || m > 65535LL*16 || n <= 0 || n > INT32_MAX || k <= 0 ||
      k > INT32_MAX || m > INT64_MAX/k/2 || k > INT64_MAX/n/2 ||
      m > INT64_MAX/n/4) return reject();
  for (unsigned i=0;i<2;++i)
    if (backward.getArgument(i).getType() != parent.getArgument(i).getType() ||
        (parent.getArgAttrDict(i) && !parent.getArgAttrDict(i).empty()) ||
        (backward.getArgAttrDict(i) && !backward.getArgAttrDict(i).empty()))
      return reject();
  if (backward.getArgument(2).getType() != result ||
      backward.getResultTypes()[x.getArgNumber()] != source) return reject();
  auto ret = dyn_cast<func::ReturnOp>(backward.getBody().front().getTerminator());
  if (!ret || ret.getNumOperands() != 2) return reject();
  Value gradient = ret.getOperand(x.getArgNumber());
  Value sourceRoot = backward.getArgument(x.getArgNumber());
  Operation *cut = nullptr;
  llvm::SmallPtrSet<Operation *,32> live;
  bool usedSource = false;
  std::function<bool(Value)> visit = [&](Value value) {
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg != sourceRoot) return false;
      usedSource = true; return true;
    }
    auto op = value.getDefiningOp();
    if (!op || op->getBlock() != &backward.getBody().front() ||
        op->getNumRegions() || !isMemoryEffectFree(op)) return false;
    if (op->getName().getStringRef() == "tessera.matmul") {
      if (cut && cut != op) return false;
      auto ta = op->getAttrOfType<BoolAttr>("transposeA");
      auto tb = op->getAttrOfType<BoolAttr>("transposeB");
      if (op->getNumOperands()!=2 || op->getNumResults()!=1 ||
          op->getOperand(0)!=backward.getArgument(2) ||
          op->getOperand(1)!=backward.getArgument(b.getArgNumber()) ||
          op->getResult(0).getType()!=source || (ta && ta.getValue()) ||
          !tb || !tb.getValue()) return false;
      cut=op; return true;
    }
    if (!live.insert(op).second) return true;
    auto name=op->getName().getStringRef();
    if (name!="tessera.rmsnorm" && name!="tessera.relu" && name!="tessera.cast" &&
        name!="tessera.normalization_stats" && name!="tessera.broadcast_in_dim" &&
        name!="tessera.mul" && name!="tessera.reduce" && name!="tessera.sub" &&
        name!="tessera.add" && name!="arith.constant" &&
        name!="tessera.compare_scalar" && name!="tessera.masked_fill") return false;
    for (Value input : op->getOperands()) if (!visit(input)) return false;
    return true;
  };
  if (!visit(gradient) || !cut || !usedSource) return reject();
  std::string paired; llvm::raw_string_ostream pairedStream(paired);
  module.print(pairedStream); pairedStream.flush();
  OpBuilder builder(module.getContext());
  llvm::json::Array graphs, steps;
  for (unsigned index=0;index<2;++index) {
    builder.setInsertionPointToEnd(module.getBody());
    std::string symbol=(backward.getSymName()+"__reverse_member_"+llvm::Twine(index)).str();
    if (module.lookupSymbol(symbol)) return reject();
    SmallVector<Type> inputs = index==0 ? SmallVector<Type>{result,rhs} : SmallVector<Type>{source,source};
    auto member=func::FuncOp::create(builder,backward.getLoc(),symbol,
        builder.getFunctionType(inputs,{source}));
    member.setPrivate();
    auto block=member.addEntryBlock(); OpBuilder body(block,block->begin()); IRMapping map;
    if (index==0) {
      map.map(cut->getOperand(0),block->getArgument(0));
      map.map(cut->getOperand(1),block->getArgument(1));
      auto product=body.clone(*cut,map);
      product->removeAttr("tessera.autodiff.activity");
      product->setAttr("rhs_storage_order",body.getStringAttr("row_major"));
      func::ReturnOp::create(body,member.getLoc(),product->getResults());
    } else {
      map.map(sourceRoot,block->getArgument(0));
      map.map(cut->getResult(0),block->getArgument(1));
      for (auto &op : backward.getBody().front().without_terminator())
        if (live.contains(&op)) {
          auto cloned=body.clone(op,map); cloned->removeAttr("tessera.autodiff.activity");
        }
      func::ReturnOp::create(body,member.getLoc(),map.lookup(gradient));
      member->setAttr(rocm ? "tessera.native.rocm_row_producer" : "tessera.native.sm120_row_producer",builder.getDictionaryAttr({
        builder.getNamedAttr("schema",builder.getI64IntegerAttr(rocm ? 7 : 6)),
        builder.getNamedAttr("rows",builder.getI64IntegerAttr(m)),
        builder.getNamedAttr("columns",builder.getI64IntegerAttr(k)),
        builder.getNamedAttr("storage",builder.getStringAttr(source.getElementType().isF16()?"f16":"bf16")),
        builder.getNamedAttr("accum",builder.getStringAttr("f32")),
        builder.getNamedAttr("body_hash",builder.getStringAttr(nativeSM120RowBodyHash(member)))}));
    }
    if (failed(verify(member))) return failure();
    auto projected=ModuleOp::create(module.getLoc());
    projected->setAttr("tessera.target",target); projected->setAttr("tessera.arch",arch);
    auto child=cast<func::FuncOp>(member->clone()); child.setPublic();
    projected.getBody()->push_back(child);
    std::string text;llvm::raw_string_ostream stream(text);projected.print(stream);stream.flush();
    graphs.push_back(text+"\n"); projected.erase();
    llvm::json::Array inputsIds = index==0 ? llvm::json::Array{int64_t(2),int64_t(b.getArgNumber())}
        : llvm::json::Array{int64_t(x.getArgNumber()),int64_t(3)};
    steps.push_back(llvm::json::Object{{"step",int64_t(index)},{"member",symbol},
        {"operation",index==0?"tessera.matmul":"native_row_region"},
        {"inputs",std::move(inputsIds)},{"outputs",llvm::json::Array{int64_t(3+index)}}});
  }
  SmallVector<Type> types(backward.getArgumentTypes()); types.append({source,source});
  llvm::json::Array buffers;
  for (unsigned i=0;i<5;++i) {
    auto type=cast<RankedTensorType>(types[i]); uint64_t bytes=type.getElementTypeBitWidth()/8;
    llvm::json::Array shape;for (auto dim:type.getShape()) {bytes*=dim;shape.push_back(dim);}
    int64_t write=i<3?-1:i==3?0:1;
    int64_t read=i==x.getArgNumber()?1:i<3?0:i==3?1:2;
    std::string storage;llvm::raw_string_ostream stream(storage);type.getElementType().print(stream);stream.flush();
    buffers.push_back(llvm::json::Object{{"id",int64_t(i)},{"shape",std::move(shape)},
        {"storage",storage},{"bytes",int64_t(bytes)},
        {"ownership",i<3?"readonly_input":i==3?"private_scratch":"returned_output"},
        {"first_write",write},{"last_read",read}});
  }
  llvm::json::Object plan{{"schema",rocm ? "tessera.native.rocm_source_reverse_outline.v1" : "tessera.native.sm120_source_reverse_outline.v1"},
    {"execution_state","artifact_only"},{"kind","paired_source_reverse"},
    {"source_graph_ir",originalSource.str()},{"paired_graph_ir",paired},
    {"root",backward.getSymName().str()},
    {"role_indices",llvm::json::Array{int64_t(x.getArgNumber()),int64_t(b.getArgNumber()),int64_t(2)}},
    {"buffers",std::move(buffers)},{"steps",std::move(steps)},
    {"outputs",llvm::json::Array{int64_t(4)}},{"member_graphs",std::move(graphs)}};
  if (rocm) { plan["target"] = target.getValue().str(); plan["architecture"] = arch.getValue().str(); }
  std::string json;llvm::raw_string_ostream stream(json);stream<<llvm::json::Value(std::move(plan));stream.flush();
  module->setAttr(rocm ? "tessera.native.rocm_reverse_program_json" : "tessera.native.sm120_reverse_program_json",
      builder.getStringAttr(llvm::encodeBase64(json)));
  return success();
}
} // namespace tessera
