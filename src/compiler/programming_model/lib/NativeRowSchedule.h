#pragma once
#include "NativeRowScalarizer.h"
#include "Tessera/IR/RowRegionContract.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/IRMapping.h"

namespace tessera::row_region {
inline mlir::FailureOr<mlir::func::FuncOp> graphRoot(mlir::ModuleOp module) {
  using namespace mlir;
  SmallVector<func::FuncOp> found;
  for (auto fn : module.getOps<func::FuncOp>())
    if (fn->hasAttr("tessera.native.sm120_row_producer") ||
        fn->hasAttr("tessera.native.rocm_row_producer")) found.push_back(fn);
  if (found.empty()) return func::FuncOp{};
  auto reject = [&]() -> FailureOr<func::FuncOp> { return module.emitError("native row Schedule requires its sealed isolated owning source graph"), failure(); };
  if (found.size() != 1 || llvm::range_size(module.getOps<func::FuncOp>()) != 1) return reject();
  auto target = module->getAttrOfType<StringAttr>("tessera.target");
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  auto fn = found.front();
  if (fn.getNumArguments() != 2 || (fn.getNumResults() != 1 && fn.getNumResults() != 2) || !fn.getBody().hasOneBlock()) return reject();
  bool rocm = fn->hasAttr("tessera.native.rocm_row_producer");
  if (rocm && (fn->hasAttr("tessera.native.sm120_row_producer") || fn.getNumResults() != 1)) return reject();
  auto c = fn->getAttrOfType<DictionaryAttr>(rocm ? "tessera.native.rocm_row_producer" : "tessera.native.sm120_row_producer");
  auto type = dyn_cast<RankedTensorType>(fn.getArgumentTypes().front());
  auto rows = c ? c.getAs<IntegerAttr>("rows") : IntegerAttr{};
  auto columns = c ? c.getAs<IntegerAttr>("columns") : IntegerAttr{};
  auto schema = c ? c.getAs<IntegerAttr>("schema") : IntegerAttr{};
  auto storage = c ? c.getAs<StringAttr>("storage") : StringAttr{};
  auto accum = c ? c.getAs<StringAttr>("accum") : StringAttr{};
  auto hash = c ? c.getAs<StringAttr>("body_hash") : StringAttr{};
  bool owningTarget = target && arch && (rocm
      ? ((arch.getValue() == "gfx1151" || arch.getValue() == "gfx1201") &&
         target.getValue() == (llvm::Twine("rocm_") + arch.getValue()).str())
      : (target.getValue() == "nvidia_sm120" && arch.getValue() == "sm_120"));
  if (!owningTarget ||
      !c || c.size() != 6 || !type || type.getRank() != 2 ||
      !rows || rows.getInt() != type.getDimSize(0) || !columns || columns.getInt() != type.getDimSize(1) ||
      !schema || schema.getInt() != (rocm ? 7 : fn.getNumResults() == 1 ? 6 : 1) || !storage || storage.getValue() != (type.getElementType().isF16() ? "f16" : "bf16") ||
      !accum || accum.getValue() != "f32" || !hash || hash.getValue() != sourceDigest(fn)) return reject();
  RowScalarizer scalarizer(fn);
  if (failed(scalarizer.validate())) return failure();
  return fn;
}
inline int64_t rowSchema(mlir::func::FuncOp fn) {
  return fn->hasAttr("tessera.native.rocm_row_producer") ? 7 : fn.getNumResults() == 1 ? 6 : 1;
}
inline llvm::StringRef rowArch(mlir::func::FuncOp fn) {
  return fn->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::StringAttr>("tessera.arch").getValue();
}
inline std::string artifactDigest(mlir::func::FuncOp fn) {
  std::string text = sourceDigest(fn) + (fn.getNumResults() == 1
      ? ";row_ssa_schema=6;serial_row;threads=128;typed_views=3;outputs=1"
      : ";row_ssa_schema=1;serial_row;threads=128;typed_views=4;outputs=2");
  if (rowSchema(fn) == 7)
    text = sourceDigest(fn) + ";row_ssa_schema=7;serial_row;threads=128;typed_views=3;outputs=1;arch=" + rowArch(fn).str();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}
inline mlir::DictionaryAttr artifactAttributes(mlir::func::FuncOp fn) {
  using namespace mlir;
  Builder b(fn.getContext());
  auto type = cast<RankedTensorType>(fn.getArgumentTypes()[0]);
  std::string shape = "family=native_row_ssa;rows=" + std::to_string(type.getDimSize(0)) +
      ";columns=" + std::to_string(type.getDimSize(1)) + ";storage=" +
      (type.getElementType().isF16() ? "f16" : "bf16") + ";inputs=2;outputs=" + std::to_string(fn.getNumResults());
  return b.getDictionaryAttr({
    b.getNamedAttr("hash", b.getStringAttr(artifactDigest(fn))),
    b.getNamedAttr("arch", b.getStringAttr(rowArch(fn))),
    b.getNamedAttr("shape_key", b.getStringAttr(shape)),
    b.getNamedAttr("numeric_policy", b.getStringAttr("explicit_fp32;storage_rounding_at_graph_casts")),
    b.getNamedAttr("tile", b.getDictionaryAttr({
      b.getNamedAttr("source_hash", b.getStringAttr(sourceDigest(fn))),
      b.getNamedAttr("algorithm", b.getStringAttr("serial_row_ssa")),
      b.getNamedAttr("workgroup_size", b.getI64IntegerAttr(128)),
      b.getNamedAttr("schema", b.getI64IntegerAttr(rowSchema(fn)))}))});
}
inline mlir::LogicalResult schedule(mlir::ModuleOp module, bool &selected) {
  using namespace mlir;
  auto root = graphRoot(module);
  if (failed(root)) return failure();
  selected = bool(*root);
  if (!selected) return success();
  for (auto &op : module.getBody()->getOperations())
    if (&op != root->getOperation()) return module.emitError("native row graph is already scheduled or contains unrelated operations");
  if (rowSchema(*root) == 7) {
    std::string graph; llvm::raw_string_ostream stream(graph);
    module.print(stream);stream.flush();
    module->setAttr("tessera.rocm.source_row_graph",StringAttr::get(module.getContext(),graph+"\n"));
  }
  OpBuilder b(module.getContext()); b.setInsertionPointToEnd(module.getBody());
  OperationState state(root->getLoc(), "schedule.artifact");
  state.addAttributes(artifactAttributes(*root).getValue()); b.create(state);
  return success();
}
inline mlir::FailureOr<mlir::LLVM::LLVMFuncOp> buildKernel(mlir::ModuleOp module, mlir::func::FuncOp graph) {
  using namespace mlir;
  RowScalarizer scalarizer(graph);
  auto lowered = scalarizer.lower(module);
  if (failed(lowered)) return failure();
  auto type = cast<RankedTensorType>(graph.getArgumentTypes()[0]);
  OpBuilder b(module.getContext()); Location loc = graph.getLoc();
  b.setInsertionPointToEnd(module.getBody());
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  unsigned buffers = 2 + graph.getNumResults();
  SmallVector<Type> arguments(buffers, ptr);
  arguments.append({b.getI64Type(), b.getI64Type()});
  auto kernel = LLVM::LLVMFuncOp::create(b, loc, "tessera_native_row_" + sourceDigest(graph).substr(0,16),
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(b.getContext()),
          arguments,false));
  auto *body = kernel.addEntryBlock(b); b.setInsertionPointToStart(body);
  auto coordinate = [&](Value value) -> Value { return arith::IndexCastOp::create(b,loc,b.getI64Type(),value); };
  Value tid = coordinate(gpu::ThreadIdOp::create(b,loc,gpu::Dimension::x));
  Value bid = coordinate(gpu::BlockIdOp::create(b,loc,gpu::Dimension::x));
  Value threads = coordinate(gpu::BlockDimOp::create(b,loc,gpu::Dimension::x));
  Value row = arith::AddIOp::create(b,loc,arith::MulIOp::create(b,loc,bid,threads),tid);
  Value expectedRows = arith::ConstantIntOp::create(b,loc,type.getDimSize(0),64);
  Value expectedColumns = arith::ConstantIntOp::create(b,loc,type.getDimSize(1),64);
  Value rm = arith::CmpIOp::create(b,loc,arith::CmpIPredicate::eq,body->getArgument(buffers),expectedRows);
  Value cm = arith::CmpIOp::create(b,loc,arith::CmpIPredicate::eq,body->getArgument(buffers+1),expectedColumns);
  auto guarded = scf::IfOp::create(b,loc,arith::AndIOp::create(b,loc,rm,cm),false);
  {
    OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(&guarded.getThenRegion().front());
    IRMapping mapping;
    for (unsigned i=0;i<buffers;++i) mapping.map(lowered->getArgument(i),body->getArgument(i));
    mapping.map(lowered->getArgument(buffers),row);
    for (Operation &op : lowered->getBody().front().without_terminator()) b.clone(op,mapping);
  }
  b.setInsertionPointToEnd(body); LLVM::ReturnOp::create(b,loc,ValueRange{});
  lowered->erase(); return kernel;
}
inline mlir::LogicalResult lower(mlir::ModuleOp module, bool &selected) {
  using namespace mlir;
  auto root = graphRoot(module);
  if (failed(root)) return failure();
  selected = bool(*root);
  if (!selected) return success();
  Operation *artifact = nullptr;
  for (Operation &op : module.getBody()->getOperations()) {
    if (&op == root->getOperation()) continue;
    if (op.getName().getStringRef() != "schedule.artifact" || artifact)
      return module.emitError("native row Tile projection requires exactly one checked Schedule artifact");
    artifact = &op;
  }
  if (!artifact || artifact->getAttrDictionary() != artifactAttributes(*root))
    return module.emitError("native row Schedule differs from its current Graph, numerical policy or physical recipe");
  OwningOpRef<ModuleOp> temporary(ModuleOp::create(module.getLoc()));
  auto kernel = buildKernel(*temporary,*root);
  if (failed(kernel)) return failure();
  auto type = cast<RankedTensorType>(root->getArgumentTypes()[0]);
  OpBuilder b(artifact);
  OperationState state(module.getLoc(),"tile.row_region_kernel");
  state.addAttribute("name",b.getStringAttr(kernel->getName()));
  state.addAttribute("arch",b.getStringAttr(rowArch(*root)));
  state.addAttribute("rows",b.getI64IntegerAttr(type.getDimSize(0)));
  state.addAttribute("columns",b.getI64IntegerAttr(type.getDimSize(1)));
  state.addAttribute("storage",TypeAttr::get(type.getElementType()));
  state.addAttribute("workgroup_size",b.getI64IntegerAttr(128));
  state.addAttribute("schema",b.getI64IntegerAttr(rowSchema(*root)));
  state.addAttribute("artifact_hash",b.getStringAttr(artifactDigest(*root)));
  state.addAttribute("source_hash",b.getStringAttr(sourceDigest(*root)));
  state.addAttribute("body_hash",b.getStringAttr(digest(*kernel)));
  state.addRegion(); auto *carrier=b.create(state);
  carrier->getRegion(0).push_back(new Block);
  kernel->getOperation()->moveBefore(&carrier->getRegion(0).front(),carrier->getRegion(0).front().end());
  root->erase(); artifact->erase();
  return verifyKernel(carrier);
}
} // namespace tessera::row_region
