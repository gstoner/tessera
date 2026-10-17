#pragma once
#include "Tessera/IR/TesseraOps.h"
#include "Tessera/IR/SliceContract.h"
#include "Tessera/IR/TileBufferViewContract.h"
#include "Tessera/IR/RowRegionContract.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Verifier.h"
#include <functional>

namespace tessera::gate_up_view {
using namespace mlir;
namespace tile = tessera::tile;
struct Source {
  func::FuncOp graph;
  RankedTensorType inputType, outputType;
  func::ReturnOp ret;
  std::string inputOrder;
};
inline FailureOr<Source> validateSource(ModuleOp source) {
  SmallVector<func::FuncOp> graphs;
  for (auto fn : source.getOps<func::FuncOp>()) graphs.push_back(fn);
  if (graphs.size() != 1) return failure();
  auto graph = graphs.front();
  if (graph.getNumArguments() != 1 || graph.getNumResults() != 1 ||
      !graph.getBody().hasOneBlock()) return failure();
  auto inputType = dyn_cast<RankedTensorType>(graph.getArgument(0).getType());
  auto outputType = dyn_cast<RankedTensorType>(graph.getResultTypes()[0]);
  if (!inputType || !outputType || inputType.getRank() != 2 || outputType.getRank() != 2 ||
      !inputType.hasStaticShape() || !outputType.hasStaticShape() ||
      (!inputType.getElementType().isF32() && !inputType.getElementType().isBF16()) ||
      inputType.getElementType() != outputType.getElementType() ||
      inputType.getEncoding() || outputType.getEncoding()) return failure();
  for (auto type : {inputType, outputType}) {
    for (auto extent : type.getShape()) if (extent <= 0 || extent > INT32_MAX) return failure();
    if (type.getDimSize(0) > INT64_MAX / 4 / type.getDimSize(1)) return failure();
  }
  if (outputType.getDimSize(0) > INT64_MAX / outputType.getDimSize(1)) return failure();
  if (outputType.getDimSize(0) * outputType.getDimSize(1) > (int64_t(1) << 31) / 12) return failure();
  std::string inputOrder = "row_major";
  if (auto attributes = graph.getArgAttrDict(0))
    for (NamedAttribute attr : attributes) {
      if (attr.getName() == "tessera.layout") {
        auto order = dyn_cast<StringAttr>(attr.getValue());
        if (!order || order.getValue() != "row_major") return failure();
        inputOrder = order.getValue().str();
      } else if (attr.getName() == "tessera.dim_names") {
        auto names = dyn_cast<ArrayAttr>(attr.getValue());
        if (!names || names.size() != 2) return failure();
        for (auto [axis, name] : llvm::enumerate(names)) {
          auto text = dyn_cast<StringAttr>(name); int64_t extent;
          if (!text || text.getValue().empty() ||
              (!text.getValue().getAsInteger(10, extent) && extent != inputType.getDimSize(axis))) return failure();
        }
      } else return failure();
    }
  if (auto attributes = graph.getResultAttrDict(0); attributes && !attributes.empty()) return failure();
  auto ret = dyn_cast<func::ReturnOp>(graph.getBody().front().back());
  if (!ret || ret.getNumOperands() != 1) return failure();
  bool bf16 = inputType.getElementType().isBF16();
  unsigned slices = 0, products = 0, casts = 0;
  for (Operation &op : graph.getBody().front().without_terminator()) {
    auto name = op.getName().getStringRef();
    bool slice = name == "tessera.slice" || name == "tessera.dynamic_slice";
    if (slice) {
      if (!tessera::slice_contract::staticBounds(&op)) return failure();
      ++slices;
    } else if (name == "tessera.silu_mul") {
      if (op.getNumOperands() != 2 || op.getNumResults() != 1 ||
          op.getResult(0).getType() != RankedTensorType::get(outputType.getShape(), Float32Type::get(source.getContext())))
        return failure();
      ++products;
    } else if (name == "tessera.cast" && bf16) {
      if (op.getNumOperands() != 1 || op.getNumResults() != 1) return failure();
      auto from = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
      auto to = dyn_cast<RankedTensorType>(op.getResult(0).getType());
      auto dtype = op.getAttrOfType<StringAttr>("dtype");
      if (!from || !to || from.getShape() != outputType.getShape() || to.getShape() != outputType.getShape() ||
          from.getEncoding() || to.getEncoding() || !dtype ||
          !((from.getElementType().isBF16() && to.getElementType().isF32() && dtype.getValue() == "fp32") ||
            (from.getElementType().isF32() && to.getElementType().isBF16() && dtype.getValue() == "bf16")))
        return failure();
      ++casts;
    } else return failure();
    for (NamedAttribute attr : op.getAttrs()) {
      if (slice && (attr.getName() == "start_indices" || attr.getName() == "slice_sizes")) continue;
      if (name == "tessera.cast" && attr.getName() == "dtype") continue;
      if (attr.getName() == "tessera.effect_kind" &&
          attr.getValue() == StringAttr::get(source.getContext(), "pure")) continue;
      return failure();
    }
  }
  auto root = ret.getOperand(0).getDefiningOp();
  if (bf16) {
    if (!root || root->getName().getStringRef() != "tessera.cast" ||
        root->getResult(0).getType() != outputType || casts != 3) return failure();
    root = root->getOperand(0).getDefiningOp();
  }
  if (slices != 2 || products != 1 || (!bf16 && casts) || !root ||
      root->getName().getStringRef() != "tessera.silu_mul") return failure();

  // This runtime-row profile is the named gate/up split, not an arbitrary
  // crop. Each original Graph still carries its own exact source-shape guard.
  if (inputType.getDimSize(0) != outputType.getDimSize(0) ||
      inputType.getDimSize(1) != 2 * outputType.getDimSize(1)) return failure();
  int64_t origins[2] = {-1, -1};
  for (unsigned i = 0; i < 2; ++i) {
    auto producer = root->getOperand(i).getDefiningOp();
    if (bf16) {
      if (!producer || producer->getName().getStringRef() != "tessera.cast" ||
          !cast<RankedTensorType>(producer->getResult(0).getType()).getElementType().isF32())
        return failure();
      producer = producer->getOperand(0).getDefiningOp();
    }
    if (!producer || producer->getNumOperands() != 1 || producer->getOperand(0) != graph.getArgument(0)) return failure();
    auto bounds = tessera::slice_contract::staticBounds(producer);
    if (!bounds || bounds->sizes != outputType.getShape() || bounds->offsets[0] != 0 ||
        (bounds->offsets[1] != 0 && bounds->offsets[1] != outputType.getDimSize(1))) return failure();
    origins[i] = bounds->offsets[1];
  }
  if (origins[0] == origins[1]) return failure();
  return Source{graph,inputType,outputType,ret,inputOrder};
}
inline FailureOr<ModuleOp> buildScalarModule(ModuleOp source) {
  auto validated = validateSource(source);
  if (failed(validated)) return failure();
  auto graph = validated->graph;
  auto inputType = validated->inputType, outputType = validated->outputType;
  auto ret = validated->ret;
  auto inputOrder = validated->inputOrder;
  auto loc = graph.getLoc();
  OpBuilder b(source.getContext());
  ModuleOp output = ModuleOp::create(loc);
  b.setInsertionPointToEnd(output.getBody());
  auto ptr = LLVM::LLVMPointerType::get(source.getContext());
  auto functionType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(source.getContext()),
      {ptr, ptr, b.getI64Type(), b.getI64Type()}, false);
  auto fn = LLVM::LLVMFuncOp::create(b, loc, "gate_up_scalar", functionType);
  Block *entry = fn.addEntryBlock(b); b.setInsertionPointToStart(entry);
  auto c = [&](int64_t value) -> Value { return arith::ConstantIntOp::create(b, loc, value, 64); };
  Value columns = c(outputType.getDimSize(1));
  Value elements = entry->getArgument(3);
  Value rows = arith::DivUIOp::create(b, loc, elements, columns);
  Value inputRows = rows, inputColumns = c(inputType.getDimSize(1));
  auto readType = tile::BufferViewType::get(source.getContext(), inputType.getElementType(), "read", inputOrder);
  auto writeType = tile::BufferViewType::get(source.getContext(), outputType.getElementType(), "write", "row_major");
  Value parent = tile::BufferViewOp::create(b, loc, readType, entry->getArgument(0), inputRows, inputColumns, inputOrder == "row_major" ? inputColumns : inputRows);
  Value out = tile::BufferViewOp::create(b, loc, writeType, entry->getArgument(1), rows, columns, columns);
  Value gid = entry->getArgument(2);
  Value active = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, gid, elements);
  auto guard = scf::IfOp::create(b, loc, active, false);
  b.setInsertionPointToStart(&guard.getThenRegion().front());
  Value row = arith::DivUIOp::create(b, loc, gid, columns);
  Value column = arith::RemUIOp::create(b, loc, gid, columns);
  auto offset = [&](Value index, int64_t delta) -> Value {
    return delta ? arith::AddIOp::create(b, loc, index, c(delta)).getResult() : index;
  };
  std::function<FailureOr<Value>(Value, Value, Value)> scalar;
  scalar = [&](Value value, Value r, Value col) -> FailureOr<Value> {
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg != graph.getArgument(0)) return failure();
      return Value(tile::LoadElementOp::create(b, loc, inputType.getElementType(), parent, r, col));
    }
    Operation *op = value.getDefiningOp();
    if (!op) return failure();
    auto name = op->getName().getStringRef();
    if (name == "tessera.slice" || name == "tessera.dynamic_slice") {
      auto bounds = tessera::slice_contract::staticBounds(op);
      if (!bounds || bounds->offsets.size() != 2) return failure();
      return scalar(op->getOperand(0), offset(r, bounds->offsets[0]), offset(col, bounds->offsets[1]));
    }
    if (name == "tessera.cast") {
      auto operand = scalar(op->getOperand(0), r, col);
      if (failed(operand)) return failure();
      auto element = cast<RankedTensorType>(value.getType()).getElementType();
      if (element.isF32()) return Value(arith::ExtFOp::create(b, loc, element, *operand));
      return Value(arith::TruncFOp::create(b, loc, element, *operand));
    }
    if (name != "tessera.silu_mul") return failure();
    auto lhs = scalar(op->getOperand(0), r, col), rhs = scalar(op->getOperand(1), r, col);
    if (failed(lhs) || failed(rhs)) return failure();
    Value one = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1.0));
    Value exponent = math::ExpOp::create(b, loc, arith::NegFOp::create(b, loc, *lhs));
    Value activated = arith::DivFOp::create(b, loc, *lhs, arith::AddFOp::create(b, loc, one, exponent));
    return Value(arith::MulFOp::create(b, loc, activated, *rhs));
  };
  auto value = scalar(ret.getOperand(0), row, column);
  if (failed(value)) { output->destroy(); return failure(); }
  tile::StoreElementOp::create(b, loc, out, row, column, *value);
  b.setInsertionPointAfter(guard); LLVM::ReturnOp::create(b, loc, ValueRange{});
  return output;
}

inline FailureOr<func::FuncOp> graphRoot(ModuleOp module) {
  auto marker = module->getAttr("tessera.native.gate_up_view");
  if (!marker) return func::FuncOp{};
  auto reject = [&]() -> FailureOr<func::FuncOp> {
    module.emitError("native gate/up view requires its gfx1201 original source contract");
    return failure();
  };
  auto target = module->getAttrOfType<StringAttr>("tessera.target");
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  if (!isa<UnitAttr>(marker) || !target || target.getValue() != "rocm" ||
      !arch || arch.getValue() != "gfx1201" ||
      llvm::range_size(module.getOps<func::FuncOp>()) != 1) return reject();
  for (NamedAttribute attr : module->getAttrs()) {
    StringRef n = attr.getName().strref();
    if (n != "tessera.native.gate_up_view" && n != "tessera.target" && n != "tessera.arch" &&
        n != "tessera.ir.version" && n != "tessera.frontend.authority" &&
        n != "tessera.launch_bindings") return reject();
  }
  if (auto raw = module->getAttr("tessera.launch_bindings")) {
    auto names = dyn_cast<ArrayAttr>(raw);
    if (!names || names.size() != 2 || !isa<StringAttr>(names[0]) || !isa<StringAttr>(names[1]) ||
        cast<StringAttr>(names[0]).getValue().empty() || cast<StringAttr>(names[1]).getValue().empty() ||
        names[0] == names[1]) return reject();
  }
  if (auto version = module->getAttr("tessera.ir.version")) {
    auto text = dyn_cast<StringAttr>(version);
    if (!text || text.getValue() != "1.0") return reject();
  }
  auto fn = *module.getOps<func::FuncOp>().begin();
  for (NamedAttribute attr : fn->getAttrs()) {
    StringRef n = attr.getName().strref();
    if (n != "sym_name" && n != "function_type" && n != "arg_attrs" && n != "res_attrs" &&
        n != "tessera.frontend.authority" && n != "tessera.structured_cfg.schema" &&
        n != "tessera.structured_cfg.digest" && n != "tessera.structured_cfg.blocks") return reject();
  }
  return fn;
}
inline std::string sourceHash(func::FuncOp fn) { return row_region::sourceDigest(fn); }
inline bool bf16Profile(func::FuncOp fn) {
  return cast<RankedTensorType>(fn.getResultTypes()[0]).getElementType().isBF16();
}
inline std::string artifactHash(func::FuncOp fn) {
  std::string text = sourceHash(fn) + (bf16Profile(fn)
      ? ";gate_up_view_schema=3;flat_gid;threads=256;parent=1;output=1;bf16_f32_compute"
      : ";gate_up_view_schema=2;flat_gid;threads=256;parent=1;output=1;f32");
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}
inline DictionaryAttr artifactAttributes(func::FuncOp fn) {
  Builder b(fn.getContext());
  auto shape = cast<RankedTensorType>(fn.getResultTypes()[0]).getShape();
  return b.getDictionaryAttr({
    b.getNamedAttr("hash", b.getStringAttr(artifactHash(fn))),
    b.getNamedAttr("arch", b.getStringAttr("gfx1201")),
    b.getNamedAttr("shape_key", b.getStringAttr("family=gate_up_view;rows=" + std::to_string(shape[0]) + ";columns=" + std::to_string(shape[1]) + (bf16Profile(fn) ? ";storage=bf16" : ";storage=f32"))),
    b.getNamedAttr("numeric_policy", b.getStringAttr(bf16Profile(fn) ? "bf16_load;explicit_fp32;source_order_silu_mul;bf16_rne_store" : "explicit_fp32;source_order_silu_mul")),
    b.getNamedAttr("tile", b.getDictionaryAttr({
      b.getNamedAttr("source_hash", b.getStringAttr(sourceHash(fn))),
      b.getNamedAttr("algorithm", b.getStringAttr("flat_pointwise_views")),
      b.getNamedAttr("schema", b.getI64IntegerAttr(bf16Profile(fn) ? 3 : 2)),
      b.getNamedAttr("workgroup_size", b.getI64IntegerAttr(256))}))});
}
inline LogicalResult schedule(ModuleOp module, bool &selected) {
  auto root = graphRoot(module);
  if (failed(root)) return failure();
  selected = bool(*root);
  if (!selected) return success();
  if (module.getBody()->getOperations().size() != 1)
    return module.emitError("gate/up view Schedule requires one unscheduled original Graph");
  if (failed(validateSource(module)))
    return module.emitError("gate/up view source is outside the native parent-view profile");
  OpBuilder b(module.getContext()); b.setInsertionPointToEnd(module.getBody());
  OperationState state(root->getLoc(), "schedule.artifact");
  state.addAttributes(artifactAttributes(*root).getValue()); b.create(state);
  return success();
}
inline LogicalResult lower(ModuleOp module, bool &selected) {
  auto root = graphRoot(module);
  if (failed(root)) return failure();
  selected = bool(*root);
  if (!selected) return success();
  Operation *artifact = nullptr;
  for (Operation &op : module.getBody()->getOperations()) {
    if (&op == root->getOperation()) continue;
    if (op.getName().getStringRef() != "schedule.artifact" || artifact)
      return module.emitError("gate/up view Tile requires one current native Schedule witness");
    artifact = &op;
  }
  if (!artifact || artifact->getAttrDictionary() != artifactAttributes(*root))
    return module.emitError("gate/up view Schedule differs from its original Graph or physical policy");
  auto built = buildScalarModule(module);
  if (failed(built)) return module.emitError("gate/up view source contract changed before Tile projection");
  OwningOpRef<ModuleOp> temporary(*built);
  auto fn = *temporary->getOps<LLVM::LLVMFuncOp>().begin();
  // The normalized body has no source M constant. Source-specific guards stay
  // in the package; sharing this physical body never relaxes those guards.
  std::string physical = row_region::digest(fn);
  fn.setSymName("tessera_gate_up_view_" + physical.substr(0,16));
  auto type = cast<RankedTensorType>(root->getResultTypes()[0]);
  OpBuilder b(artifact); OperationState state(root->getLoc(), "tile.row_region_kernel");
  state.addAttribute("name", b.getStringAttr(fn.getName()));
  state.addAttribute("arch", b.getStringAttr("gfx1201"));
  state.addAttribute("rows", b.getI64IntegerAttr(type.getDimSize(0)));
  state.addAttribute("columns", b.getI64IntegerAttr(type.getDimSize(1)));
  state.addAttribute("storage", TypeAttr::get(type.getElementType()));
  state.addAttribute("workgroup_size", b.getI64IntegerAttr(256));
  state.addAttribute("schema", b.getI64IntegerAttr(bf16Profile(*root) ? 3 : 2));
  state.addAttribute("source_hash", b.getStringAttr(sourceHash(*root)));
  state.addAttribute("artifact_hash", b.getStringAttr(artifactHash(*root)));
  state.addAttribute("body_hash", b.getStringAttr(row_region::digest(fn)));
  state.addRegion(); Operation *carrier = b.create(state);
  carrier->getRegion(0).push_back(new Block);
  fn->moveBefore(&carrier->getRegion(0).front(),carrier->getRegion(0).front().end());
  std::string original;
  llvm::raw_string_ostream originalStream(original);
  root->getOperation()->print(originalStream, OpPrintingFlags().useLocalScope()); originalStream.flush();
  module->setAttr("tessera.rocm.parent_view_source",b.getStringAttr(original));
  root->erase(); artifact->erase();
  return row_region::verifyGateUpKernel(carrier);
}
} // namespace tessera::gate_up_view
