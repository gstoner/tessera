// Compiler-owned partition of the named NVFP4 conversion/storage/product Graph.
#pragma once
#include "mlir/IR/Verifier.h"

namespace tessera {
static bool hasNativeNVFP4Graph(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::Operation *op) {
    found |= op->getName().getStringRef() == "tessera.nvfp4_requantize";
  });
  return found;
}

static mlir::func::FuncOp cloneNativeNVFP4Member(mlir::func::FuncOp member, int64_t index) {
  using namespace mlir;
  auto child = cast<func::FuncOp>(member->clone());
  const char *names[] = {"ingest", "storage", "packed_folded_w4a8"};
  child.setSymName(names[index]); child.setPublic();
  child->removeAttr("tessera.native.nvfp4_member");
  OpBuilder b(member.getContext());
  if (index < 2) {
    const char *bindings[2][6] = {{"codes","scales","globals","packed","exponents","stats"},
                                {"codes","exponents","fragment","plane",nullptr,nullptr}};
    SmallVector<Attribute> attrs;
    for (unsigned n = 0; n < (index == 0 ? 6u : 4u); ++n)
      attrs.push_back(b.getStringAttr(bindings[index][n]));
    child->setAttr("tessera.bindings", b.getArrayAttr(attrs));
  }
  return child;
}

static mlir::LogicalResult emitNativeNVFP4Program(mlir::ModuleOp module) {
  using namespace mlir;
  if (failed(verify(module))) return failure();
  if (module->hasAttr("tessera.native.nvfp4_program"))
    return module.emitError("NVFP4 native program export already exists");
  auto target = module->getAttrOfType<StringAttr>("tessera.target");
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  SmallVector<func::FuncOp> roots;
  for (auto fn : module.getOps<func::FuncOp>())
    if (!fn.isDeclaration()) roots.push_back(fn);
  if (!target || target.getValue() != "rocm_gfx1201" || !arch ||
      arch.getValue() != "gfx1201" || roots.size() != 1 ||
      !roots[0].getBody().hasOneBlock())
    return module.emitError("NVFP4 native program requires one gfx1201 Graph root");
  auto root = roots[0];
  if (root.getNumArguments() != 5 || root.getNumResults() != 1 ||
      root.getBody().front().getOperations().size() != 4)
    return root.emitError("NVFP4 native program requires its five-input three-stage Graph");
  SmallVector<Operation *> ops;
  for (auto &op : root.getBody().front().without_terminator()) ops.push_back(&op);
  auto ret = dyn_cast<func::ReturnOp>(root.getBody().front().back());
  if (ops[0]->getName().getStringRef() != "tessera.nvfp4_requantize" ||
      ops[1]->getName().getStringRef() != "tessera.mxfp4_folded_storage" ||
      ops[2]->getName().getStringRef() != "tessera.scaled_matmul" ||
      ops[0]->getNumOperands() != 3 || ops[0]->getNumResults() != 3 ||
      ops[1]->getNumOperands() != 2 || ops[1]->getNumResults() != 2 ||
      ops[2]->getNumOperands() != 4 || ops[2]->getNumResults() != 1 ||
      !ret || ret.getNumOperands() != 1 || ret.getOperand(0) != ops[2]->getResult(0) ||
      ops[1]->getOperand(0) != ops[0]->getResult(0) ||
      ops[1]->getOperand(1) != ops[0]->getResult(1) ||
      ops[2]->getOperand(1) != ops[1]->getResult(0) ||
      ops[2]->getOperand(3) != ops[1]->getResult(1))
    return root.emitError("NVFP4 native program lost its conversion/storage/product SSA edges");
  SmallVector<Value> roles(ops[0]->getOperands());
  roles.push_back(ops[2]->getOperand(0));
  roles.push_back(ops[2]->getOperand(2));
  SmallVector<int64_t> roleIndices;
  llvm::SmallDenseSet<unsigned> seen;
  for (Value value : roles) {
    auto arg = dyn_cast<BlockArgument>(value);
    if (!arg || arg.getOwner() != &root.getBody().front() ||
        !seen.insert(arg.getArgNumber()).second)
      return root.emitError("NVFP4 native program requires distinct original argument roles");
    auto attrs = root.getArgAttrDict(arg.getArgNumber());
    if (attrs && !attrs.empty())
      return root.emitError("NVFP4 native program argument metadata needs its own contract");
    roleIndices.push_back(arg.getArgNumber());
  }
  // A bounded activation-row request changes only the consumer's capacity
  // frame. Retain the original Graph witness; conversion/storage SSA and
  // numerical policy are cloned verbatim by the ordinary native partition.
  constexpr StringLiteral rowBoundKey = "tessera.native.nvfp4_m_bound";
  auto rowBoundAttr = module->getAttrOfType<IntegerAttr>(rowBoundKey);
  bool boundedRows = module->hasAttr(rowBoundKey);
  int64_t activeRows = 0, rowBound = 0;
  std::string originalIR;
  if (boundedRows) {
    auto activation = dyn_cast<RankedTensorType>(roles[3].getType());
    auto scale = dyn_cast<RankedTensorType>(roles[4].getType());
    auto output = dyn_cast<RankedTensorType>(ops[2]->getResult(0).getType());
    if (!rowBoundAttr || !rowBoundAttr.getType().isInteger(64) ||
        !activation || activation.getRank() != 2 ||
        !activation.hasStaticShape() || activation.getEncoding() ||
        !scale || scale.getRank() != 1 || !scale.hasStaticShape() ||
        scale.getEncoding() || !output || output.getRank() != 2 ||
        !output.hasStaticShape() || output.getEncoding())
      return root.emitError("NVFP4 bounded rows require static unencoded capacity types and an i64 bound");
    activeRows = activation.getDimSize(0);
    rowBound = rowBoundAttr.getInt();
    if (activeRows <= 0 || rowBound < activeRows ||
        activation.getDimSize(1) <= 0 ||
        rowBound > INT64_MAX / activation.getDimSize(1) ||
        output.getDimSize(1) <= 0 ||
        rowBound > INT64_MAX / output.getDimSize(1) / 2 ||
        rowBound > INT64_MAX / 4)
      return root.emitError("NVFP4 bounded rows exceed their checked storage capacity");
    llvm::raw_string_ostream originalStream(originalIR);
    module.print(originalStream); originalStream.flush();
    auto capacityType = [&](RankedTensorType type) {
      SmallVector<int64_t> shape(type.getShape());
      shape[0] = rowBound;
      return RankedTensorType::get(shape, type.getElementType());
    };
    root.getArgument(roleIndices[3]).setType(capacityType(activation));
    root.getArgument(roleIndices[4]).setType(capacityType(scale));
    Type result = capacityType(output);
    ops[2]->getResult(0).setType(result);
    SmallVector<Type> capacityInputs;
    for (auto argument : root.getArguments())
      capacityInputs.push_back(argument.getType());
    root.setFunctionType(FunctionType::get(module.getContext(),
        capacityInputs, TypeRange{result}));
    module->removeAttr(rowBoundKey);
    if (failed(verify(module))) return failure();
  }
  llvm::DenseMap<Value, int64_t> ids;
  SmallVector<Value> values;
  SmallVector<int64_t> writes, reads;
  for (Value value : root.getArguments()) {
    ids[value] = values.size(); values.push_back(value);
    writes.push_back(-1); reads.push_back(-1);
  }
  for (auto [index, op] : llvm::enumerate(ops)) {
    for (Value input : op->getOperands()) {
      auto found = ids.find(input);
      if (found == ids.end())
        return op->emitError("NVFP4 native program captured a value outside its SSA prefix");
      reads[found->second] = index;
    }
    for (Value output : op->getResults()) {
      ids[output] = values.size(); values.push_back(output);
      writes.push_back(index); reads.push_back(index);
    }
  }
  reads[ids.lookup(ret.getOperand(0))] = ops.size();
  std::string sourceIR; llvm::raw_string_ostream sourceStream(sourceIR);
  module.print(sourceStream); sourceStream.flush();
  OpBuilder b(module.getContext());
  llvm::json::Array bufferJSON, stepJSON, roleJSON, memberGraphs;
  for (auto [index, value] : llvm::enumerate(values)) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || !type.hasStaticShape() || type.getEncoding() ||
        !isa<FloatType, IntegerType>(type.getElementType()))
      return root.emitError("NVFP4 native program requires static unencoded byte storage");
    auto bits = type.getElementType().getIntOrFloatBitWidth();
    if (bits < 8 || bits % 8)
      return root.emitError("NVFP4 native program requires byte-addressable storage");
    uint64_t bytes = bits / 8;
    llvm::json::Array shape;
    for (int64_t dim : type.getShape()) {
      if (dim <= 0 || uint64_t(dim) > uint64_t(INT64_MAX) / bytes)
        return root.emitError("NVFP4 native program storage bytes overflow");
      bytes *= uint64_t(dim); shape.push_back(dim);
    }
    std::string storage;
    llvm::raw_string_ostream stream(storage);
    type.getElementType().print(stream); stream.flush();
    bufferJSON.push_back(llvm::json::Object{
      {"id", int64_t(index)}, {"bytes", int64_t(bytes)},
      {"shape", std::move(shape)}, {"storage", storage},
      {"ownership", index < 5 ? "readonly_input" : index == 10 ? "returned_output" : "private_scratch"},
      {"first_write", writes[index]}, {"last_read", reads[index]}});
  }
  for (unsigned index = 0; index < 3; ++index) {
    auto name = (root.getSymName() + "__nvfp4_member_" + llvm::Twine(index)).str();
    if (SymbolTable::lookupSymbolIn(module, name))
      return root.emitError("NVFP4 outlined member symbol already exists");
  }
  // Complete all admission before cloning actual operations; retain root witness.
  SmallVector<Attribute> memberSymbols;
  for (auto [index, op] : llvm::enumerate(ops)) {
    b.setInsertionPointToEnd(module.getBody());
    auto name = (root.getSymName() + "__nvfp4_member_" + llvm::Twine(index)).str();
    auto member = b.create<func::FuncOp>(op->getLoc(), name,
        b.getFunctionType(op->getOperandTypes(), op->getResultTypes()));
    member.setPrivate();
    member->setAttr("tessera.native.nvfp4_member", b.getI64IntegerAttr(index));
    memberSymbols.push_back(FlatSymbolRefAttr::get(member));
    auto block = member.addEntryBlock();
    OpBuilder body(block, block->begin());
    IRMapping mapping;
    llvm::json::Array inputs, outputs;
    for (auto [input, arg] : llvm::zip(op->getOperands(), block->getArguments())) {
      mapping.map(input, arg); inputs.push_back(ids.lookup(input));
    }
    auto cloned = body.clone(*op, mapping);
    body.create<func::ReturnOp>(op->getLoc(), cloned->getResults());
    for (Value output : op->getResults()) outputs.push_back(ids.lookup(output));
    // Serialize the exact projected member with the original module contract.
    // Portable packaging consumes this native record; it never rebuilds Graph.
    auto projected = ModuleOp::create(module.getLoc());
    projected->setAttrs(module->getAttrs());
    projected.getBody()->push_back(cloneNativeNVFP4Member(member, index));
    std::string projectedIR; llvm::raw_string_ostream projectedStream(projectedIR);
    projected.print(projectedStream); projectedStream.flush();
    memberGraphs.push_back(projectedIR + "\n");
    projected.erase();
    stepJSON.push_back(llvm::json::Object{
      {"step", int64_t(index)}, {"operation", op->getName().getStringRef().str()},
      {"member", name}, {"inputs", std::move(inputs)}, {"outputs", std::move(outputs)}});
  }
  for (int64_t index : roleIndices) roleJSON.push_back(index);
  std::string rootIR; llvm::raw_string_ostream rootStream(rootIR);
  root.print(rootStream); rootStream.flush();
  llvm::json::Object plan{{"schema", boundedRows ? "tessera.native.nvfp4_program.v2" : "tessera.native.nvfp4_program.v1"},
      {"root", root.getSymName().str()}, {"root_ir", rootIR},
      {"source_graph_ir", sourceIR + "\n"}, {"member_graphs", std::move(memberGraphs)},
      {"role_indices", std::move(roleJSON)}, {"buffers", std::move(bufferJSON)},
      {"steps", std::move(stepJSON)}, {"output", ids.lookup(ret.getOperand(0))}};
  if (boundedRows) {
    plan["active_m"] = activeRows;
    plan["m_bound"] = rowBound;
    plan["original_graph_ir"] = originalIR + "\n";
  }
  std::string json; llvm::raw_string_ostream stream(json);
  stream << llvm::json::Value(std::move(plan)); stream.flush();
  module->setAttr("tessera.native.nvfp4_program_json",
                  b.getStringAttr(llvm::encodeBase64(json)));
  module->setAttr("tessera.native.nvfp4_program", b.getDictionaryAttr({
      b.getNamedAttr("root", FlatSymbolRefAttr::get(root)),
      b.getNamedAttr("members", b.getArrayAttr(memberSymbols))}));
  return success();
}

static mlir::LogicalResult projectNativeNVFP4Member(mlir::ModuleOp module, int64_t index) {
  using namespace mlir;
  auto plan = module->getAttrOfType<DictionaryAttr>("tessera.native.nvfp4_program");
  auto members = plan ? dyn_cast_or_null<ArrayAttr>(plan.get("members")) : ArrayAttr{};
  if (!members || index < 0 || uint64_t(index) >= members.size())
    return module.emitError("NVFP4 member index is outside its native program");
  auto member = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupSymbolIn(
      module, cast<FlatSymbolRefAttr>(members[index])));
  auto lineage = member ? member->getAttrOfType<IntegerAttr>("tessera.native.nvfp4_member") : IntegerAttr{};
  if (!member || !lineage || lineage.getInt() != index)
    return module.emitError("NVFP4 native member lineage differs");
  auto child = cloneNativeNVFP4Member(member, index);
  module.getBody()->clear(); module.getBody()->push_back(child);
  module->removeAttr("tessera.native.nvfp4_program");
  module->removeAttr("tessera.native.nvfp4_program_json");
  return success();
}
} // namespace tessera
