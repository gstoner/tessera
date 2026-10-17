// Compiler-owned partition of the named NVFP4 conversion/storage/product Graph.
#pragma once
#include "mlir/IR/Verifier.h"
#include "Tessera/IR/SliceContract.h"

namespace tessera {
static bool hasNativeNVFP4Graph(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::Operation *op) {
    found |= op->getName().getStringRef() == "tessera.nvfp4_requantize" ||
             op->getName().getStringRef() == "tessera.nvfp4_expand";
  });
  return found;
}

static mlir::func::FuncOp cloneNativeNVFP4Member(mlir::func::FuncOp member, int64_t index) {
  using namespace mlir;
  auto child = cast<func::FuncOp>(member->clone());
  bool expansion = member->hasAttr("tessera.native.nvfp4_expansion_member");
  const char *names[] = {"ingest", "storage", "packed_folded_w4a8"};
  const char *expansionNames[] = {"expand", "w8a8_blockscale", "gate_up_activation"};
  child.setSymName(expansion ? expansionNames[index] : names[index]); child.setPublic();
  child->removeAttr("tessera.native.nvfp4_member");
  OpBuilder b(member.getContext());
  child->removeAttr("tessera.native.nvfp4_expansion_member");
  child->removeAttr("tessera.native.nvfp4_activation_member");
  child->removeAttr("tessera.native.nvfp4_store_cast_member");
  if (expansion && index == 0) {
    child->setAttr("tessera.bindings", b.getStrArrayAttr(
        {"codes", "scales", "globals", "values", "scale_plane"}));
  } else if (!expansion && index < 2) {
    const char *bindings[2][6] = {{"codes","scales","globals","packed","exponents","stats"},
                                {"codes","exponents","fragment","plane",nullptr,nullptr}};
    SmallVector<Attribute> attrs;
    for (unsigned n = 0; n < (index == 0 ? 6u : 4u); ++n)
      attrs.push_back(b.getStringAttr(bindings[index][n]));
    child->setAttr("tessera.bindings", b.getArrayAttr(attrs));
  }
  return child;
}

static mlir::LogicalResult verifyNativeFP32ToBF16Store(
    mlir::Value parentValue, mlir::Operation *store) {
  using namespace mlir;
  auto productType = dyn_cast<RankedTensorType>(parentValue.getType());
  auto storedType = store->getNumResults() == 1
      ? dyn_cast<RankedTensorType>(store->getResult(0).getType()) : RankedTensorType{};
  auto dtype = store->getAttrOfType<StringAttr>("dtype");
  if (store->getName().getStringRef() != "tessera.cast" ||
      store->getNumOperands() != 1 || store->getOperand(0) != parentValue ||
      !parentValue.hasOneUse() || !productType || !productType.getElementType().isF32() ||
      !storedType || !storedType.getElementType().isBF16() ||
      productType.getShape() != storedType.getShape() ||
      !dtype || dtype.getValue() != "bf16")
    return store->emitError("NVFP4 product store requires a single-use explicit FP32-to-BF16 cast");
  for (auto attr : store->getAttrs())
    if (attr.getName() != "dtype" && attr.getName() != "pure" &&
        attr.getName() != "tessera.effect_kind")
      return store->emitError("NVFP4 product store cast metadata needs its own contract");
  if (auto effect = store->getAttr("tessera.effect_kind"))
    if (!isa<StringAttr>(effect) || cast<StringAttr>(effect).getValue() != "pure")
      return store->emitError("NVFP4 product store requires a pure cast");
  if (auto pure = store->getAttr("pure"))
    if (!isa<BoolAttr>(pure) || !cast<BoolAttr>(pure).getValue())
      return store->emitError("NVFP4 product store requires a pure cast");
  return success();
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
  auto first = root.getBody().front().begin();
  bool expansion = first->getName().getStringRef() == "tessera.nvfp4_expand";
  const auto operationCount = root.getBody().front().getOperations().size();
  bool explicitStore = expansion && operationCount == 10;
  bool activation = expansion && (operationCount == 9 || explicitStore);
  const unsigned activationStart = explicitStore ? 3 : 2;
  unsigned stageCount = expansion ? (activation ? 3 : 2) : 3;
  if (root.getNumArguments() != 5 || root.getNumResults() != 1 ||
      root.getBody().front().getOperations().size() != (activation ? (explicitStore ? 10 : 9) : stageCount + 1))
    return root.emitError("NVFP4 native program requires its five-input conversion/product Graph");
  SmallVector<Operation *> ops;
  for (auto &op : root.getBody().front().without_terminator()) ops.push_back(&op);
  auto ret = dyn_cast<func::ReturnOp>(root.getBody().front().back());
  Operation *product = activation ? ops[1] : ops.back();
  bool productEdge = product->getName().getStringRef() == "tessera.scaled_matmul" &&
      product->getNumOperands() == 4 && product->getNumResults() == 1 &&
      ret && ret.getNumOperands() == 1 &&
      (activation || ret.getOperand(0) == product->getResult(0));
  if (!productEdge)
    return root.emitError("NVFP4 native program requires a returned scaled product");
  bool producerEdge = false;
  if (expansion) {
    producerEdge = ops[0]->getNumOperands() == 3 && ops[0]->getNumResults() == 2 &&
        product->getOperand(1) == ops[0]->getResult(0) &&
        product->getOperand(3) == ops[0]->getResult(1);
    auto transpose = product->getAttrOfType<BoolAttr>("transposeB");
    producerEdge &= transpose && transpose.getValue();
    auto scale = product->getAttrOfType<DictionaryAttr>("scale_layout");
    auto numeric = product->getAttrOfType<DictionaryAttr>("numeric_policy");
    auto granularity = scale ? scale.getAs<StringAttr>("granularity") : StringAttr{};
    auto format = scale ? scale.getAs<StringAttr>("format") : StringAttr{};
    auto block = scale ? scale.getAs<ArrayAttr>("block") : ArrayAttr{};
    auto accum = numeric ? numeric.getAs<StringAttr>("accum") : StringAttr{};
    auto mode = numeric ? numeric.getAs<StringAttr>("execution_mode") : StringAttr{};
    auto lhs = dyn_cast<RankedTensorType>(product->getOperand(0).getType());
    auto output = dyn_cast<RankedTensorType>(product->getResult(0).getType());
    if (!granularity || granularity.getValue() != "block" ||
        !format || format.getValue() != "fp32" || !block || block.size() != 2 ||
        !isa<IntegerAttr>(block[0]) || cast<IntegerAttr>(block[0]).getInt() != 1 ||
        !isa<IntegerAttr>(block[1]) || cast<IntegerAttr>(block[1]).getInt() != 16 ||
        !accum || accum.getValue() != "fp32" ||
        !mode || mode.getValue() != "exact_per_block" ||
        !lhs || !isa<Float8E4M3FNType>(lhs.getElementType()) ||
        !output || (!output.getElementType().isF32() && !output.getElementType().isBF16()))
      return root.emitError("source-preserving NVFP4 product requires the exact E4M3 K16 FP32-scale contract");
  } else {
    producerEdge = ops[0]->getName().getStringRef() == "tessera.nvfp4_requantize" &&
        ops[1]->getName().getStringRef() == "tessera.mxfp4_folded_storage" &&
        ops[0]->getNumOperands() == 3 && ops[0]->getNumResults() == 3 &&
        ops[1]->getNumOperands() == 2 && ops[1]->getNumResults() == 2 &&
        ops[1]->getOperand(0) == ops[0]->getResult(0) &&
        ops[1]->getOperand(1) == ops[0]->getResult(1) &&
        product->getOperand(1) == ops[1]->getResult(0) &&
        product->getOperand(3) == ops[1]->getResult(1);
  }
  if (!productEdge || !producerEdge)
    return root.emitError("NVFP4 native program lost its conversion/storage/product SSA edges");
  if (activation) {
    Value parentValue = product->getResult(0);
    if (explicitStore) {
      auto store = ops[2];
      if (failed(verifyNativeFP32ToBF16Store(parentValue,store))) return failure();
      parentValue = store->getResult(0);
    }
    auto parent = dyn_cast<RankedTensorType>(parentValue.getType());
    auto result = dyn_cast<RankedTensorType>(ret.getOperand(0).getType());
    if (!parent || !result || !parent.hasStaticShape() || !result.hasStaticShape() ||
        parent.getRank() != 2 || result.getRank() != 2 || parent.getEncoding() || result.getEncoding() ||
        !parent.getElementType().isBF16() || !result.getElementType().isBF16() ||
        parent.getDimSize(0) != result.getDimSize(0) || parent.getDimSize(1) != 2*result.getDimSize(1) ||
        result.getDimSize(0) <= 0 || result.getDimSize(1) <= 0)
      return root.emitError("NVFP4 resident activation requires its static BF16 parent/output storage");
    int64_t origins[2] = {-1, -1};
    for (unsigned i=0; i<2; ++i) {
      auto slice = ops[activationStart+i]; auto castOp = ops[activationStart+2+i];
      auto bounds = slice_contract::staticBounds(slice);
      auto castType = castOp->getNumResults() == 1
          ? dyn_cast<RankedTensorType>(castOp->getResult(0).getType()) : RankedTensorType{};
      if (!bounds || slice->getNumOperands() != 1 || slice->getNumResults() != 1 ||
          slice->getOperand(0) != parentValue || slice->getResult(0).getType() != result ||
          bounds->sizes != result.getShape() || bounds->offsets[0] != 0 ||
          (bounds->offsets[1] != 0 && bounds->offsets[1] != result.getDimSize(1)) ||
          castOp->getName().getStringRef() != "tessera.cast" || castOp->getNumOperands() != 1 ||
          castOp->getOperand(0) != slice->getResult(0) || !castType ||
          castType.getShape() != result.getShape() || !castType.getElementType().isF32())
        return root.emitError("NVFP4 resident activation lost its original view/widening SSA");
      origins[i] = bounds->offsets[1];
    }
    auto silu = ops[activationStart+4]; auto storeCast = ops[activationStart+5];
    bool ordered = silu->getNumOperands() == 2 &&
        ((silu->getOperand(0) == ops[activationStart+2]->getResult(0) && silu->getOperand(1) == ops[activationStart+3]->getResult(0)) ||
         (silu->getOperand(1) == ops[activationStart+2]->getResult(0) && silu->getOperand(0) == ops[activationStart+3]->getResult(0)));
    if (origins[0] == origins[1] || silu->getName().getStringRef() != "tessera.silu_mul" ||
        !ordered || silu->getNumResults() != 1 ||
        storeCast->getName().getStringRef() != "tessera.cast" || storeCast->getNumOperands() != 1 ||
        storeCast->getNumResults() != 1 || storeCast->getOperand(0) != silu->getResult(0) ||
        storeCast->getResult(0).getType() != result || ret.getOperand(0) != storeCast->getResult(0))
      return root.emitError("NVFP4 resident activation lost its SiLU/store SSA");
  }
  // A physical member may own several logical operations. Views and explicit
  // register casts stay inside the activation member, never device allocations.
  SmallVector<SmallVector<Operation *>> groups;
  for (unsigned i=0; i<(activation ? 2u : ops.size()); ++i) groups.push_back({ops[i]});
  if (activation) {
    if (explicitStore) groups[1].push_back(ops[2]);
    groups.emplace_back(ops.begin()+activationStart, ops.end());
  }
  SmallVector<SmallVector<Value>> stageInputs, stageOutputs;
  for (const auto &group : groups) {
    llvm::SmallDenseSet<Value> internal;
    SmallVector<Value> inputs;
    for (Operation *op : group) {
      for (Value input : op->getOperands())
        if (!internal.contains(input) && !llvm::is_contained(inputs, input)) inputs.push_back(input);
      for (Value output : op->getResults()) internal.insert(output);
    }
    stageInputs.push_back(std::move(inputs));
    stageOutputs.emplace_back(group.back()->getResults().begin(),group.back()->getResults().end());
  }
  SmallVector<Value> roles(ops[0]->getOperands());
  roles.push_back(product->getOperand(0));
  roles.push_back(product->getOperand(2));
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
  if (boundedRows && activation)
    return root.emitError("NVFP4 resident activation bounded rows require a three-member runtime expression contract");
  if (boundedRows) {
    auto activation = dyn_cast<RankedTensorType>(roles[3].getType());
    auto scale = dyn_cast<RankedTensorType>(roles[4].getType());
    auto output = dyn_cast<RankedTensorType>(product->getResult(0).getType());
    if (!rowBoundAttr || !rowBoundAttr.getType().isInteger(64) ||
        !activation || activation.getRank() != 2 ||
        !activation.hasStaticShape() || activation.getEncoding() ||
        !scale || scale.getRank() != (expansion ? 2 : 1) || !scale.hasStaticShape() ||
        scale.getEncoding() || !output || output.getRank() != 2 ||
        !output.hasStaticShape() || output.getEncoding() ||
        !isa<FloatType, IntegerType>(output.getElementType()))
      return root.emitError("NVFP4 bounded rows require static unencoded capacity types and an i64 bound");
    activeRows = activation.getDimSize(0);
    rowBound = rowBoundAttr.getInt();
    const int64_t outputBytes = output.getElementType().getIntOrFloatBitWidth() / 8;
    if (activeRows <= 0 || rowBound < activeRows ||
        activation.getDimSize(1) <= 0 ||
        rowBound > INT64_MAX / activation.getDimSize(1) ||
        scale.getDimSize(0) != activeRows || output.getDimSize(0) != activeRows ||
        (expansion && scale.getDimSize(1) != activation.getDimSize(1) / 16) ||
        output.getDimSize(1) <= 0 || outputBytes <= 0 ||
        rowBound > INT64_MAX / output.getDimSize(1) / outputBytes ||
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
    product->getResult(0).setType(result);
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
  for (auto [index, inputs] : llvm::enumerate(stageInputs)) {
    for (Value input : inputs) {
      auto found = ids.find(input);
      if (found == ids.end())
        return root.emitError("NVFP4 native program captured a value outside its SSA prefix");
      reads[found->second] = index;
    }
    for (Value output : stageOutputs[index]) {
      ids[output] = values.size(); values.push_back(output);
      writes.push_back(index); reads.push_back(index);
    }
  }
  reads[ids.lookup(ret.getOperand(0))] = groups.size();
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
      {"ownership", index < 5 ? "readonly_input" : value == ret.getOperand(0) ? "returned_output" : "private_scratch"},
      {"first_write", writes[index]}, {"last_read", reads[index]}});
  }
  for (unsigned index = 0; index < stageCount; ++index) {
    auto name = (root.getSymName() + "__nvfp4_member_" + llvm::Twine(index)).str();
    if (SymbolTable::lookupSymbolIn(module, name))
      return root.emitError("NVFP4 outlined member symbol already exists");
  }
  // Complete all admission before cloning actual operations; retain root witness.
  SmallVector<Attribute> memberSymbols;
  for (auto [index, group] : llvm::enumerate(groups)) {
    Operation *op = group.front();
    SmallVector<Type> inputTypes, outputTypes;
    for (Value value : stageInputs[index]) inputTypes.push_back(value.getType());
    for (Value value : stageOutputs[index]) outputTypes.push_back(value.getType());
    b.setInsertionPointToEnd(module.getBody());
    auto name = (root.getSymName() + "__nvfp4_member_" + llvm::Twine(index)).str();
    auto member = b.create<func::FuncOp>(op->getLoc(), name,
        b.getFunctionType(inputTypes, outputTypes));
    member.setPrivate();
    member->setAttr("tessera.native.nvfp4_member", b.getI64IntegerAttr(index));
    if (expansion)
      member->setAttr("tessera.native.nvfp4_expansion_member", b.getUnitAttr());
    if (activation && index == 2)
      member->setAttr("tessera.native.nvfp4_activation_member", b.getUnitAttr());
    if (explicitStore && index == 1)
      member->setAttr("tessera.native.nvfp4_store_cast_member", b.getUnitAttr());
    memberSymbols.push_back(FlatSymbolRefAttr::get(member));
    auto block = member.addEntryBlock();
    OpBuilder body(block, block->begin());
    IRMapping mapping;
    llvm::json::Array inputs, outputs;
    for (auto [input, arg] : llvm::zip(stageInputs[index], block->getArguments())) {
      mapping.map(input, arg); inputs.push_back(ids.lookup(input));
    }
    for (Operation *logical : group) body.clone(*logical, mapping);
    SmallVector<Value> returned;
    for (Value output : stageOutputs[index]) {
      returned.push_back(mapping.lookup(output)); outputs.push_back(ids.lookup(output));
    }
    body.create<func::ReturnOp>(op->getLoc(), returned);
    // Serialize the exact projected member with the original module contract.
    // Portable packaging consumes this native record; it never rebuilds Graph.
    auto projected = ModuleOp::create(module.getLoc());
    projected->setAttrs(module->getAttrs());
    if (activation && index == 2) {
      projected->setAttr("tessera.target", b.getStringAttr("rocm"));
      projected->setAttr("tessera.native.gate_up_view", b.getUnitAttr());
    }
    if (explicitStore && index == 1)
      projected->setAttr("tessera.native.fp8_store_cast", b.getUnitAttr());
    projected.getBody()->push_back(cloneNativeNVFP4Member(member, index));
    std::string projectedIR; llvm::raw_string_ostream projectedStream(projectedIR);
    projected.print(projectedStream); projectedStream.flush();
    memberGraphs.push_back(projectedIR + "\n");
    projected.erase();
    stepJSON.push_back(llvm::json::Object{
      {"step", int64_t(index)}, {"operation", activation && index == 2
          ? "tessera.silu_mul" : op->getName().getStringRef().str()},
      {"member", name}, {"inputs", std::move(inputs)}, {"outputs", std::move(outputs)}});
  }
  for (int64_t index : roleIndices) roleJSON.push_back(index);
  std::string rootIR; llvm::raw_string_ostream rootStream(rootIR);
  root.print(rootStream); rootStream.flush();
  // v4 adds bounded source expansion. v1/v2 retain lossy static/bounded
  // meanings and v3 remains static source expansion. v5 adds a static BF16
  // activation member; bounded three-member updates need their own schema.
  // No prior schema is repurposed.
  const char *schema = activation ? "tessera.native.nvfp4_program.v5" : expansion
      ? (boundedRows ? "tessera.native.nvfp4_program.v4" : "tessera.native.nvfp4_program.v3")
      : (boundedRows ? "tessera.native.nvfp4_program.v2" : "tessera.native.nvfp4_program.v1");
  llvm::json::Object plan{{"schema", schema},
      {"root", root.getSymName().str()}, {"root_ir", rootIR},
      {"source_graph_ir", sourceIR + "\n"}, {"member_graphs", std::move(memberGraphs)},
      {"role_indices", std::move(roleJSON)}, {"buffers", std::move(bufferJSON)},
      {"steps", std::move(stepJSON)}, {"output", ids.lookup(ret.getOperand(0))}};
  if (boundedRows) {
    plan["active_m"] = activeRows;
    plan["m_bound"] = rowBound;
    plan["original_graph_ir"] = originalIR + "\n";
    if (expansion) {
      // Frame rows belong only to activation, its K16 scale matrix and the
      // returned product. The two weight-expansion outputs remain static.
      auto rowBytes = [&](Value value) -> int64_t {
        auto type = cast<RankedTensorType>(value.getType());
        return (type.getNumElements() / rowBound) *
               (type.getElementType().getIntOrFloatBitWidth() / 8);
      };
      plan["row_frame"] = llvm::json::Object{
          {"input_slots", llvm::json::Array{roleIndices[3], roleIndices[4]}},
          {"output_slot", ids.lookup(ret.getOperand(0))},
          {"row_strides_bytes", llvm::json::Array{
              rowBytes(roles[3]), rowBytes(roles[4]), rowBytes(product->getResult(0))}},
          {"consumer_step", int64_t(stageCount - 1)}};
    }
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
  bool activationMember = member->hasAttr("tessera.native.nvfp4_activation_member");
  bool storeCastMember = member->hasAttr("tessera.native.nvfp4_store_cast_member");
  auto child = cloneNativeNVFP4Member(member, index);
  module.getBody()->clear();
  if (activationMember) {
    OpBuilder b(module.getContext());
    module->setAttr("tessera.target", b.getStringAttr("rocm"));
    module->setAttr("tessera.native.gate_up_view", b.getUnitAttr());
  }
  if (storeCastMember)
    module->setAttr("tessera.native.fp8_store_cast", UnitAttr::get(module.getContext()));
  module.getBody()->push_back(child);
  module->removeAttr("tessera.native.nvfp4_program");
  module->removeAttr("tessera.native.nvfp4_program_json");
  return success();
}
} // namespace tessera
