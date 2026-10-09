// Compiler-owned SSA program export for block-scaled products and adjoint regions.
#pragma once
#include "Tessera/IR/TesseraOps.h"
#include "Tessera/IR/ScaledBatchContract.h"
#include "Tessera/IR/TransposeUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>
#include <functional>
#include "llvm/Support/JSON.h"
#include "llvm/Support/Base64.h"

#include "NativeNVFP4Program.h"
#include "NativeSM120TensorProgram.h"
#include "NativeFloatingScaledProduct.h"

namespace tessera {
static mlir::LogicalResult emitNativeScaledMatmulProgram(mlir::ModuleOp module, bool primal = false, bool reverse = false) {
  using namespace mlir;
  if (primal && hasNativeNVFP4Graph(module))
    return emitNativeNVFP4Program(module);
  if (primal && hasNativeSM120TensorGraph(module))
    return emitNativeSM120TensorProgram(module);
  if (module->hasAttr("tessera.autodiff.scaled_program"))
    return module.emitError("scaled JVP program export already exists");
  llvm::SmallVector<func::FuncOp> roots;
  module.walk([&](func::FuncOp fn) {
    auto role = fn->getAttrOfType<StringAttr>("tessera.autodiff.role");
    if ((primal && !fn.isDeclaration() && !role) ||
        (!primal && role && role.getValue() == (reverse ? "backward" : "jvp"))) roots.push_back(fn);
  });
  if (roots.size() != 1 || !roots.front().getBody().hasOneBlock())
    return module.emitError("scaled JVP program needs one single-block paired root");
  auto root = roots.front();
  auto ret = dyn_cast<func::ReturnOp>(root.getBody().front().back());
  if (!ret || (!reverse && ret.getNumOperands() != (primal ? 1 : 2)))
    return root.emitError("scaled program has an invalid output role count");
  SmallVector<Value> returned;
  SmallVector<int64_t> gradientRoles;
  llvm::DenseMap<Value, int64_t> gradientForValue;
  llvm::SmallPtrSet<Operation *, 8> cotangentPermutations;
  auto isScaleSum = [](Operation *op) {
    if (isa<AddOp>(op)) return true;
    auto sum = dyn_cast<arith::AddFOp>(op);
    auto type = sum ? dyn_cast<RankedTensorType>(sum.getType()) : RankedTensorType{};
    return type && type.getElementType().isF32() &&
           sum.getFastmath() == arith::FastMathFlags::none;
  };
  if (reverse) {
    auto forwardRef = root->getAttrOfType<FlatSymbolRefAttr>("tessera.autodiff.forward");
    auto forward = forwardRef ? dyn_cast_or_null<func::FuncOp>(
        SymbolTable::lookupSymbolIn(module, forwardRef)) : func::FuncOp{};
    auto request = forward ? forward->getAttrOfType<ArrayAttr>("tessera.autodiff.wrt_indices")
                           : ArrayAttr{};
    if (!forward || !request || request.empty() || forward.getNumResults() != 1 ||
        forward.getNumArguments() < 4 || forward.getNumArguments() >= 128 ||
        ret.getNumOperands() != forward.getNumArguments() ||
        root.getNumArguments() != forward.getNumArguments() + 1)
      return root.emitError("scaled transpose export needs explicit scale roles and one output seed");
    llvm::SmallDenseSet<int64_t> floatingArguments;
    forward.walk([&](ScaledMatmulOp product) {
      for (unsigned index = 0; index < product->getNumOperands(); ++index) {
        if (!product.isLinearInOperand(index)) continue;
        auto argument = dyn_cast<BlockArgument>(product->getOperand(index));
        if (argument && argument.getOwner() == &forward.getBody().front())
          floatingArguments.insert(argument.getArgNumber());
      }
    });
    std::function<LogicalResult(Value)> visitCotangent =
        [&](Value value) -> LogicalResult {
      auto argument = dyn_cast<BlockArgument>(value);
      if (argument && argument.getOwner() == &root.getBody().front() &&
          argument.getArgNumber() == root.getNumArguments() - 1)
        return success();
      Operation *definition = value.getDefiningOp();
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!definition || definition->getBlock() != &root.getBody().front() ||
          !isa<TransposeOp>(definition) || definition->getNumOperands() != 1 ||
          !type || !type.hasStaticShape() || !type.getElementType().isF32() ||
          type.getEncoding() || type.getRank() < 1 || type.getRank() > 8 ||
          !transposePermutation(definition))
        return root.emitError("scale adjoint computed capture must be a native output-cotangent permutation");
      if (!cotangentPermutations.insert(definition).second) return success();
      return visitCotangent(definition->getOperand(0));
    };
    std::function<LogicalResult(Value, int64_t)> visit =
        [&](Value value, int64_t role) -> LogicalResult {
      auto found = gradientForValue.find(value);
      if (found != gradientForValue.end())
        return found->second == role ? success() :
            root.emitError("scaled transpose shares an adjoint across distinct scale roles");
      Operation *definition = value.getDefiningOp();
      if (!definition || definition->getBlock() != &root.getBody().front() ||
          definition->getNumResults() != 1 ||
          (!isa<tensor::GenerateOp>(definition) && !isScaleSum(definition)))
        return root.emitError("scaled transpose needs native scale reductions and their sums");
      gradientForValue[value] = role;
      if (isScaleSum(definition)) {
        for (Value operand : definition->getOperands())
          if (failed(visit(operand, role))) return failure();
      } else {
        llvm::SetVector<Value> captures;
        getUsedValuesDefinedAbove(definition->getRegions(), captures);
        for (Value capture : captures) {
          auto argument = dyn_cast<BlockArgument>(capture);
          if (argument && argument.getOwner() == &root.getBody().front()) continue;
          if (failed(visitCotangent(capture))) return failure();
        }
      }
      return success();
    };
    for (auto attr : request) {
      auto role = dyn_cast<IntegerAttr>(attr);
      if (!role || role.getInt() < 0 ||
          role.getInt() >= forward.getNumArguments() ||
          !floatingArguments.contains(role.getInt()) ||
          llvm::is_contained(gradientRoles, role.getInt()))
        return root.emitError("scaled transpose export requires unique floating scale roles");
      auto type = dyn_cast<RankedTensorType>(forward.getArgument(role.getInt()).getType());
      if (!type || !type.getElementType().isF32())
        return root.emitError("scaled transpose export requires unique floating scale roles");
      gradientRoles.push_back(role.getInt());
      returned.push_back(ret.getOperand(role.getInt()));
      if (failed(visit(returned.back(), role.getInt()))) return failure();
    }
  } else llvm::append_range(returned, ret.getOperands());

  llvm::SmallVector<Operation *> ops;
  llvm::DenseMap<Operation *, SmallVector<Value>> capturedInputs;
  bool hasScaledProduct = false;
  for (Operation &op : root.getBody().front().without_terminator()) {
    if (reverse) {
      // Select requested actual reductions, preserving the complete backward
      // root as witness. Unreturned matrix-storage zeros are not members.
      if (cotangentPermutations.contains(&op)) {
        llvm::append_range(capturedInputs[&op], op.getOperands());
        ops.push_back(&op);
        continue;
      }
      if (op.getNumResults() != 1 || !gradientForValue.contains(op.getResult(0)))
        continue;
      if (isScaleSum(&op)) {
        llvm::append_range(capturedInputs[&op], op.getOperands());
        ops.push_back(&op);
        continue;
      }
      auto kind = op.getAttrOfType<StringAttr>("tessera.autodiff.scale_adjoint");
      if (op.getName().getStringRef() != "tensor.generate" || !kind ||
          (kind.getValue() != "lhs_scale" && kind.getValue() != "rhs_scale" &&
           kind.getValue() != "lhs_matrix" && kind.getValue() != "rhs_matrix") ||
          op.getNumRegions() != 1)
        return op.emitError("scaled transpose result must retain its native reduction");
      llvm::SetVector<Value> captures;
      getUsedValuesDefinedAbove(op.getRegions(), captures);
      // Root argument order determines stable ABI order, regardless of the
      // order in which the scalar reduction body happens to read its inputs.
      for (Value argument : root.getArguments())
        if (captures.contains(argument)) capturedInputs[&op].push_back(argument);
      // Computed seed permutations follow root arguments in stable SSA order.
      for (Operation &producer : root.getBody().front().without_terminator())
        if (cotangentPermutations.contains(&producer) &&
            captures.contains(producer.getResult(0)))
          capturedInputs[&op].push_back(producer.getResult(0));
      if (capturedInputs[&op].size() != captures.size())
        return op.emitError("scaled transpose captured a value outside its native input/cotangent frame");
      hasScaledProduct = true;
    } else {
      if (!isa<ScaledMatmulOp, AddOp, TransposeOp>(op) || op.getNumResults() != 1)
        return op.emitError("scaled JVP program needs scaled products, native sums and result permutations");
      if (isa<TransposeOp>(op)) {
        auto type = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
        auto axes = transposePermutation(&op);
        if (!axes || !type.hasStaticShape() || !type.getElementType().isF32() ||
            type.getEncoding() || type.getRank() < 1 || type.getRank() > 8)
          return op.emitError("scaled result permutation requires static unencoded rank-1..8 f32 storage");
        auto *producer = op.getOperand(0).getDefiningOp();
        if (!producer || producer->getBlock() != &root.getBody().front() ||
            !isa<ScaledMatmulOp, AddOp, TransposeOp>(producer))
          return op.emitError("scaled result permutation must retain a native computed producer");
      }
      llvm::append_range(capturedInputs[&op], op.getOperands());
      hasScaledProduct |= isa<ScaledMatmulOp>(op);
    }
    ops.push_back(&op);
  }
  if (!hasScaledProduct)
    return root.emitError("scaled program has no admitted native product/reduction");

  llvm::DenseMap<Value, int64_t> ids;
  llvm::SmallVector<Value> values;
  llvm::SmallVector<int64_t> writes, reads;
  auto addValue = [&](Value value, int64_t write) {
    ids[value] = values.size();
    values.push_back(value);
    writes.push_back(write);
    reads.push_back(write);
  };
  for (auto arg : root.getArguments()) addValue(arg, -1);
  for (auto [index, op] : llvm::enumerate(ops)) {
    for (Value input : capturedInputs[op]) {
      auto found = ids.find(input);
      if (found == ids.end())
        return op->emitError("scaled JVP program captured a value outside its SSA prefix");
      reads[found->second] = index;
    }
    addValue(op->getResult(0), index);
    auto name = (root.getSymName() + "__member_" + llvm::Twine(index)).str();
    if (SymbolTable::lookupSymbolIn(module, name))
      return op->emitError("scaled JVP program member symbol already exists");
  }
  llvm::SmallVector<int64_t> outputs;
  for (Value value : returned) {
    auto found = ids.find(value);
    if (found == ids.end() || found->second < root.getNumArguments())
      return root.emitError("scaled JVP outputs must be owned computed buffers");
    outputs.push_back(found->second);
    reads[found->second] = ops.size(); // Escape until the paired call returns.
  }
  if (!primal && outputs.size() == 2 && outputs[0] == outputs[1])
    return root.emitError("scaled JVP outputs must have distinct ownership");

  OpBuilder b(module.getContext());
  llvm::SmallVector<Attribute> buffers;
  for (auto [id, value] : llvm::enumerate(values)) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || !type.hasStaticShape() ||
        !isa<FloatType, IntegerType>(type.getElementType()))
      return root.emitError("scaled JVP program needs static integer/floating storage");
    uint64_t count = 1;
    for (int64_t dim : type.getShape()) {
      if (dim <= 0 || count > uint64_t(INT64_MAX) / uint64_t(dim))
        return root.emitError("scaled JVP program storage extent is invalid or overflows");
      count *= dim;
    }
    unsigned bits = type.getElementType().getIntOrFloatBitWidth();
    if (bits < 8 || bits % 8 || type.getEncoding())
      return root.emitError("scaled JVP program needs byte-addressable unencoded storage");
    uint64_t bytesPerElement = bits / 8;
    if (count > uint64_t(INT64_MAX) / bytesPerElement)
      return root.emitError("scaled JVP program storage bytes overflow");
    buffers.push_back(b.getDictionaryAttr({
        b.getNamedAttr("id", b.getI64IntegerAttr(id)),
        b.getNamedAttr("type", TypeAttr::get(type)),
        b.getNamedAttr("logical_bytes", b.getI64IntegerAttr(count*bytesPerElement)),
        b.getNamedAttr("argument_attributes",
            id < root.getNumArguments() && root.getArgAttrDict(id)
                ? root.getArgAttrDict(id) : b.getDictionaryAttr({})) ,
        b.getNamedAttr("ownership", b.getStringAttr(
            id < root.getNumArguments() ? "readonly_input" :
            llvm::is_contained(outputs, int64_t(id)) ? "returned_output" : "private_scratch")),
        b.getNamedAttr("first_write", b.getI64IntegerAttr(writes[id])),
        b.getNamedAttr("last_read", b.getI64IntegerAttr(reads[id]))}));
  }

  // Admission is complete before mutation. Clone the actual operation SSA,
  // retaining all semantic attributes; production Python never reconstructs it.
  llvm::SmallVector<Attribute> steps;
  b.setInsertionPointToEnd(module.getBody());
  for (auto [index, op] : llvm::enumerate(ops)) {
    auto name = (root.getSymName() + "__member_" + llvm::Twine(index)).str();
    auto member = b.create<func::FuncOp>(
        op->getLoc(), name,
        b.getFunctionType(llvm::to_vector(llvm::map_range(capturedInputs[op],
            [](Value value) { return value.getType(); })), op->getResultTypes()));
    member.setPrivate();
    member->setAttr("tessera.autodiff.program_member", b.getUnitAttr());
    auto *entry = member.addEntryBlock();
    for (auto [argument, input] : llvm::enumerate(capturedInputs[op])) {
      auto source = dyn_cast<BlockArgument>(input);
      if (source && source.getOwner() == &root.getBody().front()) {
        auto attrs = root.getArgAttrDict(source.getArgNumber());
        if (attrs && !attrs.empty()) member.setArgAttrs(argument, attrs);
      }
    }
    OpBuilder body(entry, entry->begin());
    IRMapping map;
    llvm::SmallVector<int64_t> inputs;
    for (auto [input, arg] : llvm::zip(capturedInputs[op], entry->getArguments())) {
      map.map(input, arg);
      inputs.push_back(ids.lookup(input));
    }
    Operation *cloned;
    if (reverse && isa<arith::AddFOp>(op)) {
      // Native AD emits strict tensor arith.addf for accumulated adjoints.
      // Reify that exact sum as the registered Graph operation consumed by
      // Schedule; preserve the unmodified AD root as the semantic witness.
      OperationState state(op->getLoc(), AddOp::getOperationName());
      for (Value operand : op->getOperands()) state.addOperands(map.lookup(operand));
      state.addTypes(op->getResultTypes());
      cloned = body.create(state);
    } else cloned = body.clone(*op, map);
    body.create<func::ReturnOp>(op->getLoc(), cloned->getResults());
    if (!reverse && isNativeFloatingScaledProduct(cloned) &&
        failed(expandNativeFloatingScaledProduct(cloned)))
      return failure();
    steps.push_back(b.getDictionaryAttr({
        b.getNamedAttr("member", FlatSymbolRefAttr::get(member)),
        b.getNamedAttr("inputs", b.getDenseI64ArrayAttr(inputs)),
        b.getNamedAttr("output", b.getI64IntegerAttr(ids.lookup(op->getResult(0))))}));
    b.setInsertionPointToEnd(module.getBody());
  }
  // Keep the paired root as the semantic witness. Its export is a program
  // contract, not permission for a single-kernel generator to erase the sum.
  llvm::json::Array manifestBuffers, manifestSteps, manifestOutputs;
  for (auto attr : buffers) {
    auto buffer = cast<DictionaryAttr>(attr);
    auto type = cast<RankedTensorType>(cast<TypeAttr>(buffer.get("type")).getValue());
    llvm::json::Array shape;
    for (int64_t dim : type.getShape()) shape.push_back(dim);
    std::string storage;
    llvm::raw_string_ostream storageStream(storage);
    type.getElementType().print(storageStream);
    storageStream.flush();
    auto ownership = cast<StringAttr>(buffer.get("ownership")).getValue();
    manifestBuffers.push_back(llvm::json::Object{
        {"id", cast<IntegerAttr>(buffer.get("id")).getInt()},
        {"bytes", cast<IntegerAttr>(buffer.get("logical_bytes")).getInt()},
        {"elements", type.getNumElements()},
        {"shape", std::move(shape)}, {"storage", storage},
        {"ownership", ownership == "readonly_input" ? 0 : ownership == "private_scratch" ? 1 : 2},
        {"first_write", cast<IntegerAttr>(buffer.get("first_write")).getInt()},
        {"last_read", cast<IntegerAttr>(buffer.get("last_read")).getInt()}});
  }
  for (auto [index, attr] : llvm::enumerate(steps)) {
    auto step = cast<DictionaryAttr>(attr);
    llvm::json::Array inputs;
    for (int64_t id : cast<DenseI64ArrayAttr>(step.get("inputs")).asArrayRef())
      inputs.push_back(id);
    llvm::json::Object manifestStep{
        {"step", int64_t(index)},
        {"member", cast<FlatSymbolRefAttr>(step.get("member")).getValue().str()},
        {"operation", reverse && isa<arith::AddFOp>(ops[index]) ?
            AddOp::getOperationName().str() : ops[index]->getName().getStringRef().str()},
        {"inputs", std::move(inputs)},
        {"output", cast<IntegerAttr>(step.get("output")).getInt()}};
    if (!reverse && isNativeFloatingScaledProduct(ops[index])) {
      auto product = cast<ScaledMatmulOp>(ops[index]);
      auto block = product.getScaleLayoutAttr().getAs<ArrayAttr>("block");
      manifestStep["lowering"] = "structured_f32_scaled_product";
      manifestStep["continuous_contract"] = llvm::json::Object{
          {"scale_n", cast<IntegerAttr>(block[0]).getInt()},
          {"scale_k", cast<IntegerAttr>(block[1]).getInt()},
          {"transposeA", bool(product.getTransposeA())},
          {"transposeB", bool(product.getTransposeB())}};
    }
    if (isa<TransposeOp>(ops[index])) {
      llvm::json::Array axes;
      auto permutation = transposePermutation(ops[index]);
      for (int64_t axis : *permutation) axes.push_back(axis);
      manifestStep["permutation"] = std::move(axes);
    }
    if (auto transpose = ops[index]->getAttrOfType<BoolAttr>("transposeA"); transpose && transpose.getValue())
      manifestStep["transposeA"] = true;
    if (auto batching = ops[index]->getAttrOfType<StringAttr>("batching"))
      manifestStep["batching"] = batching.getValue().str();
    else if (needsScalarScaledPlane(ops[index]))
      manifestStep["batching"] = "broadcast";
    if (auto role = ops[index]->getAttrOfType<StringAttr>("tessera.autodiff.scale_adjoint")) {
      manifestStep["gradient_role"] = role.getValue().str();
    }
    if (reverse && gradientForValue.contains(ops[index]->getResult(0)))
      manifestStep["gradient_argument"] = gradientForValue.lookup(ops[index]->getResult(0));
    if (reverse && cotangentPermutations.contains(ops[index]))
      manifestStep["cotangent_source"] = int64_t(root.getNumArguments() - 1);
    manifestSteps.push_back(std::move(manifestStep));
  }
  for (int64_t id : outputs) manifestOutputs.push_back(id);
  std::string rootIR;
  llvm::raw_string_ostream rootStream(rootIR);
  root.print(rootStream);
  rootStream.flush();
  llvm::json::Object manifest{
      {"schema", 1}, {"kind", primal ? "primal" : reverse ? "scale_vjp" : "paired_jvp"},
      {"root", root.getSymName().str()}, {"root_ir", rootIR},
      {"argument_count", int64_t(root.getNumArguments())},
      {"buffers", std::move(manifestBuffers)}, {"steps", std::move(manifestSteps)},
      {"outputs", std::move(manifestOutputs)}};
  if (reverse) {
    llvm::json::Array roles;
    for (int64_t role : gradientRoles) roles.push_back(role);
    manifest["gradient_roles"] = std::move(roles);
  }
  std::string manifestText;
  llvm::raw_string_ostream manifestStream(manifestText);
  manifestStream << llvm::json::Value(std::move(manifest));
  manifestStream.flush();
  module->setAttr("tessera.autodiff.scaled_program_json",
      b.getStringAttr(llvm::encodeBase64(manifestText)));
  module->setAttr("tessera.autodiff.scaled_program", b.getDictionaryAttr({
      b.getNamedAttr("schema", b.getI64IntegerAttr(1)),
      b.getNamedAttr("root", FlatSymbolRefAttr::get(root)),
      b.getNamedAttr("kind", b.getStringAttr(primal ? "primal" : reverse ? "scale_vjp" : "paired_jvp")),
      b.getNamedAttr("argument_count", b.getI64IntegerAttr(root.getNumArguments())),
      b.getNamedAttr("outputs", b.getDenseI64ArrayAttr(outputs)),
      b.getNamedAttr("buffers", b.getArrayAttr(buffers)),
      b.getNamedAttr("steps", b.getArrayAttr(steps))}));
  return success();
}
static mlir::LogicalResult projectNativeScaledMatmulMember(
    mlir::ModuleOp module, int64_t index) {
  using namespace mlir;
  if (module->hasAttr("tessera.native.nvfp4_program"))
    return projectNativeNVFP4Member(module, index);
  if (module->hasAttr("tessera.native.sm120_tensor_program"))
    return projectNativeSM120TensorMember(module, index);
  auto program = module->getAttrOfType<DictionaryAttr>(
      "tessera.autodiff.scaled_program");
  auto steps = program ? dyn_cast_or_null<ArrayAttr>(program.get("steps")) : ArrayAttr{};
  if (!steps || index < 0 || uint64_t(index) >= steps.size())
    return module.emitError("scaled JVP member index is outside its native program");
  auto step = cast<DictionaryAttr>(steps[index]);
  auto symbol = cast<FlatSymbolRefAttr>(step.get("member"));
  auto member = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupSymbolIn(module, symbol));
  if (!member || !member->hasAttr("tessera.autodiff.program_member"))
    return module.emitError("scaled JVP member lost its native outlined function");
  if (module->hasAttr("tessera.autodiff.scaled_member") ||
      module->hasAttr("tessera.autodiff.scaled_program_witness") ||
      module->hasAttr("tessera.launch_bindings"))
    return module.emitError("scaled JVP member projection conflicts with an existing binding contract");
  std::string witness;
  llvm::raw_string_ostream stream(witness);
  module.print(stream);
  stream.flush();
  auto child = cast<func::FuncOp>(member->clone());
  // The witness retains every product/sum and lifetime. The isolated child is
  // compiled by existing Schedule/Tile consumers, never reconstructed in Python.
  auto kind = child.getBody().front().front().getName().getStringRef();
  OpBuilder b(module.getContext());
  auto contract = b.getDictionaryAttr({
      b.getNamedAttr("schema", b.getI64IntegerAttr(1)),
      b.getNamedAttr("step", b.getI64IntegerAttr(index)),
      b.getNamedAttr("inputs", step.get("inputs")),
      b.getNamedAttr("output", step.get("output")),
      b.getNamedAttr("member", b.getStringAttr(child.getSymName())),
      b.getNamedAttr("operation", b.getStringAttr(kind)),
      b.getNamedAttr("kind", program.get("kind")),
      b.getNamedAttr("type", TypeAttr::get(child.getFunctionType()))});
  module.getBody()->clear();
  module.getBody()->push_back(child);
  module->removeAttr("tessera.autodiff.scaled_program");
  module->setAttr("tessera.autodiff.scaled_member", contract);
  module->setAttr("tessera.autodiff.scaled_program_witness",
                  b.getStringAttr(witness));
  auto target = module->getAttrOfType<StringAttr>("tessera.target");
  if ((kind == "tessera.add" || kind == "tessera.transpose") &&
      target && target.getValue() == "rocm") {
    llvm::SmallVector<Attribute> bindings;
    for (unsigned arg = 0; arg < child.getNumArguments(); ++arg)
      bindings.push_back(b.getStringAttr((llvm::Twine("member_input_") + llvm::Twine(arg)).str()));
    bindings.push_back(b.getStringAttr("member_output"));
    module->setAttr("tessera.launch_bindings", b.getArrayAttr(bindings));
  }
  return success();
}
} // namespace tessera
