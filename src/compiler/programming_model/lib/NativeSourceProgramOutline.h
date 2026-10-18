#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/ADT/StringExtras.h"

namespace tessera::source_program {
inline std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return text + "\n";
}

// Outline original typed SSA. Schedule/Tile and execution remain separate:
// neither numerical expressions nor physical kernels are manufactured here.
inline mlir::LogicalResult outline(mlir::ModuleOp module) {
  using namespace mlir;
  auto reject = [&]() {
    return module.emitError("native source program requires its owning static "
                            "half RMSNorm to saved-LSE attention SSA");
  };
  auto target = module->getAttrOfType<StringAttr>("tessera.target");
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  if (!target || target.getValue() != "nvidia_sm120" || !arch ||
      arch.getValue() != "sm_120" ||
      module->hasAttr("tessera.native.source_program_json"))
    return reject();
  SmallVector<func::FuncOp> roots(module.getOps<func::FuncOp>());
  if (roots.size() != 1 ||
      llvm::range_size(module.getBody()->getOperations()) != 1)
    return reject();
  auto root = roots.front();
  // This export binds a primal program. Pending differentiation intent must
  // be consumed by its AD transform, never discarded while cloning members.
  auto pendingDifferentiation = [](Operation *operation) {
    for (NamedAttribute attribute : operation->getAttrs()) {
      llvm::StringRef name = attribute.getName().getValue();
      if (name == "tessera.autodiff" ||
          name.starts_with("tessera.autodiff."))
        return true;
    }
    return false;
  };
  if (pendingDifferentiation(module.getOperation()) ||
      pendingDifferentiation(root.getOperation()))
    return reject();
  if (!root.getBody().hasOneBlock() || root.getNumArguments() != 3 ||
      root.getNumResults() != 2)
    return reject();
  SmallVector<Operation *> ops;
  for (auto &op : root.getBody().front().without_terminator())
    ops.push_back(&op);
  if (ops.size() != 2 ||
      ops[0]->getName().getStringRef() != "tessera.rmsnorm" ||
      ops[1]->getName().getStringRef() != "tessera.flash_attn")
    return reject();
  auto norm = ops[0];
  auto attention = ops[1];
  auto saved = attention->getAttrOfType<StringAttr>("lse_checkpoint");
  if (norm->getNumOperands() != 1 || norm->getNumResults() != 1 ||
      attention->getNumOperands() != 3 || attention->getNumResults() != 2 ||
      attention->getOperand(0) != norm->getResult(0) ||
      !saved || saved.getValue() != "saved" ||
      norm->getNumRegions() || attention->getNumRegions() ||
      !isMemoryEffectFree(norm) || !isMemoryEffectFree(attention))
    return reject();
  auto q = dyn_cast<BlockArgument>(norm->getOperand(0));
  auto k = dyn_cast<BlockArgument>(attention->getOperand(1));
  auto v = dyn_cast<BlockArgument>(attention->getOperand(2));
  if (!q || !k || !v || q.getOwner() != &root.getBody().front() ||
      k.getOwner() != q.getOwner() || v.getOwner() != q.getOwner() ||
      q == k || q == v || k == v)
    return reject();
  auto qt = dyn_cast<RankedTensorType>(q.getType());
  auto kt = dyn_cast<RankedTensorType>(k.getType());
  auto vt = dyn_cast<RankedTensorType>(v.getType());
  auto ot = dyn_cast<RankedTensorType>(attention->getResult(0).getType());
  auto lt = dyn_cast<RankedTensorType>(attention->getResult(1).getType());
  SmallVector<RankedTensorType> types;
  for (Type type : root.getArgumentTypes()) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor) return reject();
    types.push_back(tensor);
  }
  types.push_back(qt); types.push_back(ot); types.push_back(lt);
  if (!qt || !kt || !vt || !ot || !lt ||
      (!qt.getElementType().isF16() && !qt.getElementType().isBF16()) ||
      kt.getElementType() != qt.getElementType() ||
      vt.getElementType() != qt.getElementType() ||
      norm->getResult(0).getType() != qt ||
      ot.getElementType() != qt.getElementType() ||
      !lt.getElementType().isF32())
    return reject();
  for (auto [i, type] : llvm::enumerate(types)) {
    if (!type.hasStaticShape() || type.getEncoding() ||
        type.getRank() != (i == 5 ? 3 : 4))
      return reject();
    int64_t elements = 1;
    for (int64_t dim : type.getShape()) {
      if (dim <= 0 || dim > INT32_MAX || elements > INT64_MAX / 4 / dim)
        return reject();
      elements *= dim;
    }
  }
  auto ret = dyn_cast<func::ReturnOp>(root.getBody().front().getTerminator());
  if (!ret || ret.getNumOperands() != 2 ||
      ret.getOperand(0) == ret.getOperand(1) ||
      llvm::any_of(ret.getOperands(), [&](Value value) {
        return value != attention->getResult(0) &&
               value != attention->getResult(1);
      }))
    return reject();
  for (unsigned i = 0; i < root.getNumArguments(); ++i)
    if (auto attrs = root.getArgAttrDict(i); attrs && !attrs.empty())
      return reject();

  const std::string original = printModule(module);
  OpBuilder builder(module.getContext());
  SmallVector<func::FuncOp> members;
  llvm::json::Array graphs;
  for (unsigned i = 0; i < 2; ++i) {
    auto op = ops[i];
    std::string name = (root.getSymName() + "__member_" + llvm::Twine(i)).str();
    if (module.lookupSymbol(name)) return reject();
    builder.setInsertionPointToEnd(module.getBody());
    auto member = func::FuncOp::create(
        builder, op->getLoc(), name,
        builder.getFunctionType(op->getOperandTypes(), op->getResultTypes()));
    member.setPrivate();
    auto block = member.addEntryBlock();
    OpBuilder body(block, block->begin());
    IRMapping mapping;
    for (auto [operand, argument] :
         llvm::zip_equal(op->getOperands(), block->getArguments()))
      mapping.map(operand, argument);
    auto copy = body.clone(*op, mapping);
    func::ReturnOp::create(body, op->getLoc(), copy->getResults());
    SmallVector<Attribute> bindings;
    for (unsigned n = 0; n < op->getNumOperands(); ++n)
      bindings.push_back(builder.getStringAttr("input_" + std::to_string(n)));
    member->setAttr("tessera.argument_bindings", builder.getArrayAttr(bindings));
    bindings.clear();
    for (unsigned n = 0; n < op->getNumResults(); ++n)
      bindings.push_back(builder.getStringAttr("output_" + std::to_string(n)));
    member->setAttr("tessera.result_bindings", builder.getArrayAttr(bindings));
    auto child = ModuleOp::create(module.getLoc());
    child->setAttr("tessera.target", target);
    child->setAttr("tessera.arch", arch);
    auto detached = cast<func::FuncOp>(member->clone());
    detached.setPublic();
    child.getBody()->push_back(detached);
    graphs.push_back(printModule(child));
    child.erase();
    members.push_back(member);
  }

  // Retain an actual typed func.call graph, not only a serialized plan.
  IRMapping rewritten;
  builder.setInsertionPoint(norm);
  for (unsigned i = 0; i < 2; ++i) {
    SmallVector<Value> inputs;
    for (Value operand : ops[i]->getOperands())
      inputs.push_back(rewritten.lookupOrDefault(operand));
    OperationState state(ops[i]->getLoc(), "func.call");
    state.addOperands(inputs);
    state.addTypes(ops[i]->getResultTypes());
    state.addAttribute("callee",
        FlatSymbolRefAttr::get(module.getContext(), members[i].getSymName()));
    auto call = builder.create(state);
    for (auto [oldValue, newValue] :
         llvm::zip_equal(ops[i]->getResults(), call->getResults()))
      rewritten.map(oldValue, newValue);
  }
  llvm::json::Array outputs;
  for (unsigned i = 0; i < 2; ++i) {
    outputs.push_back(ret.getOperand(i) == attention->getResult(0) ? 4 : 5);
    ret.setOperand(i, rewritten.lookup(ret.getOperand(i)));
  }
  attention->erase();
  norm->erase();
  SmallVector<StringAttr> stale;
  for (auto attr : root->getAttrs())
    if (attr.getName().getValue().starts_with("tessera.structured_cfg."))
      stale.push_back(attr.getName());
  for (auto name : stale) root->removeAttr(name);
  root->setAttr("tessera.native.source_program", builder.getUnitAttr());

  llvm::json::Array buffers;
  for (auto [i, type] : llvm::enumerate(types)) {
    llvm::json::Array shape;
    int64_t elements = 1;
    for (int64_t dim : type.getShape()) {
      shape.push_back(dim); elements *= dim;
    }
    bool input = i < 3;
    buffers.push_back(llvm::json::Object{
      {"id", int64_t(i)}, {"shape", std::move(shape)},
      {"storage", type.getElementType().isF16() ? "f16" :
                  type.getElementType().isBF16() ? "bf16" : "f32"},
      {"bytes", elements * (type.getElementType().isF32() ? 4 : 2)},
      {"ownership", input ? "readonly_input" : i == 3 ? "private_scratch" : "returned_output"},
      {"first_write", input ? -1 : i == 3 ? 0 : 1},
      {"last_read", input ? (i == q.getArgNumber() ? 0 : 1) : i == 3 ? 1 : 2}});
  }
  llvm::json::Array steps;
  steps.push_back(llvm::json::Object{
    {"step", 0}, {"member", members[0].getSymName().str()},
    {"inputs", llvm::json::Array{int64_t(q.getArgNumber())}},
    {"outputs", llvm::json::Array{3}}});
  steps.push_back(llvm::json::Object{
    {"step", 1}, {"member", members[1].getSymName().str()},
    {"inputs", llvm::json::Array{3, int64_t(k.getArgNumber()), int64_t(v.getArgNumber())}},
    {"outputs", llvm::json::Array{4, 5}}});
  llvm::json::Object plan{
    {"schema", "tessera.native.source_program.v1"},
    {"kind", "norm_attention_saved"}, {"execution_state", "artifact_only"},
    {"target", target.getValue().str()}, {"architecture", arch.getValue().str()},
    {"root", root.getSymName().str()}, {"source_graph_ir", original},
    {"source_hash", llvm::toHex(llvm::SHA256::hash(
         llvm::arrayRefFromStringRef(original)), true)},
    {"outlined_graph_ir", printModule(module)},
    {"role_indices", llvm::json::Array{int64_t(q.getArgNumber()),
         int64_t(k.getArgNumber()), int64_t(v.getArgNumber())}},
    {"buffers", std::move(buffers)}, {"steps", std::move(steps)},
    {"outputs", std::move(outputs)}, {"member_graphs", std::move(graphs)}};
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << llvm::json::Value(std::move(plan)); stream.flush();
  module->setAttr("tessera.native.source_program_json",
                  builder.getStringAttr(llvm::encodeBase64(text)));
  return success();
}
} // namespace tessera::source_program
