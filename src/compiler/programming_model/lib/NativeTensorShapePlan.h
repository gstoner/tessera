// Native MLIR shape computation accompanies, but never replaces, device kernels.
#pragma once
#include "Tessera/IR/NativeTensorProgramContract.h"

namespace tensor_program {
// Emitted at the leaf's physical ABI selection point. Its parameters are the
// operand/result dimensions in ABI order, not inferred operation names.
template <typename Emit>
inline void attachShapeProgram(mlir::LLVM::LLVMFuncOp kernel,
                                mlir::Operation *graph, Emit emit) {
  using namespace mlir;
  auto member = graph->getParentOfType<ModuleOp>();
  if (!member->hasAttr("tessera.native.tensor_program_shape_profile")) return;
  Location loc = graph->getLoc();
  OwningOpRef<ModuleOp> plan(ModuleOp::create(loc));
  OpBuilder builder(plan->getBody(), plan->getBody()->end());
  auto scalars = kernel->getAttrOfType<DenseI64ArrayAttr>("tessera.native.program_scalars");
  unsigned buffers = graph->getNumOperands() + 1;
  auto function = func::FuncOp::create(builder, loc, "__tessera_native_member_shape",
      builder.getFunctionType(SmallVector<Type>(2 * buffers, builder.getI64Type()),
                             SmallVector<Type>(scalars.size() + 3, builder.getI64Type())));
  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  SmallVector<Value> dimensions(entry->getArguments().begin(), entry->getArguments().end());
  auto values = emit(builder, loc, ArrayRef<Value>(dimensions));
  func::ReturnOp::create(builder, loc, values);
  std::string text;
  llvm::raw_string_ostream stream(text); plan->print(stream); stream.flush();
  kernel->setAttr("tessera.native.program_shape_ir", builder.getStringAttr(text));
}

inline mlir::Value shapeConstant(mlir::OpBuilder &builder, mlir::Location loc, int64_t value) {
  return mlir::arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(value));
}
inline mlir::Value shapeCeilDiv(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value value, int64_t divisor) {
  auto rounded = mlir::arith::AddIOp::create(builder, loc, value,
      shapeConstant(builder, loc, divisor - 1));
  return mlir::arith::DivUIOp::create(builder, loc, rounded,
      shapeConstant(builder, loc, divisor));
}

// Build a pure host shape function. No output is stored before every root,
// derived dimension and launch extent is checked. C++ executes the compiled
// function before committing retained-frame state or enqueuing any GPU work.
inline mlir::FailureOr<llvm::json::Object> exportShapeProgram(
    mlir::ModuleOp module, mlir::func::FuncOp host,
    const llvm::DenseMap<mlir::Value, int64_t> &ids, size_t stageCount) {
  using namespace mlir;
  auto reject = [&]() -> FailureOr<llvm::json::Object> {
    module.emitError("native shape export requires verified pure member shape functions and capacity bindings");
    return failure();
  };
  Operation *anchor = nullptr;
  for (Operation &op : host.getBody().front())
    if (tessera::native_tensor_program::isInvocation(&op)) { anchor = &op; break; }
  tessera::native_tensor_program::ShapeMemoryContract memory;
  if (!anchor || failed(memory.verify(module, host, anchor))) return failure();
  auto variables = module->getAttrOfType<DenseI64ArrayAttr>("tessera.native.tensor_program_shape_variables");
  Location loc = module.getLoc();
  OwningOpRef<ModuleOp> plan(ModuleOp::create(loc));
  OpBuilder builder(plan->getBody(), plan->getBody()->end());
  auto ptr = LLVM::LLVMPointerType::get(module.getContext());
  auto function = func::FuncOp::create(builder, loc, "__tessera_tensor_shape_plan",
      builder.getFunctionType({ptr, ptr}, {builder.getI32Type()}));
  Block *entry = function.addEntryBlock(), *invalid = function.addBlock();
  builder.setInsertionPointToStart(invalid);
  func::ReturnOp::create(builder, loc,
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(1)).getResult());
  builder.setInsertionPointToStart(entry);
  auto integer = [&](int64_t value) { return shapeConstant(builder, loc, value); };
  auto address = [&](Value base, int64_t index) -> Value {
    return LLVM::GEPOp::create(builder, loc, base.getType(), builder.getI64Type(),
                               base, ValueRange{integer(index)});
  };
  auto guard = [&](Value valid) {
    Block *next = function.addBlock();
    cf::CondBranchOp::create(builder, loc, valid, next, ValueRange{}, invalid, ValueRange{});
    builder.setInsertionPointToStart(next);
  };
  auto compare = [&](arith::CmpIPredicate predicate, Value a, Value b) -> Value {
    return arith::CmpIOp::create(builder, loc, predicate, a, b);
  };
  auto positiveCapacity = [&](Value actual, int64_t capacity) {
    auto positive = compare(arith::CmpIPredicate::ugt, actual, integer(0));
    auto fits = compare(arith::CmpIPredicate::ule, actual, integer(capacity));
    guard(arith::AndIOp::create(builder, loc, positive, fits));
  };
  SmallVector<Value> roots, dimensions;
  for (unsigned i = 0; i < 2 * host.getNumArguments(); ++i)
    roots.push_back(LLVM::LoadOp::create(builder, loc, builder.getI64Type(), address(entry->getArgument(0), i)));
  for (unsigned symbol = 0; symbol < variables.size() / 3; ++symbol) {
    auto arg = variables[symbol * 3], axis = variables[symbol * 3 + 1];
    Value actual = roots[arg * 2 + axis];
    positiveCapacity(actual, variables[symbol * 3 + 2]);
    dimensions.push_back(actual);
  }
  for (unsigned i = 0; i < host.getNumArguments(); ++i)
    for (unsigned axis = 0; axis < 2; ++axis) {
      int64_t symbol = memory.bindings[i][axis];
      Value wanted = symbol < 0 ? integer(memory.capacities[i].getDimSize(axis)) : dimensions[symbol];
      guard(compare(arith::CmpIPredicate::eq, roots[i * 2 + axis], wanted));
    }
  SmallVector<Value> output;
  for (auto [i, capacity] : llvm::enumerate(memory.capacities))
    for (unsigned axis = 0; axis < 2; ++axis) {
      int64_t symbol = memory.bindings[i][axis];
      output.push_back(symbol < 0 ? integer(capacity.getDimSize(axis)) : dimensions[symbol]);
    }
  for (Operation &op : host.getBody().front()) {
    if (!tessera::native_tensor_program::isInvocation(&op)) continue;
    auto source = op.getAttrOfType<StringAttr>("shape_ir");
    auto scalars = op.getAttrOfType<DenseI64ArrayAttr>("scalars");
    auto geometry = op.getAttrOfType<DenseI64ArrayAttr>("geometry");
    if (!source || source.getValue().empty() || !scalars || !geometry) return reject();
    auto member = parseSourceString<ModuleOp>(source.getValue(), module.getContext());
    if (!member || failed(verify(*member))) return reject();
    SmallVector<func::FuncOp> functions(member->getOps<func::FuncOp>());
    if (functions.size() != 1 || std::distance(member->getBody()->begin(), member->getBody()->end()) != 1) return reject();
    auto leaf = functions.front();
    if (!leaf.getBody().hasOneBlock() || leaf.getNumArguments() != 2 * op.getNumOperands() ||
        leaf.getNumResults() != scalars.size() + 3 ||
        !llvm::all_of(leaf.getArgumentTypes(), [](Type type) { return type.isInteger(64); }) ||
        !llvm::all_of(leaf.getResultTypes(), [](Type type) { return type.isInteger(64); })) return reject();
    auto returned = dyn_cast<func::ReturnOp>(leaf.getBody().front().getTerminator());
    if (!returned) return reject();
    IRMapping mapping;
    for (auto [i, value] : llvm::enumerate(op.getOperands())) {
      auto found = ids.find(value);
      if (found == ids.end()) return reject();
      mapping.map(leaf.getArgument(i * 2), output[found->second * 2]);
      mapping.map(leaf.getArgument(i * 2 + 1), output[found->second * 2 + 1]);
    }
    for (Operation &calculation : leaf.getBody().front().without_terminator()) {
      if (!isa<arith::ConstantOp,arith::MulIOp,arith::AddIOp,arith::DivUIOp>(calculation) ||
          calculation.getNumResults() != 1 || !calculation.getResult(0).getType().isInteger(64)) return reject();
      if (auto division = dyn_cast<arith::DivUIOp>(calculation)) {
        auto constant = division.getRhs().getDefiningOp<arith::ConstantOp>();
        auto value = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
        if (!value || value.getInt() <= 0) return reject();
      }
      builder.clone(calculation, mapping);
    }
    SmallVector<Value> values;
    for (auto [i, returnedValue] : llvm::enumerate(returned.getOperands())) {
      Value actual = mapping.lookup(returnedValue);
      int64_t capacity = i < scalars.size() ? scalars[i] : geometry[i - scalars.size()];
      positiveCapacity(actual, capacity);
      values.push_back(actual);
    }
    if (scalars.size() > 8) return reject();
    for (unsigned i = 0; i < 8; ++i)
      output.push_back(i < scalars.size() ? values[i] : integer(0));
    llvm::append_range(output, ArrayRef<Value>(values).drop_front(scalars.size()));
  }
  if (output.size() != 2 * memory.owned.size() + 11 * stageCount) return reject();
  for (auto [index, value] : llvm::enumerate(output))
    LLVM::StoreOp::create(builder, loc, value, address(entry->getArgument(1), index));
  func::ReturnOp::create(builder, loc,
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(0)).getResult());
  if (failed(verify(*plan))) return reject();
  std::string text;
  llvm::raw_string_ostream stream(text); plan->print(stream); stream.flush();
  return llvm::json::Object{
    {"entry", function.getName().str()}, {"root_words", int64_t(roots.size())},
    {"plan_words", int64_t(output.size())}, {"ir", text},
    {"sha256", llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true)}};
}
} // namespace tensor_program
