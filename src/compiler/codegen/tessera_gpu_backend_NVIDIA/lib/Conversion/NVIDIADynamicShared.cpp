//===- NVIDIADynamicShared.cpp - launch-sized CUDA shared memory ----------===//
//
// NVPTX dynamic shared memory is an external zero-length addrspace(3) symbol;
// cuLaunchKernel's sharedMemBytes argument provides its physical extent.  This
// pass replaces runtime-sized byte allocas with slices of that symbol and
// records the launch expression needed by the descriptor/runtime boundary.
//
// NVIDIA owns this physical lowering.  It deliberately does not reuse ROCm's
// LDS pass or its schedule: the common contract is only the 16-byte-aligned
// interference-slot expression.

#include "tessera/gpu/BackendRegistration.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

using namespace mlir;

namespace {

static Value emitAligned(OpBuilder &builder, Location loc, Value value) {
  Type type = value.getType();
  Value fifteen = LLVM::ConstantOp::create(
      builder, loc, type, builder.getIntegerAttr(type, 15));
  Value minusSixteen = LLVM::ConstantOp::create(
      builder, loc, type, builder.getIntegerAttr(type, -16));
  Value padded =
      LLVM::AddOp::create(builder, loc, value, fifteen).getResult();
  return LLVM::AndOp::create(builder, loc, padded, minusSixteen).getResult();
}

static Value emitUnsignedMax(OpBuilder &builder, Location loc, Value lhs,
                             Value rhs) {
  Value greater = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::ugt, lhs, rhs);
  return LLVM::SelectOp::create(builder, loc, greater, lhs, rhs).getResult();
}

static DictionaryAttr expressionAttr(OpBuilder &builder, StringRef kind,
                                     ArrayRef<Attribute> operands = {}) {
  NamedAttrList attributes;
  attributes.append("kind", builder.getStringAttr(kind));
  if (!operands.empty())
    attributes.append("operands", builder.getArrayAttr(operands));
  return attributes.getDictionary(builder.getContext());
}

static std::optional<Attribute>
serializeSizeExpression(Value value, Block *entry, OpBuilder &builder,
                        llvm::SmallDenseSet<Value, 32> &visiting) {
  if (!visiting.insert(value).second)
    return std::nullopt;
  auto finish = [&](std::optional<Attribute> result) {
    visiting.erase(value);
    return result;
  };
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (argument.getOwner() == entry) {
      NamedAttrList attributes;
      attributes.append("kind", builder.getStringAttr("argument"));
      attributes.append(
          "index", builder.getI64IntegerAttr(argument.getArgNumber()));
      return finish(attributes.getDictionary(builder.getContext()));
    }
    SmallVector<Attribute> incomingExpressions;
    llvm::SmallPtrSet<Operation *, 8> visitedTerminators;
    for (Block *predecessor : argument.getOwner()->getPredecessors()) {
      Operation *terminator = predecessor->getTerminator();
      if (!visitedTerminators.insert(terminator).second)
        continue;
      auto branch = dyn_cast<BranchOpInterface>(terminator);
      if (!branch)
        return finish(std::nullopt);
      bool found = false;
      for (unsigned successor = 0;
           successor < terminator->getNumSuccessors();
           ++successor) {
        if (terminator->getSuccessor(successor) !=
            argument.getOwner())
          continue;
        SuccessorOperands operands =
            branch.getSuccessorOperands(successor);
        if (argument.getArgNumber() >= operands.size())
          return finish(std::nullopt);
        auto incoming = serializeSizeExpression(
            operands[argument.getArgNumber()], entry, builder, visiting);
        if (!incoming)
          return finish(std::nullopt);
        incomingExpressions.push_back(*incoming);
        found = true;
      }
      if (!found)
        return finish(std::nullopt);
    }
    if (incomingExpressions.empty())
      return finish(std::nullopt);
    if (incomingExpressions.size() == 1)
      return finish(incomingExpressions.front());
    return finish(
        expressionAttr(builder, "path_max", incomingExpressions));
  }

  Operation *definition = value.getDefiningOp();
  if (!definition)
    return finish(std::nullopt);
  if (auto constant = dyn_cast<LLVM::ConstantOp>(definition)) {
    auto integer = dyn_cast<IntegerAttr>(constant.getValue());
    if (!integer || integer.getInt() < 0)
      return finish(std::nullopt);
    NamedAttrList attributes;
    attributes.append("kind", builder.getStringAttr("constant"));
    attributes.append("value", builder.getI64IntegerAttr(integer.getInt()));
    return finish(attributes.getDictionary(builder.getContext()));
  }

  StringRef name = definition->getName().getStringRef();
  StringRef kind;
  SmallVector<unsigned> operandIndices;
  if (name == "llvm.add") {
    kind = "add";
    operandIndices = {0, 1};
  } else if (name == "llvm.mul") {
    kind = "multiply";
    operandIndices = {0, 1};
  } else if (name == "llvm.select") {
    // The host launch boundary cannot in general reproduce a device-only
    // predicate. Conservatively allocate the larger branch expression.
    kind = "path_max";
    operandIndices = {1, 2};
  } else if (name == "llvm.zext" || name == "llvm.sext" ||
             name == "llvm.trunc") {
    operandIndices = {0};
  } else {
    return finish(std::nullopt);
  }
  SmallVector<Attribute> operands;
  for (unsigned index : operandIndices) {
    auto operand = serializeSizeExpression(
        definition->getOperand(index), entry, builder, visiting);
    if (!operand)
      return finish(std::nullopt);
    operands.push_back(*operand);
  }
  if (kind.empty())
    return finish(operands.front());
  return finish(expressionAttr(builder, kind, operands));
}

static FailureOr<Value>
emitSizeExpression(Attribute attribute, Block &entry, OpBuilder &builder,
                   Location loc, Type type) {
  auto expression = dyn_cast<DictionaryAttr>(attribute);
  if (!expression)
    return failure();
  auto kind = expression.getAs<StringAttr>("kind");
  if (!kind)
    return failure();
  if (kind.getValue() == "argument") {
    auto index = expression.getAs<IntegerAttr>("index");
    if (!index || index.getInt() < 0 ||
        static_cast<unsigned>(index.getInt()) >= entry.getNumArguments())
      return failure();
    Value argument = entry.getArgument(index.getInt());
    if (argument.getType() != type)
      return failure();
    return argument;
  }
  if (kind.getValue() == "constant") {
    auto value = expression.getAs<IntegerAttr>("value");
    if (!value)
      return failure();
    return LLVM::ConstantOp::create(
        builder, loc, type, builder.getIntegerAttr(type, value.getInt()))
        .getResult();
  }
  auto operands = expression.getAs<ArrayAttr>("operands");
  if (!operands || operands.empty())
    return failure();
  SmallVector<Value> values;
  for (Attribute operand : operands) {
    FailureOr<Value> emitted =
        emitSizeExpression(operand, entry, builder, loc, type);
    if (failed(emitted))
      return failure();
    values.push_back(*emitted);
  }
  Value result = values.front();
  for (Value rhs : llvm::drop_begin(values)) {
    if (kind.getValue() == "add")
      result = LLVM::AddOp::create(builder, loc, result, rhs).getResult();
    else if (kind.getValue() == "multiply")
      result = LLVM::MulOp::create(builder, loc, result, rhs).getResult();
    else if (kind.getValue() == "path_max")
      result = emitUnsignedMax(builder, loc, result, rhs);
    else
      return failure();
  }
  return result;
}

static bool interfere(const Liveness &liveness, Value lhs, Value rhs) {
  llvm::SmallPtrSet<Operation *, 32> left;
  for (Operation *op : liveness.resolveLiveness(lhs))
    left.insert(op);
  for (Operation *op : liveness.resolveLiveness(rhs))
    if (left.contains(op))
      return true;
  return false;
}

struct NVIDIADynamicSharedPass
    : PassWrapper<NVIDIADynamicSharedPass,
                  OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NVIDIADynamicSharedPass)

  StringRef getArgument() const final {
    return "nvidia-materialize-dynamic-shared";
  }
  StringRef getDescription() const final {
    return "Pack runtime-sized NVPTX addrspace(3) byte arenas into one "
           "launch-provided dynamic shared-memory symbol";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    gpu::GPUModuleOp module = getOperation();
    SmallVector<LLVM::AllocaOp> dynamicShared;
    module.walk([&](LLVM::AllocaOp alloca) {
      auto pointer = dyn_cast<LLVM::LLVMPointerType>(alloca.getType());
      if (pointer && pointer.getAddressSpace() == 3)
        dynamicShared.push_back(alloca);
    });
    if (dynamicShared.empty())
      return;

    OpBuilder globalBuilder(module.getBodyRegion());
    globalBuilder.setInsertionPointToStart(module.getBody());
    auto arrayType =
        LLVM::LLVMArrayType::get(IntegerType::get(&getContext(), 8), 0);
    constexpr StringLiteral symbol = "__tessera_dynamic_smem";
    auto global = module.lookupSymbol<LLVM::GlobalOp>(symbol);
    if (!global) {
      global = LLVM::GlobalOp::create(
          globalBuilder, module.getLoc(), arrayType, /*isConstant=*/false,
          LLVM::Linkage::External, symbol, Attribute(), /*alignment=*/16,
          /*addrSpace=*/3, /*dsoLocal=*/false, /*thread_local_=*/false,
          SymbolRefAttr(), ArrayRef<NamedAttribute>{},
          ArrayRef<Attribute>{});
    }

    llvm::MapVector<Operation *, SmallVector<LLVM::AllocaOp>> perKernel;
    for (LLVM::AllocaOp alloca : dynamicShared)
      if (auto fn = alloca->getParentOfType<LLVM::LLVMFuncOp>())
        perKernel[fn.getOperation()].push_back(alloca);

    for (auto &[fnOperation, allocas] : perKernel) {
      auto fn = cast<LLVM::LLVMFuncOp>(fnOperation);
      Block &entry = fn.getBody().front();
      Type countType = allocas.front().getArraySize().getType();
      llvm::DenseMap<Operation *, Attribute> countExpressions;
      for (LLVM::AllocaOp alloca : allocas) {
        if (!alloca.getElemType().isInteger(8)) {
          alloca.emitOpError(
              "NVIDIA dynamic shared-memory packing requires byte arenas");
          return signalPassFailure();
        }
        if (alloca.getArraySize().getType() != countType) {
          alloca.emitOpError(
              "NVIDIA dynamic shared-memory byte counts must use one "
              "integer type");
          return signalPassFailure();
        }
        OpBuilder builder(alloca);
        llvm::SmallDenseSet<Value, 32> visiting;
        auto expression = serializeSizeExpression(
            alloca.getArraySize(), &entry, builder, visiting);
        if (!expression) {
          alloca.emitOpError(
              "NVIDIA dynamic shared-memory size must be a serializable "
              "kernel argument, CFG-forwarded path maximum, or non-negative "
              "constant/add/multiply expression");
          return signalPassFailure();
        }
        countExpressions[alloca.getOperation()] = *expression;
      }

      Liveness liveness(fn);
      SmallVector<SmallVector<LLVM::AllocaOp>> slots;
      for (LLVM::AllocaOp alloca : allocas) {
        auto reusable = llvm::find_if(
            slots, [&](ArrayRef<LLVM::AllocaOp> slot) {
              return llvm::none_of(slot, [&](LLVM::AllocaOp resident) {
                return interfere(liveness, alloca.getResult(),
                                 resident.getResult());
              });
            });
        if (reusable == slots.end())
          slots.push_back({alloca});
        else
          reusable->push_back(alloca);
      }

      SmallVector<Attribute> slotAttributes;
      for (auto [slotOrdinal, slot] : llvm::enumerate(slots)) {
        SmallVector<Attribute> alternatives;
        for (LLVM::AllocaOp alloca : slot) {
          OpBuilder builder(alloca);
          Location loc = alloca.getLoc();
          Value cursor = LLVM::ConstantOp::create(
              builder, loc, countType,
              builder.getIntegerAttr(countType, 0));
          for (int64_t prior = 0;
               prior < static_cast<int64_t>(slotOrdinal); ++prior) {
            FailureOr<Value> first = emitSizeExpression(
                countExpressions[slots[prior].front().getOperation()],
                entry, builder, loc, countType);
            if (failed(first)) {
              alloca.emitOpError(
                  "failed to reconstruct NVIDIA dynamic shared-memory "
                  "launch expression");
              return signalPassFailure();
            }
            Value maximum = *first;
            for (LLVM::AllocaOp alternative :
                 llvm::drop_begin(slots[prior])) {
              FailureOr<Value> emitted = emitSizeExpression(
                  countExpressions[alternative.getOperation()],
                  entry, builder, loc, countType);
              if (failed(emitted)) {
                alloca.emitOpError(
                    "failed to reconstruct NVIDIA dynamic shared-memory "
                    "launch expression");
                return signalPassFailure();
              }
              maximum =
                  emitUnsignedMax(builder, loc, maximum, *emitted);
            }
            cursor = emitAligned(builder, loc, cursor);
            cursor =
                LLVM::AddOp::create(builder, loc, cursor, maximum).getResult();
          }
          Value offset = emitAligned(builder, loc, cursor);
          Value base = LLVM::AddressOfOp::create(builder, loc, global);
          Value slice = LLVM::GEPOp::create(
              builder, loc, base.getType(), builder.getI8Type(), base,
              ValueRange{offset});
          alloca.replaceAllUsesWith(slice);

          NamedAttrList arena;
          arena.append(
              "expression",
              countExpressions[alloca.getOperation()]);
          arena.append("alignment", builder.getI64IntegerAttr(16));
          alternatives.push_back(arena.getDictionary(&getContext()));
        }
        OpBuilder builder(fn);
        NamedAttrList descriptor;
        descriptor.append(
            "ordinal",
            builder.getI64IntegerAttr(static_cast<int64_t>(slotOrdinal)));
        descriptor.append("alternatives",
                          builder.getArrayAttr(alternatives));
        slotAttributes.push_back(descriptor.getDictionary(&getContext()));
      }
      for (LLVM::AllocaOp alloca : allocas)
        alloca.erase();

      OpBuilder builder(fn);
      fn->setAttr("tessera.nvidia.dynamic_smem_launch_bytes",
                  builder.getUnitAttr());
      fn->setAttr("tessera.nvidia.dynamic_smem_launch_reduction",
                  builder.getStringAttr("aligned_sum_of_slot_maxima"));
      fn->setAttr("tessera.nvidia.dynamic_smem_slots",
                  builder.getArrayAttr(slotAttributes));
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> tessera::createNVIDIADynamicSharedPass() {
  return std::make_unique<NVIDIADynamicSharedPass>();
}
