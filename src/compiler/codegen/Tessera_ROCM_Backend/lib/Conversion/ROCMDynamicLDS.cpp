//===- ROCMDynamicLDS.cpp - launch-sized LDS materialization --------------===//
//
// memref.alloca with memory space 3 lowers to llvm.alloca addrspace(3), which
// is private stack allocation with a non-default pointer type, not HIP dynamic
// LDS. Replace runtime-sized arenas in each kernel with slices of one AMDGPU
// external zero-length workgroup symbol. hipModuleLaunchKernel's sharedMemBytes
// argument supplies the storage for that symbol.
//
// Runtime byte arenas are colored into interference slots using MLIR SSA
// liveness. Arenas whose pointer lifetimes cannot overlap (including nested
// branch alternatives and loop-local values from distinct lifetime regions)
// reuse one slot. Values live together -- such as an outer arena carried across
// a loop and the loop-local arena -- interfere and receive distinct slots.
//
//   slot_bytes[s] = max(bytes[a] for a in slot[s])
//   slot_offset[0] = 0
//   slot_offset[s] = align_up(slot_offset[s-1] + slot_bytes[s-1], 16)
//   sharedMemBytes = align_up(slot_offset[last] + slot_bytes[last], 16)
//
// TileBufferArenaPass emits byte arenas, so the alloca element count is already
// the runtime byte expression. The first executable contract requires those
// sizes to be kernel arguments, which makes every slot maximum available on
// every CFG path. The function carries slot descriptors and the
// `aligned_sum_of_slot_maxima` reduction name so launch metadata evaluates the
// same expression.

#include "TesseraROCM/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

static bool typeCarriesPointer(Type type) {
  if (isa<LLVM::LLVMPointerType>(type))
    return true;
  if (auto array = dyn_cast<LLVM::LLVMArrayType>(type))
    return typeCarriesPointer(array.getElementType());
  if (auto structure = dyn_cast<LLVM::LLVMStructType>(type))
    return llvm::any_of(structure.getBody(), typeCarriesPointer);
  return false;
}

static bool enqueueForwardedAliases(Operation *user, Value value,
                                    SmallVectorImpl<Value> &worklist) {
  auto branch = dyn_cast<BranchOpInterface>(user);
  if (!branch)
    return false;

  bool forwarded = false;
  for (unsigned successor = 0; successor < user->getNumSuccessors();
       ++successor) {
    SuccessorOperands operands = branch.getSuccessorOperands(successor);
    Block *successorBlock = user->getSuccessor(successor);
    for (unsigned argument = 0; argument < operands.size(); ++argument) {
      if (operands[argument] != value)
        continue;
      worklist.push_back(successorBlock->getArgument(argument));
      forwarded = true;
    }
  }
  return forwarded;
}

static llvm::SmallPtrSet<Operation *, 32>
resolveAllocationLifetime(const Liveness &liveness, Value root) {
  llvm::SmallPtrSet<Operation *, 32> live;
  llvm::SmallDenseSet<Value, 32> visited;
  SmallVector<Value> worklist{root};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (Operation *op : liveness.resolveLiveness(value))
      live.insert(op);
    for (Operation *user : value.getUsers()) {
      live.insert(user);
      bool propagated = false;
      for (Value result : user->getResults())
        if (typeCarriesPointer(result.getType())) {
          worklist.push_back(result);
          propagated = true;
        }
      propagated |= enqueueForwardedAliases(user, value, worklist);
      if (propagated)
        continue;

      StringRef name = user->getName().getStringRef();
      bool terminalMemoryAddress =
          (name == "llvm.load" && user->getOperand(0) == value) ||
          (name == "llvm.store" && user->getNumOperands() > 1 &&
           user->getOperand(1) == value);
      if (terminalMemoryAddress)
        continue;

      // A pointer consumed without a pointer-carrying result may escape (store
      // as data, call, ptrtoint/inttoptr round-trip, return, or an unknown
      // dialect op). Keep the arena live for the whole kernel rather than
      // allowing a later slot to alias storage whose lifetime is unknowable.
      liveness.getOperation()->walk(
          [&](Operation *operation) { live.insert(operation); });
    }
  }
  return live;
}

static bool interfere(const Liveness &liveness, Value lhs, Value rhs) {
  llvm::SmallPtrSet<Operation *, 32> lhsLive =
      resolveAllocationLifetime(liveness, lhs);
  for (Operation *op : resolveAllocationLifetime(liveness, rhs))
    if (lhsLive.contains(op))
      return true;
  return false;
}

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

static bool collectKernelArgumentLeaves(
    Value value, Block *entry, SmallVectorImpl<BlockArgument> &leaves,
    llvm::SmallDenseSet<Value, 16> &visited) {
  if (!visited.insert(value).second)
    return true;
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return false;
  if (argument.getOwner() == entry) {
    if (!llvm::is_contained(leaves, argument))
      leaves.push_back(argument);
    return true;
  }

  bool foundIncoming = false;
  for (Block *predecessor : argument.getOwner()->getPredecessors()) {
    Operation *terminator = predecessor->getTerminator();
    auto branch = dyn_cast<BranchOpInterface>(terminator);
    if (!branch)
      return false;
    bool foundSuccessor = false;
    for (unsigned successor = 0;
         successor < terminator->getNumSuccessors(); ++successor) {
      if (terminator->getSuccessor(successor) != argument.getOwner())
        continue;
      foundSuccessor = true;
      SuccessorOperands operands = branch.getSuccessorOperands(successor);
      Value incoming = operands[argument.getArgNumber()];
      if (!incoming ||
          !collectKernelArgumentLeaves(incoming, entry, leaves, visited))
        return false;
    }
    if (!foundSuccessor)
      return false;
    foundIncoming = true;
  }
  return foundIncoming;
}

struct ROCMDynamicLDS
    : PassWrapper<ROCMDynamicLDS, OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ROCMDynamicLDS)

  StringRef getArgument() const final {
    return "rocm-materialize-dynamic-lds";
  }
  StringRef getDescription() const final {
    return "Pack runtime-sized addrspace(3) byte allocas per ROCm kernel into "
           "one launch-sized external LDS allocation";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    gpu::GPUModuleOp module = getOperation();
    SmallVector<LLVM::AllocaOp> dynamicLds;
    module.walk([&](LLVM::AllocaOp alloca) {
      auto pointer = dyn_cast<LLVM::LLVMPointerType>(alloca.getType());
      if (pointer && pointer.getAddressSpace() == 3)
        dynamicLds.push_back(alloca);
    });
    if (dynamicLds.empty())
      return;

    OpBuilder globalBuilder(module.getBodyRegion());
    globalBuilder.setInsertionPointToStart(module.getBody());
    auto arrayType = LLVM::LLVMArrayType::get(
        IntegerType::get(&getContext(), 8), 0);
    constexpr StringLiteral symbol = "__tessera_dynamic_lds";
    auto global = module.lookupSymbol<LLVM::GlobalOp>(symbol);
    if (!global) {
      global = LLVM::GlobalOp::create(
          globalBuilder, module.getLoc(), arrayType, /*isConstant=*/false,
          LLVM::Linkage::External, symbol, Attribute(), /*alignment=*/16,
          /*addrSpace=*/3, /*dsoLocal=*/false, /*thread_local_=*/ false,
          SymbolRefAttr(), ArrayRef<NamedAttribute>{},
          ArrayRef<Attribute>{});
    }

    llvm::MapVector<Operation *, SmallVector<LLVM::AllocaOp>> perKernel;
    for (LLVM::AllocaOp alloca : dynamicLds)
      if (auto fn = alloca->getParentOfType<LLVM::LLVMFuncOp>())
        perKernel[fn.getOperation()].push_back(alloca);

    for (auto &[fnOp, allocas] : perKernel) {
      auto fn = cast<LLVM::LLVMFuncOp>(fnOp);
      if (llvm::any_of(allocas, [](LLVM::AllocaOp alloca) {
            return !alloca.getElemType().isInteger(8);
          })) {
        allocas.front().emitOpError(
            "runtime LDS packing expects byte arenas emitted by "
            "TileBufferArenaPass");
        return signalPassFailure();
      }

      Type countType = allocas.front().getArraySize().getType();
      llvm::DenseMap<Operation *, SmallVector<BlockArgument>> countLeaves;
      for (LLVM::AllocaOp alloca : allocas) {
        Value count = alloca.getArraySize();
        SmallVector<BlockArgument> leaves;
        llvm::SmallDenseSet<Value, 16> visited;
        if (!collectKernelArgumentLeaves(
                count, &fn.getBody().front(), leaves, visited) ||
            leaves.empty()) {
          alloca.emitOpError(
              "ROCM_DYNAMIC_LDS_SIZE_NOT_KERNEL_ARGUMENT: runtime LDS size "
              "expressions must currently be LLVM kernel arguments or CFG "
              "block arguments that recursively forward kernel arguments");
          return signalPassFailure();
        }
        if (count.getType() != countType ||
            llvm::any_of(leaves, [&](BlockArgument leaf) {
              return leaf.getType() != countType;
            })) {
          alloca.emitOpError(
              "ROCM_DYNAMIC_LDS_SIZE_NOT_KERNEL_ARGUMENT: all runtime LDS "
              "byte-count kernel arguments must use one integer type");
          return signalPassFailure();
        }
        llvm::sort(leaves, [](BlockArgument lhs, BlockArgument rhs) {
          return lhs.getArgNumber() < rhs.getArgNumber();
        });
        countLeaves[alloca.getOperation()] = std::move(leaves);
      }

      Liveness liveness(fn);
      SmallVector<SmallVector<LLVM::AllocaOp>> slots;
      for (LLVM::AllocaOp alloca : allocas) {
        auto available = llvm::find_if(
            slots, [&](ArrayRef<LLVM::AllocaOp> slot) {
              return llvm::none_of(slot, [&](LLVM::AllocaOp resident) {
                return interfere(liveness, alloca.getResult(),
                                 resident.getResult());
              });
            });
        if (available == slots.end()) {
          slots.emplace_back();
          slots.back().push_back(alloca);
        } else {
          available->push_back(alloca);
        }
      }

      SmallVector<Attribute> slotAttrs;
      SmallVector<Attribute> flatDescriptors;
      llvm::DenseMap<Operation *, int64_t> arenaOrdinals;
      for (auto [ordinal, alloca] : llvm::enumerate(allocas))
        arenaOrdinals[alloca.getOperation()] =
            static_cast<int64_t>(ordinal);

      for (auto [slotOrdinal, slot] : llvm::enumerate(slots)) {
        SmallVector<Attribute> descriptors;
        for (LLVM::AllocaOp alloca : slot) {
          OpBuilder builder(alloca);
          Location loc = alloca.getLoc();
          Value cursor = LLVM::ConstantOp::create(
              builder, loc, countType, builder.getIntegerAttr(countType, 0));
          for (int64_t prior = 0;
               prior < static_cast<int64_t>(slotOrdinal); ++prior) {
            ArrayRef<BlockArgument> firstLeaves =
                countLeaves[slots[prior].front().getOperation()];
            Value slotMaximum = firstLeaves.front();
            for (BlockArgument leaf : llvm::drop_begin(firstLeaves))
              slotMaximum = emitUnsignedMax(builder, loc, slotMaximum, leaf);
            for (LLVM::AllocaOp alternative : llvm::drop_begin(slots[prior]))
              for (BlockArgument leaf :
                   countLeaves[alternative.getOperation()])
                slotMaximum =
                    emitUnsignedMax(builder, loc, slotMaximum, leaf);
            cursor = emitAligned(builder, loc, cursor);
            cursor = LLVM::AddOp::create(builder, loc, cursor, slotMaximum)
                         .getResult();
          }
          Value offset = emitAligned(builder, loc, cursor);
          Value base = LLVM::AddressOfOp::create(builder, loc, global);
          Value slice = LLVM::GEPOp::create(
              builder, loc, base.getType(), builder.getI8Type(), base,
              ValueRange{offset});
          alloca.replaceAllUsesWith(slice);

          NamedAttrList descriptor;
          descriptor.append(
              "ordinal",
              builder.getI64IntegerAttr(arenaOrdinals[alloca.getOperation()]));
          descriptor.append("slot",
                            builder.getI64IntegerAttr(
                                static_cast<int64_t>(slotOrdinal)));
          descriptor.append("alignment", builder.getI64IntegerAttr(16));
          descriptor.append("element_bytes", builder.getI64IntegerAttr(1));
          ArrayRef<BlockArgument> leaves =
              countLeaves[alloca.getOperation()];
          if (leaves.size() == 1) {
            descriptor.append("count_source",
                              builder.getStringAttr("argument"));
            descriptor.append(
                "count_argument",
                builder.getI64IntegerAttr(leaves.front().getArgNumber()));
          } else {
            descriptor.append(
                "count_source",
                builder.getStringAttr("cfg_argument_expression"));
            SmallVector<Attribute> arguments;
            for (BlockArgument leaf : leaves)
              arguments.push_back(
                  builder.getI64IntegerAttr(leaf.getArgNumber()));
            descriptor.append("count_arguments",
                              builder.getArrayAttr(arguments));
          }
          Attribute descriptorAttr =
              descriptor.getDictionary(&getContext());
          descriptors.push_back(descriptorAttr);
          flatDescriptors.push_back(descriptorAttr);
        }
        OpBuilder builder(fn);
        NamedAttrList slotDescriptor;
        slotDescriptor.append("ordinal",
                              builder.getI64IntegerAttr(
                                  static_cast<int64_t>(slotOrdinal)));
        slotDescriptor.append("arenas", builder.getArrayAttr(descriptors));
        slotAttrs.push_back(slotDescriptor.getDictionary(&getContext()));
      }
      for (LLVM::AllocaOp alloca : allocas)
        alloca.erase();
      OpBuilder builder(fn);
      fn->setAttr("tessera.rocm.dynamic_lds_launch_bytes",
                  builder.getUnitAttr());
      fn->setAttr("tessera.rocm.dynamic_lds_launch_reduction",
                  builder.getStringAttr("aligned_sum_of_slot_maxima"));
      fn->setAttr("tessera.rocm.dynamic_lds_slots",
                  builder.getArrayAttr(slotAttrs));
      fn->setAttr("tessera.rocm.dynamic_lds_packed_arenas",
                  builder.getArrayAttr(flatDescriptors));
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createROCMDynamicLDSPass() {
  return std::make_unique<ROCMDynamicLDS>();
}
