//===- ROCMDynamicLDS.cpp - launch-sized LDS materialization --------------===//
//
// memref.alloca with memory space 3 lowers to llvm.alloca addrspace(3), which
// is private stack allocation with a non-default pointer type, not HIP dynamic
// LDS. Replace the single runtime-sized arena in a kernel with the AMDGPU
// external zero-length workgroup symbol. hipModuleLaunchKernel's sharedMemBytes
// argument supplies the storage for that symbol.
//
// Multiple independent byte arenas in one control-flow path are packed into
// that allocation with a deterministic 16-byte-aligned prefix sum. Distinct
// direct successor blocks of one branch reuse offset zero; their arenas cannot
// execute simultaneously, so the host launch contract is a maximum of aligned
// path totals:
//
//   path_offset[p][0] = 0
//   path_offset[p][i] = align_up(path_offset[p][i-1] + bytes[p][i-1], 16)
//   sharedMemBytes = max_p align_up(path_offset[p][last] + bytes[p][last], 16)
//
// TileBufferArenaPass emits byte arenas, so the alloca element count is already
// the runtime byte expression. The function carries both the per-path arena
// descriptors and the `max_of_aligned_sums` reduction name so launch metadata
// can evaluate exactly the same expression.

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

static bool areDirectMutuallyExclusive(ArrayRef<Block *> blocks) {
  if (blocks.size() <= 1)
    return true;
  for (Block *predecessor : blocks.front()->getPredecessors()) {
    Operation *terminator = predecessor->getTerminator();
    if (!terminator || terminator->getNumSuccessors() < blocks.size())
      continue;
    if (llvm::all_of(blocks, [&](Block *candidate) {
          return llvm::is_contained(terminator->getSuccessors(), candidate);
        }))
      return true;
  }
  return false;
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
          /*addrSpace=*/3, /*dsoLocal=*/false, /*threadLocal=*/false,
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

      llvm::MapVector<Block *, SmallVector<LLVM::AllocaOp>> perPath;
      for (LLVM::AllocaOp alloca : allocas)
        perPath[alloca->getBlock()].push_back(alloca);
      SmallVector<Block *> pathBlocks;
      pathBlocks.reserve(perPath.size());
      for (auto &entry : perPath)
        pathBlocks.push_back(entry.first);
      if (!areDirectMutuallyExclusive(pathBlocks)) {
        allocas.front().emitOpError(
            "ROCM_DYNAMIC_LDS_MULTIPLE_ARENAS: runtime LDS arenas in "
            "different blocks must be direct mutually-exclusive successors "
            "of one branch; sequential, nested, looping, or escaping arena "
            "regions require a general lifetime-aware path expression");
        return signalPassFailure();
      }

      SmallVector<Attribute> paths;
      SmallVector<Attribute> flatDescriptors;
      for (auto [pathOrdinal, entry] : llvm::enumerate(perPath)) {
        auto &pathAllocas = entry.second;
        llvm::sort(pathAllocas, [](LLVM::AllocaOp lhs, LLVM::AllocaOp rhs) {
          return lhs->isBeforeInBlock(rhs);
        });
        Value cursor;
        SmallVector<Attribute> descriptors;
        for (auto [arenaOrdinal, alloca] : llvm::enumerate(pathAllocas)) {
          OpBuilder builder(alloca);
          Location loc = alloca.getLoc();
          Value count = alloca.getArraySize();
          Type indexType = count.getType();
          if (!cursor) {
            cursor = LLVM::ConstantOp::create(
                builder, loc, indexType,
                builder.getIntegerAttr(indexType, 0));
          } else if (cursor.getType() != indexType) {
            alloca.emitOpError(
                "all packed runtime LDS byte counts in one path must use one "
                "integer type");
            return signalPassFailure();
          }
          Value fifteen = LLVM::ConstantOp::create(
              builder, loc, indexType, builder.getIntegerAttr(indexType, 15));
          Value minusSixteen = LLVM::ConstantOp::create(
              builder, loc, indexType, builder.getIntegerAttr(indexType, -16));
          Value padded =
              LLVM::AddOp::create(builder, loc, cursor, fifteen).getResult();
          Value offset =
              LLVM::AndOp::create(builder, loc, padded, minusSixteen).getResult();
          Value base = LLVM::AddressOfOp::create(builder, loc, global);
          Value slice = LLVM::GEPOp::create(
              builder, loc, base.getType(), builder.getI8Type(), base,
              ValueRange{offset});
          alloca.replaceAllUsesWith(slice);
          cursor = LLVM::AddOp::create(builder, loc, offset, count).getResult();

          NamedAttrList descriptor;
          descriptor.append("ordinal",
                            builder.getI64IntegerAttr(arenaOrdinal));
          descriptor.append("alignment", builder.getI64IntegerAttr(16));
          descriptor.append("element_bytes", builder.getI64IntegerAttr(1));
          if (auto argument = dyn_cast<BlockArgument>(count)) {
            descriptor.append("count_source",
                              builder.getStringAttr("argument"));
            descriptor.append(
                "count_argument",
                builder.getI64IntegerAttr(argument.getArgNumber()));
          } else {
            descriptor.append("count_source",
                              builder.getStringAttr("ssa_expression"));
          }
          Attribute descriptorAttr =
              descriptor.getDictionary(&getContext());
          descriptors.push_back(descriptorAttr);
          flatDescriptors.push_back(descriptorAttr);
          alloca.erase();
        }
        OpBuilder builder(fn);
        NamedAttrList path;
        path.append("ordinal", builder.getI64IntegerAttr(pathOrdinal));
        path.append("arenas", builder.getArrayAttr(descriptors));
        paths.push_back(path.getDictionary(&getContext()));
      }
      OpBuilder builder(fn);
      fn->setAttr("tessera.rocm.dynamic_lds_launch_bytes",
                  builder.getUnitAttr());
      fn->setAttr("tessera.rocm.dynamic_lds_launch_reduction",
                  builder.getStringAttr("max_of_aligned_sums"));
      fn->setAttr("tessera.rocm.dynamic_lds_paths",
                  builder.getArrayAttr(paths));
      // Compatibility mirror for consumers that only need the flat arena
      // inventory. Offsets are path-relative when more than one path exists.
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
