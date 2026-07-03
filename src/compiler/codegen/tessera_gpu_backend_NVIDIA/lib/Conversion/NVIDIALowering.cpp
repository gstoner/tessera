#include "tessera/gpu/BackendRegistration.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "TesseraNVIDIADialect.h.inc"

using namespace mlir;

namespace tessera {
namespace {

static constexpr int kHopperSM = 90;
static constexpr int kBlackwellSM = 100;          // datacenter Blackwell (sm_100a)
static constexpr int kConsumerBlackwellSM = 120;  // consumer Blackwell (sm_120)

static StringRef archStringForSM(int sm) {
  if (sm >= 120)
    return "sm_120";
  if (sm >= 100)
    return "sm_100a";
  if (sm >= 90)
    return "sm_90a";
  if (sm >= 80)
    return "sm_80";
  return "sm_unknown";
}

static bool isTileOp(Operation *op, StringRef suffix) {
  return op->getName().getStringRef() == suffix;
}

static Operation *createContractOp(OpBuilder &builder, Location loc,
                                   StringRef name, ValueRange operands,
                                   TypeRange results,
                                   ArrayRef<NamedAttribute> attrs = {}) {
  OperationState state(loc, name);
  state.addOperands(operands);
  state.addTypes(results);
  state.addAttributes(attrs);
  return builder.create(state);
}

struct LowerTileToNVIDIAPass
    : PassWrapper<LowerTileToNVIDIAPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToNVIDIAPass)

  LowerTileToNVIDIAPass() = default;
  explicit LowerTileToNVIDIAPass(int sm) { smVersion = sm; }
  LowerTileToNVIDIAPass(const LowerTileToNVIDIAPass &other)
      : PassWrapper(other) {
    smVersion = other.smVersion;
  }

  Option<int> smVersion{*this, "sm",
                        llvm::cl::desc("Target NVIDIA SM version"),
                        llvm::cl::init(kHopperSM)};

  StringRef getArgument() const final { return "lower-tile-to-nvidia"; }

  StringRef getDescription() const final {
    return "Lower Tessera Tile IR to NVIDIA Hopper/Blackwell Target IR contracts";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    tessera::nvidia::TesseraNVIDIADialect>();
  }

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    ctx->loadDialect<tessera::nvidia::TesseraNVIDIADialect>();

    ModuleOp module = getOperation();
    OpBuilder moduleBuilder(ctx);
    module->setAttr("tessera.nvidia.arch",
                    moduleBuilder.getStringAttr(archStringForSM(smVersion)));

    SmallVector<Operation *> worklist;
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tile.mma" || name == "tile.async_copy" ||
          name == "tile.wait_async" || name.starts_with("tile.tmem."))
        worklist.push_back(op);
    });

    Value lastAsyncToken;
    for (Operation *op : worklist) {
      OpBuilder builder(op);
      Location loc = op->getLoc();
      StringRef name = op->getName().getStringRef();
      SmallVector<NamedAttribute> attrs;
      attrs.push_back(builder.getNamedAttr("arch",
                                           builder.getStringAttr(archStringForSM(smVersion))));

      if (name.starts_with("tile.tmem.") &&
          (smVersion < kBlackwellSM || smVersion >= kConsumerBlackwellSM)) {
        // TMEM is datacenter-Blackwell-only (sm_100a). Consumer Blackwell
        // sm_120 has no TMEM — it is NOT a superset of sm_100.
        op->emitError("NVIDIA TMEM lowering requires datacenter Blackwell "
                      "SM100 (consumer sm_120 has no TMEM)");
        signalPassFailure();
        return;
      }

      if (isTileOp(op, "tile.mma")) {
        if (op->getNumOperands() < 2 || op->getNumResults() > 1) {
          op->emitError("NVIDIA lowering requires tile.mma(lhs, rhs) -> optional result");
          signalPassFailure();
          return;
        }

        TypeRange resultTypes = op->getResultTypes();
        ValueRange operands = op->getOperands();
        if (smVersion >= kConsumerBlackwellSM) {
          // Consumer Blackwell (RTX 50-series, sm_120): warp-level `mma.sync`.
          // NOT a superset of datacenter sm_100 — no tcgen05/TMEM, no Hopper
          // wgmma (FP4 rides `mma.sync.aligned...block_scale`). Grounded in
          // gpu_target._CUDA_13_3_FEATURES[SM_120]. Mirrors the Python emitter
          // target_ir.py::_lower_nvidia_op; the tma_async_copy + mbarrier
          // companions come from the separate tile.async_copy / tile.wait_async
          // ops in the worklist (as with the wgmma path below).
          attrs.push_back(builder.getNamedAttr("shape",
                                               builder.getStringAttr("m16n8k16")));
          attrs.push_back(builder.getNamedAttr("dtype_ab",
                                               builder.getStringAttr("bf16")));
          attrs.push_back(builder.getNamedAttr("dtype_c",
                                               builder.getStringAttr("f32")));
          attrs.push_back(builder.getNamedAttr("block_scaled",
                                               builder.getBoolAttr(false)));
          Operation *target = createContractOp(builder, loc,
                                               "tessera_nvidia.mma_sync",
                                               operands, resultTypes, attrs);
          op->replaceAllUsesWith(target->getResults());
          op->erase();
          continue;
        }
        if (smVersion >= kBlackwellSM) {
          attrs.push_back(builder.getNamedAttr("shape",
                                               builder.getStringAttr("m128n128k32")));
          attrs.push_back(builder.getNamedAttr("accum",
                                               builder.getStringAttr("tmem_f32")));
          attrs.push_back(builder.getNamedAttr("cta_group",
                                               builder.getI64IntegerAttr(2)));
          attrs.push_back(builder.getNamedAttr("source",
                                               builder.getStringAttr("tessera.matmul")));
          auto alloc = createContractOp(builder, loc, "tessera_nvidia.tmem_alloc",
                                        ValueRange{}, TypeRange{},
                                        {builder.getNamedAttr("columns",
                                                              builder.getI64IntegerAttr(128)),
                                         builder.getNamedAttr("arch",
                                                              builder.getStringAttr(archStringForSM(smVersion)))});
          (void)alloc;
          Operation *target = createContractOp(builder, loc,
                                               "tessera_nvidia.tcgen05_mma",
                                               operands, resultTypes, attrs);
          op->replaceAllUsesWith(target->getResults());
          op->erase();
          continue;
        }

        if (smVersion >= kHopperSM) {
          attrs.push_back(builder.getNamedAttr("shape",
                                               builder.getStringAttr("m64n64k16")));
          attrs.push_back(builder.getNamedAttr("dtype_ab",
                                               builder.getStringAttr("bf16")));
          attrs.push_back(builder.getNamedAttr("dtype_c",
                                               builder.getStringAttr("f32")));
          attrs.push_back(builder.getNamedAttr("warpgroup",
                                               builder.getI64IntegerAttr(4)));
          Operation *target = createContractOp(builder, loc,
                                               "tessera_nvidia.wgmma",
                                               operands, resultTypes, attrs);
          op->replaceAllUsesWith(target->getResults());
          op->erase();
          continue;
        }

        attrs.push_back(builder.getNamedAttr("shape",
                                             builder.getStringAttr("m16n16k16")));
        Operation *target = createContractOp(builder, loc, "tessera_nvidia.wmma",
                                             operands, resultTypes, attrs);
        op->replaceAllUsesWith(target->getResults());
        op->erase();
        continue;
      }

      if (isTileOp(op, "tile.async_copy")) {
        if (smVersion < kHopperSM) {
          op->emitError("NVIDIA TMA lowering requires Hopper SM90+");
          signalPassFailure();
          return;
        }
        attrs.push_back(builder.getNamedAttr("src_space",
                                             builder.getStringAttr("global")));
        attrs.push_back(builder.getNamedAttr("dst_space",
                                             builder.getStringAttr("shared")));
        attrs.push_back(builder.getNamedAttr("bytes",
                                             builder.getI64IntegerAttr(16)));
        Operation *target = createContractOp(builder, loc,
                                             "tessera_nvidia.tma_async_copy",
                                             op->getOperands(), op->getResultTypes(),
                                             attrs);
        if (target->getNumResults() > 0)
          lastAsyncToken = target->getResult(0);
        op->replaceAllUsesWith(target->getResults());
        op->erase();
        continue;
      }

      if (isTileOp(op, "tile.wait_async")) {
        if (smVersion < kHopperSM) {
          op->emitError("NVIDIA mbarrier lowering requires Hopper SM90+");
          signalPassFailure();
          return;
        }
        if (lastAsyncToken)
          createContractOp(builder, loc, "tessera_nvidia.mbarrier",
                           ValueRange{lastAsyncToken}, TypeRange{}, attrs);
        else
          createContractOp(builder, loc, "tessera_nvidia.mbarrier",
                           ValueRange{}, TypeRange{}, attrs);
        op->erase();
        continue;
      }

      if (name.starts_with("tile.tmem.")) {
        StringRef contractName = "tessera_nvidia.tmem_store";
        if (name == "tile.tmem.alloc")
          contractName = "tessera_nvidia.tmem_alloc";
        else if (name == "tile.tmem.load")
          contractName = "tessera_nvidia.tmem_load";
        createContractOp(builder, loc, contractName, op->getOperands(),
                         op->getResultTypes(), attrs);
        op->erase();
      }
    }
  }
};

static LLVM::LLVMFuncOp declareVoidMarker(ModuleOp module, StringRef name) {
  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  auto fnType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(module.getContext()), {}, false);
  return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
}

static StringRef markerForTargetOp(StringRef opName) {
  if (opName == "tessera_nvidia.wgmma")
    return "llvm.nvvm.wgmma.contract";
  if (opName == "tessera_nvidia.tma_async_copy")
    return "llvm.nvvm.cp.async.bulk.tensor.contract";
  if (opName == "tessera_nvidia.mbarrier")
    return "llvm.nvvm.mbarrier.contract";
  if (opName == "tessera_nvidia.wmma")
    return "llvm.nvvm.mma.sync.contract";
  if (opName == "tessera_nvidia.mma_sync")  // consumer Blackwell sm_120 warp-level MMA
    return "llvm.nvvm.mma.sync.contract";
  if (opName == "tessera_nvidia.tcgen05_mma")
    return "llvm.nvvm.tcgen05.mma.contract";
  if (opName == "tessera_nvidia.tmem_alloc")
    return "llvm.nvvm.tmem.alloc.contract";
  if (opName == "tessera_nvidia.tmem_load")
    return "llvm.nvvm.tmem.load.contract";
  if (opName == "tessera_nvidia.tmem_store")
    return "llvm.nvvm.tmem.store.contract";
  if (opName == "tessera_nvidia.cuda_kernel")
    return "llvm.nvvm.cuda.kernel.contract";
  return "llvm.nvvm.tessera.nvidia.diagnostic.contract";
}

struct LowerNVIDIAToNVVMPass
    : PassWrapper<LowerNVIDIAToNVVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerNVIDIAToNVVMPass)

  StringRef getArgument() const final { return "lower-tessera-nvidia-to-nvvm"; }

  StringRef getDescription() const final {
    return "Lower Tessera NVIDIA Target IR contracts to LLVM/NVVM artifact markers";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect, NVVM::NVVMDialect>();
  }

  void runOnOperation() final {
    getContext().loadDialect<LLVM::LLVMDialect, NVVM::NVVMDialect>();
    ModuleOp module = getOperation();
    SmallVector<Operation *> targetOps;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef().starts_with("tessera_nvidia."))
        targetOps.push_back(op);
    });

    for (Operation *op : targetOps) {
      OpBuilder builder(op);
      auto marker = declareVoidMarker(module, markerForTargetOp(op->getName().getStringRef()));
      builder.create<LLVM::CallOp>(op->getLoc(), TypeRange{},
                                   SymbolRefAttr::get(marker), ValueRange{});
      if (!op->use_empty())
        op->dropAllUses();
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLowerTileToNVIDIAPass(int sm) {
  return std::make_unique<LowerTileToNVIDIAPass>(sm);
}

std::unique_ptr<Pass> createLowerNVIDIAToNVVMPass() {
  return std::make_unique<LowerNVIDIAToNVVMPass>();
}

void buildTesseraNVIDIABackendPipeline(OpPassManager &pm) {
  pm.addPass(createLowerTileToNVIDIAPass(kHopperSM));
  pm.addPass(createLowerNVIDIAToNVVMPass());
}

void buildTesseraHopperBackendPipeline(OpPassManager &pm) {
  pm.addPass(createLowerTileToNVIDIAPass(kHopperSM));
  pm.addPass(createLowerNVIDIAToNVVMPass());
}

void buildTesseraBlackwellBackendPipeline(OpPassManager &pm) {
  pm.addPass(createLowerTileToNVIDIAPass(kBlackwellSM));
  pm.addPass(createLowerNVIDIAToNVVMPass());
}

void registerTesseraNVIDIABackendPasses() {
  registerPass([]() { return createLowerTileToNVIDIAPass(kHopperSM); });
  registerPass([]() { return createLowerNVIDIAToNVVMPass(); });

  PassPipelineRegistration<> nvidiaPipeline(
      "tessera-lower-to-nvidia",
      "Lower Tessera Tile IR to NVIDIA NVVM/PTX artifact contracts",
      [](OpPassManager &pm) { buildTesseraNVIDIABackendPipeline(pm); });
  PassPipelineRegistration<> hopperPipeline(
      "tessera-lower-to-hopper",
      "Lower Tessera Tile IR to Hopper WGMMA/TMA artifact contracts",
      [](OpPassManager &pm) { buildTesseraHopperBackendPipeline(pm); });
  PassPipelineRegistration<> blackwellPipeline(
      "tessera-lower-to-blackwell",
      "Lower Tessera Tile IR to Blackwell TCGEN05/TMEM artifact contracts",
      [](OpPassManager &pm) { buildTesseraBlackwellBackendPipeline(pm); });
}

void registerTesseraNVIDIABackendDialects(DialectRegistry &registry) {
  registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                  NVVM::NVVMDialect, tessera::nvidia::TesseraNVIDIADialect>();
}

} // namespace tessera
