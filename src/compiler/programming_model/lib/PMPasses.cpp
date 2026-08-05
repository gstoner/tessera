//===- PMPasses.cpp — Programming Model v1.1 passes and pipelines ---------===//
//
// Library-owned pass bodies. Previously these lived in
// `tools/tessera-opt/PassPipelinesPM11.cpp`, a driver source, so the passes
// could not be constructed or lit-tested independently of that driver (W0.6).
//
// See PMPasses.h for the maturity contract: the verifier is general, while the
// two lowering passes are real only for the bounded E2E-REAL-2 static-matmul
// contract and fail closed outside it.
//
//===----------------------------------------------------------------------===//

#include "tessera/ProgrammingModel/PMPasses.h"
#include "tessera/ProgrammingModel/ScheduleDialect.h"
#include "Tessera/Dialect/Tile/TileDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

#include <optional>

using namespace mlir;

namespace tessera {

// ---------------------------------------------------------------------------
// Dialect registration
// ---------------------------------------------------------------------------

void registerPMPipelinesV11(DialectRegistry &registry) {
  schedule::registerScheduleDialect(registry);
}

// ---------------------------------------------------------------------------
// PMV11 verifier pass — walks the module and verifies every op whose dialect
// name starts with schedule/cache/tile. This is a real verifier.
// ---------------------------------------------------------------------------

namespace {
struct PMV11VerifierPass
    : public PassWrapper<PMV11VerifierPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PMV11VerifierPass)

  StringRef getArgument() const override { return "tessera-pm-verify"; }
  StringRef getDescription() const override {
    return "Verify all Schedule / Cache / TileMemory v1.1 ops";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool anyFailed = false;
    mod.walk([&](Operation *op) -> WalkResult {
      StringRef name = op->getName().getStringRef();
      if (!name.starts_with("schedule.") && !name.starts_with("cache.") &&
          !name.starts_with("tile."))
        return WalkResult::advance();

      if (failed(verifyOp(op)))
        anyFailed = true;
      return WalkResult::advance();
    });
    if (anyFailed) signalPassFailure();
  }

private:
  // Inline lightweight verification (mirrors ScheduleOps.cpp dispatcher).
  LogicalResult verifyOp(Operation *op) {
    StringRef name = op->getName().getStringRef();

    // schedule.mesh.define
    if (name == "schedule.mesh.define") {
      auto dims  = op->getAttrOfType<ArrayAttr>("dims");
      auto names = op->getAttrOfType<ArrayAttr>("axis_names");
      if (!dims || dims.empty())
        return op->emitOpError("requires non-empty 'dims'");
      if (!names || names.size() != dims.size())
        return op->emitOpError("'axis_names' must have same length as 'dims'");
      return success();
    }

    // schedule.pipeline.region
    if (name == "schedule.pipeline.region") {
      auto mb = op->getAttrOfType<IntegerAttr>("micro_batches");
      if (!mb || mb.getInt() < 1)
        return op->emitOpError("'micro_batches' must be >= 1");
      return success();
    }

    // tile.async_copy
    if (name == "tile.async_copy") {
      auto stage = op->getAttrOfType<IntegerAttr>("stage");
      if (!stage || stage.getInt() < 0)
        return op->emitOpError("'stage' must be >= 0");
      return success();
    }

    // tile.mbarrier.alloc
    if (name == "tile.mbarrier.alloc") {
      auto count = op->getAttrOfType<IntegerAttr>("count");
      if (!count || count.getInt() <= 0)
        return op->emitOpError("'count' must be > 0");
      auto scope = op->getAttrOfType<StringAttr>("scope");
      if (!scope || !isValidScope(scope.getValue()))
        return op->emitOpError("'scope' must be one of thread, warp, block, cluster, device, mesh");
      if (!supportsMBarrier(op))
        return op->emitOpError("mbarrier requires target/arch containing sm90, sm100, sm120, hopper, or blackwell");
      return success();
    }

    if (name == "tile.mbarrier.arrive_expect_tx") {
      auto bytes = op->getAttrOfType<IntegerAttr>("bytes");
      if (!bytes || bytes.getInt() <= 0)
        return op->emitOpError("'bytes' must be > 0");
      auto scope = op->getAttrOfType<StringAttr>("scope");
      if (!scope || !isValidScope(scope.getValue()))
        return op->emitOpError("'scope' must be one of thread, warp, block, cluster, device, mesh");
      auto semantics = op->getAttrOfType<StringAttr>("semantics");
      if (!semantics || (semantics.getValue() != "release" &&
                         semantics.getValue() != "acq_rel" &&
                         semantics.getValue() != "seq_cst"))
        return op->emitOpError("'semantics' must be release, acq_rel, or seq_cst");
      return success();
    }

    if (name == "tile.mbarrier.try_wait") {
      if (op->getNumOperands() != 2)
        return op->emitOpError("expected exactly 2 operands (barrier, token)");
      return success();
    }

    if (name == "tile.atomic") {
      auto order = op->getAttrOfType<StringAttr>("order");
      if (!order || !isValidOrder(order.getValue()))
        return op->emitOpError("'order' must be relaxed, acquire, release, acq_rel, or seq_cst");
      auto scope = op->getAttrOfType<StringAttr>("scope");
      if (!scope || !isValidScope(scope.getValue()))
        return op->emitOpError("'scope' must be one of thread, warp, block, cluster, device, mesh");
      return success();
    }

    if (name == "tile.barrier") {
      auto divergent = op->getAttrOfType<BoolAttr>("divergent");
      if (divergent && divergent.getValue())
        return op->emitOpError("barrier cannot be marked divergent");
      return success();
    }

    // schedule.knob
    if (name == "schedule.knob") {
      auto choices = op->getAttrOfType<ArrayAttr>("choices");
      if (!choices || choices.empty())
        return op->emitOpError("'choices' must be non-empty");
      auto logits = op->getAttrOfType<ArrayAttr>("logits");
      if (logits && logits.size() != choices.size())
        return op->emitOpError("'logits' must have same size as 'choices'");
      return success();
    }

    return success(); // other ops: no custom constraint
  }

  bool isValidScope(StringRef scope) const {
    return scope == "thread" || scope == "warp" || scope == "block" ||
           scope == "cluster" || scope == "device" || scope == "mesh";
  }

  bool isValidOrder(StringRef order) const {
    return order == "relaxed" || order == "acquire" || order == "release" ||
           order == "acq_rel" || order == "seq_cst";
  }

  bool supportsMBarrier(Operation *op) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module) return false;
    auto target = module->getAttrOfType<StringAttr>("target");
    auto arch = module->getAttrOfType<StringAttr>("arch");
    StringRef value = target ? target.getValue() : (arch ? arch.getValue() : "");
    return value.contains("sm90") || value.contains("sm_90") ||
           value.contains("sm100") || value.contains("sm_100") ||
           value.contains("sm120") || value.contains("sm_120") ||
           value.contains("hopper") || value.contains("blackwell");
  }
};
} // anonymous namespace

// ---------------------------------------------------------------------------
// Graph -> Schedule — bounded mixed-level static matmul contract.
// ---------------------------------------------------------------------------

namespace {
struct MatmulSchedule {
  StringRef target;
  StringRef arch;
  StringRef storage;
  StringRef accum;
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t tileM = 16;
  int64_t tileN = 16;
  int64_t tileK = 16;
  int64_t macroTileM = 16;
  int64_t macroTileN = 16;
  int64_t warps = 1;
  int64_t pipelineDepth = 1;
  StringRef rasterOrder = "row_major";
  int64_t rasterGroup = 1;
};

static StringRef moduleString(ModuleOp module, StringRef primary,
                              StringRef fallback) {
  if (auto value = module->getAttrOfType<StringAttr>(primary))
    return value.getValue();
  if (auto value = module->getAttrOfType<StringAttr>(fallback))
    return value.getValue();
  return {};
}

static FailureOr<MatmulSchedule> getMatmulSchedule(Operation *op) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module || op->getNumOperands() != 2 || op->getNumResults() != 1)
    return failure();
  auto lhs = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto rhs = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto out = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!lhs || !rhs || !out || lhs.getRank() != 2 || rhs.getRank() != 2 ||
      out.getRank() != 2 || !lhs.hasStaticShape() || !rhs.hasStaticShape() ||
      !out.hasStaticShape())
    return failure();
  if (auto transpose = op->getAttrOfType<BoolAttr>("transposeA");
      transpose && transpose.getValue())
    return failure();
  if (auto transpose = op->getAttrOfType<BoolAttr>("transposeB");
      transpose && transpose.getValue())
    return failure();

  MatmulSchedule schedule;
  schedule.target = moduleString(module, "tessera.target", "target");
  schedule.arch = moduleString(module, "tessera.arch", "arch");
  schedule.m = lhs.getDimSize(0);
  schedule.k = lhs.getDimSize(1);
  schedule.n = rhs.getDimSize(1);
  if (schedule.m <= 0 || schedule.n <= 0 || schedule.k <= 0 ||
      rhs.getDimSize(0) != schedule.k || out.getDimSize(0) != schedule.m ||
      out.getDimSize(1) != schedule.n)
    return failure();

  Type lhsElement = lhs.getElementType();
  Type rhsElement = rhs.getElementType();
  Type outElement = out.getElementType();
  bool x86 = schedule.target == "x86" || schedule.arch.contains("avx512") ||
             schedule.arch.contains("zen5");
  // This bounded physical schedule is gfx1151-owned.  The shared macro-tile
  // vocabulary is portable, but gfx1200/gfx1250 must supply their own exact-
  // device schedule and instruction-family profile rather than inheriting it.
  bool rocm = schedule.arch.contains("gfx1151");
  if (x86 && lhsElement.isF32() && rhsElement.isF32() && outElement.isF32()) {
    schedule.storage = "f32";
    schedule.accum = "f32";
    if (schedule.arch.empty())
      schedule.arch = "x86-avx512";
    return schedule;
  }
  if (rocm && lhsElement.isF16() && rhsElement.isF16() && outElement.isF32()) {
    schedule.storage = "f16";
    schedule.accum = "f32";
    // gfx1151's committed production GEMM is a 2x4 register-blocked WMMA
    // macro-tile.  Schedule IR carries logical element extents, not the
    // backend's mt/nt spelling, so preserve that decision as 32x64x16.
    schedule.macroTileM = 32;
    schedule.macroTileN = 64;
    if (schedule.arch.empty())
      schedule.arch = "gfx1151";
    return schedule;
  }
  return failure();
}

static std::string scheduleDigest(const MatmulSchedule &schedule) {
  std::string contract =
      (Twine("target=") + schedule.target + ";arch=" + schedule.arch +
       ";M=" + Twine(schedule.m) + ";N=" + Twine(schedule.n) +
       ";K=" + Twine(schedule.k) + ";storage=" + schedule.storage +
       ";accum=" + schedule.accum + ";tile=" + Twine(schedule.tileM) + "x" +
       Twine(schedule.tileN) + "x" + Twine(schedule.tileK) +
       ";macro_tile=" + Twine(schedule.macroTileM) + "x" +
       Twine(schedule.macroTileN) + ";warps=" +
       Twine(schedule.warps) + ";pipeline_depth=" +
       Twine(schedule.pipelineDepth) + ";raster=row_major;group=" +
       Twine(schedule.rasterGroup))
          .str();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(contract)),
                     /*LowerCase=*/true);
}

struct SemanticKernelSchedule {
  StringRef family;
  StringRef kind;
  StringRef target;
  StringRef arch;
  StringRef storage;
  StringRef accum = "f32";
  SmallVector<int64_t> inputShape;
  SmallVector<int64_t> outputShape;
  int64_t axis = -1;
  bool keepdims = false;
  int64_t rows = 1;
  int64_t columns = 1;
  int64_t outer = 1;
  int64_t axisExtent = 1;
  int64_t inner = 1;
  int64_t workgroupSize = 1;
};

static StringRef storageName(Type type) {
  if (type.isF16()) return "f16";
  if (type.isBF16()) return "bf16";
  if (type.isF32()) return "f32";
  return {};
}

static FailureOr<SemanticKernelSchedule> getSemanticKernelSchedule(Operation *op) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module || op->getNumOperands() != 1 || op->getNumResults() != 1)
    return failure();
  auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!input || !output || input.getRank() < 1 || !input.hasStaticShape() ||
      !output.hasStaticShape())
    return failure();

  SemanticKernelSchedule schedule;
  schedule.target = moduleString(module, "tessera.target", "target");
  schedule.arch = moduleString(module, "tessera.arch", "arch");
  schedule.storage = storageName(input.getElementType());
  schedule.inputShape.assign(input.getShape().begin(), input.getShape().end());
  schedule.outputShape.assign(output.getShape().begin(), output.getShape().end());
  bool x86 = schedule.target == "x86" || schedule.arch.contains("avx512") ||
             schedule.arch.contains("zen5");
  bool rocm = schedule.arch.contains("gfx1151");
  if (!x86 && !rocm)
    return failure();

  StringRef opName = op->getName().getStringRef();
  if (opName == "tessera.softmax") {
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    if ((axisAttr && axisAttr.getInt() != -1) || input != output ||
        (x86 && schedule.storage != "f32") ||
        (rocm && schedule.storage != "f16" && schedule.storage != "f32"))
      return failure();
    schedule.family = "softmax";
    schedule.rows = 1;
    for (int64_t dim : input.getShape().drop_back()) schedule.rows *= dim;
    schedule.columns = input.getShape().back();
    schedule.workgroupSize = rocm ? 256 : 1;
    return schedule;
  }

  if (opName != "tessera.reduce")
    return failure();
  auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
  int64_t axis = axisAttr ? axisAttr.getInt() : -1;
  if (axis < 0) axis += input.getRank();
  if (axis < 0 || axis >= input.getRank() || !output.getElementType().isF32())
    return failure();
  auto kindAttr = op->getAttrOfType<StringAttr>("kind");
  if (!kindAttr || (kindAttr.getValue() != "sum" &&
                    kindAttr.getValue() != "mean" &&
                    kindAttr.getValue() != "max"))
    return failure();
  bool keepdims = false;
  SmallVector<int64_t> expected(input.getShape().begin(), input.getShape().end());
  if (keepdims) expected[axis] = 1;
  else expected.erase(expected.begin() + axis);
  if (ArrayRef<int64_t>(expected) != output.getShape() ||
      schedule.storage != "f32" ||
      (x86 && axis != input.getRank() - 1))
    return failure();
  schedule.family = "reduce";
  schedule.kind = kindAttr.getValue();
  schedule.axis = axis;
  schedule.keepdims = keepdims;
  for (int64_t dim : input.getShape().take_front(axis)) schedule.outer *= dim;
  schedule.axisExtent = input.getDimSize(axis);
  for (int64_t dim : input.getShape().drop_front(axis + 1)) schedule.inner *= dim;
  schedule.workgroupSize = rocm ? 256 : 1;
  return schedule;
}

static std::string semanticKernelDigest(const SemanticKernelSchedule &schedule) {
  std::string inputShape;
  std::string outputShape;
  for (int64_t dim : schedule.inputShape)
    inputShape += (inputShape.empty() ? "" : "x") + Twine(dim).str();
  for (int64_t dim : schedule.outputShape)
    outputShape += (outputShape.empty() ? "" : "x") + Twine(dim).str();
  std::string contract =
      (Twine("family=") + schedule.family + ";kind=" + schedule.kind +
       ";target=" + schedule.target + ";arch=" + schedule.arch +
       ";input=" + inputShape + ";output=" + outputShape +
       ";storage=" + schedule.storage + ";accum=" + schedule.accum +
       ";axis=" + Twine(schedule.axis) + ";keepdims=" +
       Twine(schedule.keepdims ? 1 : 0) + ";rows=" + Twine(schedule.rows) +
       ";columns=" + Twine(schedule.columns) + ";outer=" +
       Twine(schedule.outer) + ";axis_extent=" + Twine(schedule.axisExtent) +
       ";inner=" + Twine(schedule.inner) + ";workgroup=" +
       Twine(schedule.workgroupSize) + ";exp=accurate;ftz=0;schedule=serial;nan=propagate")
          .str();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(contract)),
                     /*LowerCase=*/true);
}

struct GraphToSchedulePass
    : public PassWrapper<GraphToSchedulePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GraphToSchedulePass)

  StringRef getArgument() const override { return "tessera-graph-to-schedule"; }
  StringRef getDescription() const override {
    return "Create a content-addressed mixed-level schedule.matmul SSA edge "
           "for bounded static x86-f32 and ROCm-f16/f32 Graph matmul";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<schedule::ScheduleDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod.getContext());
    SmallVector<Operation *> matmuls;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.matmul")
        matmuls.push_back(op);
    });
    for (Operation *op : matmuls) {
      FailureOr<MatmulSchedule> selected = getMatmulSchedule(op);
      if (failed(selected)) {
        op->emitError("E2E-REAL-2 Graph->Schedule requires static rank-2 "
                      "x86 f32->f32 or ROCm f16->f32 matmul with no transpose");
        return signalPassFailure();
      }
      std::string digest = scheduleDigest(*selected);
      op->setAttr("schedule.artifact_hash", builder.getStringAttr(digest));

      builder.setInsertionPointAfter(op);
      OperationState state(op->getLoc(), "schedule.matmul");
      state.addOperands(op->getResult(0));
      state.addTypes(op->getResult(0).getType());
      state.addAttribute("artifact_hash", builder.getStringAttr(digest));
      state.addAttribute("arch", builder.getStringAttr(selected->arch));
      state.addAttribute("tile_m", builder.getI64IntegerAttr(selected->tileM));
      state.addAttribute("tile_n", builder.getI64IntegerAttr(selected->tileN));
      state.addAttribute("tile_k", builder.getI64IntegerAttr(selected->tileK));
      state.addAttribute("macro_tile_m",
                         builder.getI64IntegerAttr(selected->macroTileM));
      state.addAttribute("macro_tile_n",
                         builder.getI64IntegerAttr(selected->macroTileN));
      state.addAttribute("warps", builder.getI64IntegerAttr(selected->warps));
      state.addAttribute("pipeline_depth",
                         builder.getI64IntegerAttr(selected->pipelineDepth));
      state.addAttribute("storage", builder.getStringAttr(selected->storage));
      state.addAttribute("accum", builder.getStringAttr(selected->accum));
      state.addAttribute("a_layout", builder.getStringAttr("row_major"));
      state.addAttribute("b_layout", builder.getStringAttr("col_major"));
      state.addAttribute("raster_order",
                         builder.getStringAttr(selected->rasterOrder));
      state.addAttribute("raster_group",
                         builder.getI64IntegerAttr(selected->rasterGroup));
      Operation *scheduled = builder.create(state);
      for (OpOperand &use : llvm::make_early_inc_range(op->getResult(0).getUses()))
        if (use.getOwner() != scheduled)
          use.set(scheduled->getResult(0));

      builder.setInsertionPointAfter(scheduled);
      OperationState artifactState(op->getLoc(), "schedule.artifact");
      artifactState.addAttribute("hash", builder.getStringAttr(digest));
      artifactState.addAttribute("arch", builder.getStringAttr(selected->arch));
      artifactState.addAttribute(
          "shape_key",
          builder.getStringAttr((Twine("M=") + Twine(selected->m) + ";N=" +
                                 Twine(selected->n) + ";K=" +
                                 Twine(selected->k) + ";dtype=" +
                                 selected->storage)
                                    .str()));
      artifactState.addAttribute(
          "tile", builder.getDictionaryAttr({
                      builder.getNamedAttr(
                          "m", builder.getI64IntegerAttr(selected->tileM)),
                      builder.getNamedAttr(
                          "n", builder.getI64IntegerAttr(selected->tileN)),
                      builder.getNamedAttr(
                          "k", builder.getI64IntegerAttr(selected->tileK)),
                      builder.getNamedAttr(
                          "macro_m",
                          builder.getI64IntegerAttr(selected->macroTileM)),
                      builder.getNamedAttr(
                          "macro_n",
                          builder.getI64IntegerAttr(selected->macroTileN)),
                      builder.getNamedAttr(
                          "warps", builder.getI64IntegerAttr(selected->warps)),
                      builder.getNamedAttr(
                          "pipeline_depth",
                          builder.getI64IntegerAttr(selected->pipelineDepth)),
                  }));
      artifactState.addAttribute(
          "numeric_policy",
          builder.getStringAttr((Twine(selected->storage) + "->" +
                                 selected->accum)
                                    .str()));
      builder.create(artifactState);
    }

    SmallVector<Operation *> semanticKernels;
    mod.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tessera.softmax" || name == "tessera.reduce")
        semanticKernels.push_back(op);
    });
    for (Operation *op : semanticKernels) {
      FailureOr<SemanticKernelSchedule> selected = getSemanticKernelSchedule(op);
      if (failed(selected)) {
        op->emitError("E2E-REAL-5 Graph->Schedule requires a supported static "
                      "x86 or gfx1151 softmax/reduction contract");
        return signalPassFailure();
      }
      std::string digest = semanticKernelDigest(*selected);
      op->setAttr("schedule.artifact_hash", builder.getStringAttr(digest));
      builder.setInsertionPointAfter(op);
      OperationState state(op->getLoc(),
                           selected->family == "softmax" ? "schedule.softmax"
                                                         : "schedule.reduce");
      state.addOperands(op->getResult(0));
      state.addTypes(op->getResult(0).getType());
      state.addAttribute("artifact_hash", builder.getStringAttr(digest));
      state.addAttribute("arch", builder.getStringAttr(selected->arch));
      state.addAttribute("storage", builder.getStringAttr(selected->storage));
      state.addAttribute("accum", builder.getStringAttr(selected->accum));
      state.addAttribute("axis", builder.getI64IntegerAttr(selected->axis));
      state.addAttribute("workgroup_size",
                         builder.getI64IntegerAttr(selected->workgroupSize));
      if (selected->family == "softmax") {
        state.addAttribute("exp_mode", builder.getStringAttr("accurate"));
        state.addAttribute("ftz", builder.getBoolAttr(false));
      } else {
        state.addAttribute("kind", builder.getStringAttr(selected->kind));
        state.addAttribute("keepdims", builder.getBoolAttr(selected->keepdims));
        state.addAttribute("schedule", builder.getStringAttr("serial"));
        state.addAttribute("nan_mode", builder.getStringAttr("propagate"));
        state.addAttribute("inner_is_one", builder.getBoolAttr(selected->inner == 1));
      }
      Operation *scheduled = builder.create(state);
      for (OpOperand &use : llvm::make_early_inc_range(op->getResult(0).getUses()))
        if (use.getOwner() != scheduled)
          use.set(scheduled->getResult(0));

      builder.setInsertionPointAfter(scheduled);
      OperationState artifactState(op->getLoc(), "schedule.artifact");
      artifactState.addAttribute("hash", builder.getStringAttr(digest));
      artifactState.addAttribute("arch", builder.getStringAttr(selected->arch));
      artifactState.addAttribute(
          "shape_key",
          builder.getStringAttr((Twine("family=") + selected->family +
                                 ";storage=" + selected->storage +
                                 ";axis=" + Twine(selected->axis))
                                    .str()));
      artifactState.addAttribute(
          "tile", builder.getDictionaryAttr({
                      builder.getNamedAttr("workgroup_size",
                                           builder.getI64IntegerAttr(selected->workgroupSize)),
                  }));
      artifactState.addAttribute(
          "numeric_policy",
          builder.getStringAttr((Twine(selected->storage) + "->" + selected->accum)
                                    .str()));
      builder.create(artifactState);
    }
  }
};
} // anonymous namespace

// ---------------------------------------------------------------------------
// Schedule -> Tile — consume the mixed-level scheduled Graph matmul atomically.
// ---------------------------------------------------------------------------

namespace {
struct ScheduleToTilePass
    : public PassWrapper<ScheduleToTilePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleToTilePass)

  StringRef getArgument() const override { return "tessera-schedule-to-tile"; }
  StringRef getDescription() const override {
    return "Consume bounded schedule.matmul plus its Graph producer and emit "
           "one six-operand launch-level tile.matmul_kernel";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    LLVM::LLVMDialect, memref::MemRefDialect,
                    schedule::ScheduleDialect>();
    tile::registerTileDialect(registry);
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod.getContext());
    SmallVector<schedule::MatmulOp> scheduledMatmuls;
    mod.walk([&](schedule::MatmulOp op) { scheduledMatmuls.push_back(op); });
    for (schedule::MatmulOp scheduled : scheduledMatmuls) {
      Operation *graph = scheduled.getSubject().getDefiningOp();
      if (!graph || graph->getName().getStringRef() != "tessera.matmul" ||
          graph->getNumOperands() != 2 || graph->getNumResults() != 1) {
        scheduled.emitError(
            "E2E-REAL-2 requires subject to be the retained Graph matmul result");
        return signalPassFailure();
      }
      auto selected = getMatmulSchedule(graph);
      if (failed(selected) || scheduleDigest(*selected) != scheduled.getArtifactHash()) {
        scheduled.emitError(
            "scheduled decision does not match the retained Graph matmul contract");
        return signalPassFailure();
      }
      if (scheduled.getTileMAttr().getInt() != selected->tileM ||
          scheduled.getTileNAttr().getInt() != selected->tileN ||
          scheduled.getTileKAttr().getInt() != selected->tileK ||
          scheduled.getMacroTileMAttr().getInt() != selected->macroTileM ||
          scheduled.getMacroTileNAttr().getInt() != selected->macroTileN ||
          scheduled.getWarpsAttr().getInt() != selected->warps ||
          scheduled.getPipelineDepthAttr().getInt() != selected->pipelineDepth ||
          scheduled.getStorage() != selected->storage ||
          scheduled.getAccum() != selected->accum ||
          scheduled.getArch() != selected->arch ||
          scheduled.getALayout() != "row_major" ||
          scheduled.getBLayout() != "col_major" ||
          scheduled.getRasterOrder() != selected->rasterOrder ||
          scheduled.getRasterGroupAttr().getInt() != selected->rasterGroup) {
        scheduled.emitError("scheduled tile or numeric policy was altered after hashing");
        return signalPassFailure();
      }
      auto graphDigest = graph->getAttrOfType<StringAttr>("schedule.artifact_hash");
      SmallVector<schedule::ArtifactOp> matchingArtifacts;
      mod.walk([&](schedule::ArtifactOp artifact) {
        if (artifact.getHash() == scheduled.getArtifactHash())
          matchingArtifacts.push_back(artifact);
      });
      if (!graphDigest || graphDigest.getValue() != scheduled.getArtifactHash() ||
          matchingArtifacts.size() != 1) {
        scheduled.emitError(
            "requires exactly one matching Graph hash and schedule.artifact");
        return signalPassFailure();
      }

      auto lhsType = cast<RankedTensorType>(graph->getOperand(0).getType());
      auto rhsType = cast<RankedTensorType>(graph->getOperand(1).getType());
      auto outType = cast<RankedTensorType>(graph->getResult(0).getType());
      Location loc = scheduled.getLoc();
      builder.setInsertionPoint(scheduled);
      auto pointerType = LLVM::LLVMPointerType::get(&getContext());
      auto toPointer = [&](Value tensor, RankedTensorType type) {
        auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
        Value buffer = builder.create<bufferization::ToBufferOp>(loc, memrefType, tensor);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer =
            builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), index);
        return builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer)
            .getResult();
      };
      Value a = toPointer(graph->getOperand(0), lhsType);
      Value b = toPointer(graph->getOperand(1), rhsType);
      auto outputMemref =
          MemRefType::get(outType.getShape(), outType.getElementType());
      Value output = builder.create<memref::AllocOp>(loc, outputMemref);
      Value outputIndex =
          builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, output);
      Value outputInteger = builder.create<arith::IndexCastOp>(
          loc, builder.getI64Type(), outputIndex);
      Value d = builder.create<LLVM::IntToPtrOp>(loc, pointerType, outputInteger);
      Value m = builder.create<arith::ConstantIntOp>(loc, selected->m, 64);
      Value n = builder.create<arith::ConstantIntOp>(loc, selected->n, 64);
      Value k = builder.create<arith::ConstantIntOp>(loc, selected->k, 64);

      StringRef family = selected->target == "rocm" ? "wmma" : "auto";
      auto mma = tile::TileMmaDescAttr::get(
          &getContext(), family, 16, 16, 16, selected->storage,
          selected->storage, selected->accum, "row_major", "col_major", 1);
      auto epilogue = tile::TileEpilogueAttr::get(
          &getContext(), /*bias=*/false, "none", selected->accum);

      OperationState kernelState(loc, "tile.matmul_kernel");
      kernelState.addOperands({a, b, d, m, n, k});
      kernelState.addAttribute("mma", mma);
      kernelState.addAttribute("epilogue", epilogue);
      kernelState.addAttribute("warps",
                               builder.getI64IntegerAttr(selected->warps));
      kernelState.addAttribute("staging", builder.getStringAttr("global"));
      kernelState.addAttribute(
          "numeric_policy",
          builder.getDictionaryAttr({
              builder.getNamedAttr("storage",
                                   builder.getStringAttr(selected->storage)),
              builder.getNamedAttr("accum",
                                   builder.getStringAttr(selected->accum)),
          }));
      kernelState.addAttribute("tessera.canonical_k_loop",
                               builder.getBoolAttr(true));
      kernelState.addAttribute("tessera.tile_m",
                               builder.getI64IntegerAttr(selected->tileM));
      kernelState.addAttribute("tessera.tile_n",
                               builder.getI64IntegerAttr(selected->tileN));
      kernelState.addAttribute("tessera.tile_k",
                               builder.getI64IntegerAttr(selected->tileK));
      kernelState.addAttribute(
          "tessera.macro_tile_m",
          builder.getI64IntegerAttr(selected->macroTileM));
      kernelState.addAttribute(
          "tessera.macro_tile_n",
          builder.getI64IntegerAttr(selected->macroTileN));
      kernelState.addAttribute("tessera.pipeline_depth",
                               builder.getI64IntegerAttr(selected->pipelineDepth));
      kernelState.addAttribute("tessera.raster_order",
                               builder.getStringAttr(selected->rasterOrder));
      kernelState.addAttribute("tessera.raster_group",
                               builder.getI64IntegerAttr(selected->rasterGroup));
      kernelState.addAttribute("tessera.schedule_hash",
                               builder.getStringAttr(scheduled.getArtifactHash()));
      builder.create(kernelState);

      Value result = builder.create<bufferization::ToTensorOp>(
          loc, outType, output);
      scheduled.getScheduled().replaceAllUsesWith(result);
      scheduled.erase();
      if (graph->use_empty())
        graph->erase();

      for (schedule::ArtifactOp artifact : matchingArtifacts)
        artifact.erase();
    }

    SmallVector<Operation *> scheduledKernels;
    mod.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "schedule.softmax" || name == "schedule.reduce")
        scheduledKernels.push_back(op);
    });
    for (Operation *scheduled : scheduledKernels) {
      bool isSoftmax = scheduled->getName().getStringRef() == "schedule.softmax";
      Operation *graph = scheduled->getOperand(0).getDefiningOp();
      if (!graph || graph->getNumOperands() != 1 || graph->getNumResults() != 1) {
        scheduled->emitError(
            "E2E-REAL-5 requires the retained Graph semantic-kernel result");
        return signalPassFailure();
      }
      auto selected = getSemanticKernelSchedule(graph);
      auto hash = scheduled->getAttrOfType<StringAttr>("artifact_hash");
      if (failed(selected) || !hash || semanticKernelDigest(*selected) != hash.getValue() ||
          (isSoftmax && selected->family != "softmax") ||
          (!isSoftmax && selected->family != "reduce")) {
        scheduled->emitError(
            "scheduled decision does not match the retained Graph kernel contract");
        return signalPassFailure();
      }
      auto attrString = [&](StringRef name) -> StringRef {
        if (auto attr = scheduled->getAttrOfType<StringAttr>(name)) return attr.getValue();
        return {};
      };
      auto attrInt = [&](StringRef name) -> std::optional<int64_t> {
        if (auto attr = scheduled->getAttrOfType<IntegerAttr>(name)) return attr.getInt();
        return std::nullopt;
      };
      auto attrBool = [&](StringRef name) -> std::optional<bool> {
        if (auto attr = scheduled->getAttrOfType<BoolAttr>(name)) return attr.getValue();
        return std::nullopt;
      };
      bool altered = attrString("arch") != selected->arch ||
                     attrString("storage") != selected->storage ||
                     attrString("accum") != selected->accum ||
                     attrInt("axis") != selected->axis ||
                     attrInt("workgroup_size") != selected->workgroupSize;
      if (isSoftmax)
        altered = altered || attrString("exp_mode") != "accurate" ||
                  attrBool("ftz") != false;
      else
        altered = altered || attrString("kind") != selected->kind ||
                  attrBool("keepdims") != selected->keepdims ||
                  attrString("schedule") != "serial" ||
                  attrString("nan_mode") != "propagate" ||
                  attrBool("inner_is_one") != (selected->inner == 1);
      if (altered) {
        scheduled->emitError(
            "scheduled semantic-kernel policy was altered after hashing");
        return signalPassFailure();
      }
      auto graphDigest = graph->getAttrOfType<StringAttr>("schedule.artifact_hash");
      SmallVector<schedule::ArtifactOp> matchingArtifacts;
      mod.walk([&](schedule::ArtifactOp artifact) {
        if (artifact.getHash() == hash.getValue()) matchingArtifacts.push_back(artifact);
      });
      if (!graphDigest || graphDigest.getValue() != hash.getValue() ||
          matchingArtifacts.size() != 1) {
        scheduled->emitError(
            "requires exactly one matching Graph hash and schedule.artifact");
        return signalPassFailure();
      }

      auto inputType = cast<RankedTensorType>(graph->getOperand(0).getType());
      auto outputType = cast<RankedTensorType>(graph->getResult(0).getType());
      Location loc = scheduled->getLoc();
      builder.setInsertionPoint(scheduled);
      auto pointerType = LLVM::LLVMPointerType::get(&getContext());
      auto inputMemref = MemRefType::get(inputType.getShape(), inputType.getElementType());
      Value inputBuffer = builder.create<bufferization::ToBufferOp>(
          loc, inputMemref, graph->getOperand(0));
      Value inputIndex =
          builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, inputBuffer);
      Value inputInteger =
          builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), inputIndex);
      Value inputPointer =
          builder.create<LLVM::IntToPtrOp>(loc, pointerType, inputInteger);
      auto outputMemref =
          MemRefType::get(outputType.getShape(), outputType.getElementType());
      Value outputBuffer = builder.create<memref::AllocOp>(loc, outputMemref);
      Value outputIndex =
          builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outputBuffer);
      Value outputInteger =
          builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), outputIndex);
      Value outputPointer =
          builder.create<LLVM::IntToPtrOp>(loc, pointerType, outputInteger);

      OperationState kernelState(
          loc, isSoftmax ? "tile.softmax_kernel" : "tile.reduce_kernel");
      if (isSoftmax) {
        Value rows = builder.create<arith::ConstantIntOp>(loc, selected->rows, 64);
        Value columns =
            builder.create<arith::ConstantIntOp>(loc, selected->columns, 64);
        kernelState.addOperands({inputPointer, outputPointer, rows, columns});
        kernelState.addAttribute("storage", builder.getStringAttr(selected->storage));
        kernelState.addAttribute("accum", builder.getStringAttr(selected->accum));
        kernelState.addAttribute("axis", builder.getI64IntegerAttr(-1));
        kernelState.addAttribute("exp_mode", builder.getStringAttr("accurate"));
        kernelState.addAttribute("ftz", builder.getBoolAttr(false));
      } else {
        Value outer = builder.create<arith::ConstantIntOp>(loc, selected->outer, 64);
        Value extent =
            builder.create<arith::ConstantIntOp>(loc, selected->axisExtent, 64);
        Value inner = builder.create<arith::ConstantIntOp>(loc, selected->inner, 64);
        kernelState.addOperands(
            {inputPointer, outputPointer, outer, extent, inner});
        kernelState.addAttribute("storage", builder.getStringAttr(selected->storage));
        kernelState.addAttribute("accum", builder.getStringAttr(selected->accum));
        kernelState.addAttribute("kind", builder.getStringAttr(selected->kind));
        kernelState.addAttribute("axis", builder.getI64IntegerAttr(selected->axis));
        kernelState.addAttribute("keepdims", builder.getBoolAttr(selected->keepdims));
        kernelState.addAttribute("schedule", builder.getStringAttr("serial"));
        kernelState.addAttribute("nan_mode", builder.getStringAttr("propagate"));
        kernelState.addAttribute("inner_is_one",
                                 builder.getBoolAttr(selected->inner == 1));
      }
      kernelState.addAttribute("tessera.workgroup_size",
                               builder.getI64IntegerAttr(selected->workgroupSize));
      kernelState.addAttribute("tessera.schedule_hash", hash);
      builder.create(kernelState);

      Value result = builder.create<bufferization::ToTensorOp>(
          loc, outputType, outputBuffer);
      scheduled->getResult(0).replaceAllUsesWith(result);
      scheduled->erase();
      if (graph->use_empty()) graph->erase();
      for (schedule::ArtifactOp artifact : matchingArtifacts) artifact.erase();
    }
  }
};
} // anonymous namespace

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

std::unique_ptr<mlir::Pass> createPMV11VerifierPass() {
  return std::make_unique<PMV11VerifierPass>();
}

std::unique_ptr<mlir::Pass> createGraphToSchedulePass() {
  return std::make_unique<GraphToSchedulePass>();
}

std::unique_ptr<mlir::Pass> createScheduleToTilePass() {
  return std::make_unique<ScheduleToTilePass>();
}

// ---------------------------------------------------------------------------
// Pipeline builders (called from the tessera-opt driver)
// ---------------------------------------------------------------------------

void buildPMV11VerifyPipeline(OpPassManager &pm) {
  pm.addPass(createPMV11VerifierPass());
  pm.addPass(mlir::createCSEPass());          // expose duplicate ops
  pm.addPass(mlir::createCanonicalizerPass()); // fold trivial patterns
}

void buildPMV11LegalizePipeline(OpPassManager &pm) {
  pm.addPass(createPMV11VerifierPass());    // validate before transforms
  pm.addPass(createGraphToSchedulePass());
  pm.addPass(createScheduleToTilePass());
  pm.addPass(mlir::createCanonicalizerPass());
}

// Register all passes so tessera-opt --help shows them.
void registerPMV11Passes() {
  PassRegistration<PMV11VerifierPass>();
  PassRegistration<GraphToSchedulePass>();
  PassRegistration<ScheduleToTilePass>();

  // Pipelines
  PassPipelineRegistration<>(
      "tessera-pm-verify-pipeline",
      "Verify all Programming Model v1.1 ops",
      [](OpPassManager &pm) { buildPMV11VerifyPipeline(pm); });

  PassPipelineRegistration<>(
      "tessera-pm-legalize-pipeline",
      "Bounded static matmul Graph -> Schedule -> launch Tile lowering",
      [](OpPassManager &pm) { buildPMV11LegalizePipeline(pm); });
}

} // namespace tessera
