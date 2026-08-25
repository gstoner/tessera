//===- PMPasses.cpp — Programming Model v1.1 passes and pipelines ---------===//
//
// Library-owned pass bodies. Previously these lived in
// `tools/tessera-opt/PassPipelinesPM11.cpp`, a driver source, so the passes
// could not be constructed or lit-tested independently of that driver (W0.6).
//
// See PMPasses.h for the maturity contract: the verifier is general, while the
// two lowering passes are real only for explicitly registered bounded family
// contracts and fail closed outside them.
//
//===----------------------------------------------------------------------===//

#include "tessera/ProgrammingModel/PMPasses.h"
#include "tessera/ProgrammingModel/ScheduleDialect.h"
#include "Tessera/Dialect/Tile/TileDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>

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
  // Inline lightweight verification — the single production implementation of
  // the schedule./cache./tile. structural checks (Decision #31; the unbuilt
  // ScheduleOps.cpp duplicate was deleted 2026-08-10).
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

    // tile.async_copy / tile.wait_async — `stage` is an OPTIONAL legacy-form
    // grouping key under the declared contract (TileOps.td): the production
    // dependency encoding is the `!tile.async_token` SSA edge, which carries no
    // stage at all. Requiring the attribute here rejected the production
    // TileIRLoweringPass output; only well-formedness is checked. The full
    // contract (token placement, producer identity) is owned by the registered
    // dialect's AsyncCopyOp/WaitAsyncOp verifiers in TileOps.cpp.
    if (name == "tile.async_copy" || name == "tile.wait_async") {
      if (auto stage = op->getAttrOfType<IntegerAttr>("stage");
          stage && stage.getInt() < 0)
        return op->emitOpError("'stage' must be >= 0 when present");
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
  bool dynamicM = false;
  bool dynamicN = false;
  bool dynamicK = false;
  bool bias = false;
  bool residual = false;
  StringRef activation = "none";
  StringRef output = "f32";
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
  if (!module || op->getNumOperands() < 2 || op->getNumOperands() > 4 ||
      op->getNumResults() != 1)
    return failure();
  auto lhs = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto rhs = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto out = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!lhs || !rhs || !out || lhs.getRank() != 2 || rhs.getRank() != 2 ||
      out.getRank() != 2)
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
  bool nvidia_sm120 = schedule.target == "nvidia_sm120" &&
                      (schedule.arch.empty() || schedule.arch.contains("sm_120"));
  bool rocm_gfx1151 =
      (schedule.target == "rocm" || schedule.target == "rocm_gfx1151") &&
      (schedule.arch.empty() || schedule.arch.contains("gfx1151"));
  SmallVector<int64_t> bounds;
  if (auto attr = op->getAttrOfType<ArrayAttr>("shape_bounds")) {
    for (Attribute value : attr) {
      auto integer = dyn_cast<IntegerAttr>(value);
      if (!integer)
        return failure();
      bounds.push_back(integer.getInt());
    }
  }
  auto bounded = [&](int64_t extent, unsigned boundIndex,
                     bool &dynamic) -> FailureOr<int64_t> {
    dynamic = ShapedType::isDynamic(extent);
    if (!dynamic)
      return extent;
    if ((!nvidia_sm120 && !rocm_gfx1151) || bounds.size() != 3 ||
        bounds[boundIndex] <= 0)
      return failure();
    return bounds[boundIndex];
  };
  auto m = bounded(lhs.getDimSize(0), 0, schedule.dynamicM);
  auto n = bounded(rhs.getDimSize(1), 1, schedule.dynamicN);
  auto k = bounded(lhs.getDimSize(1), 2, schedule.dynamicK);
  if (failed(m) || failed(n) || failed(k))
    return failure();
  schedule.m = *m;
  schedule.n = *n;
  schedule.k = *k;
  auto compatible = [](int64_t extent, int64_t expected) {
    return ShapedType::isDynamic(extent) || extent == expected;
  };
  if (schedule.m <= 0 || schedule.n <= 0 || schedule.k <= 0 ||
      !compatible(rhs.getDimSize(0), schedule.k) ||
      !compatible(out.getDimSize(0), schedule.m) ||
      !compatible(out.getDimSize(1), schedule.n))
    return failure();

  Type lhsElement = lhs.getElementType();
  Type rhsElement = rhs.getElementType();
  Type outElement = out.getElementType();
  schedule.bias = bool(op->getAttrOfType<StringAttr>("bias"));
  schedule.residual = bool(op->getAttrOfType<StringAttr>("residual"));
  if (auto activation = op->getAttrOfType<StringAttr>("activation"))
    schedule.activation = activation.getValue();
  if (!llvm::is_contained({"none", "relu", "gelu", "silu"},
                          schedule.activation))
    return failure();
  if (op->getNumOperands() !=
      static_cast<unsigned>(2 + schedule.bias + schedule.residual))
    return failure();
  unsigned epilogueOperand = 2;
  if (schedule.bias) {
    auto bias = dyn_cast<RankedTensorType>(
        op->getOperand(epilogueOperand++).getType());
    if (!bias || bias.getRank() != 1 || !bias.getElementType().isF32() ||
        !compatible(bias.getDimSize(0), schedule.n))
      return failure();
  }
  if (schedule.residual) {
    auto residual = dyn_cast<RankedTensorType>(
        op->getOperand(epilogueOperand).getType());
    if (!residual || residual.getRank() != 2 ||
        !residual.getElementType().isF32() ||
        !compatible(residual.getDimSize(0), schedule.m) ||
        !compatible(residual.getDimSize(1), schedule.n))
      return failure();
  }
  bool x86 = schedule.target == "x86" || schedule.arch.contains("avx512") ||
             schedule.arch.contains("zen5");
  // This bounded physical schedule is gfx1151-owned.  The shared macro-tile
  // vocabulary is portable, but gfx1200/gfx1250 must supply their own exact-
  // device schedule and instruction-family profile rather than inheriting it.
  bool rocm = schedule.arch.contains("gfx1151");
  if ((schedule.bias || schedule.residual || schedule.activation != "none") &&
      !nvidia_sm120)
    return failure();
  // Apple GPU has no rank-2 f32 cooperative-matrix GEMM: the shared launch
  // contract is consumed as a batch-1 MPS BMM (apple_gpu_bmm_f32_batch1).  The
  // macro-tile below is a logical default only; the MPS route owns its own
  // tiling, which the Apple package records as an explicit dropped decision.
  bool apple_gpu = schedule.target == "apple_gpu";
  if (x86 && lhsElement.isF32() && rhsElement.isF32() && outElement.isF32()) {
    schedule.storage = "f32";
    schedule.accum = "f32";
    if (schedule.arch.empty())
      schedule.arch = "x86-avx512";
    return schedule;
  }
  if (apple_gpu && lhsElement.isF32() && rhsElement.isF32() &&
      outElement.isF32()) {
    schedule.storage = "f32";
    schedule.accum = "f32";
    if (schedule.arch.empty())
      schedule.arch = "apple7";
    return schedule;
  }
  if (apple_gpu && lhsElement.isF16() && rhsElement.isF16() &&
      outElement.isF32()) {
    // Apple7+ simdgroup_matrix GEMM: f16 storage, f32 accumulation.  Unlike the
    // f32 batch-1 BMM above, this is compiler-emitted MSL with a device timer,
    // so its promotion is not gated by APPLE-DEVICE-EVENT-1.  The 32x32
    // macro-tile mirrors the proven APPLE-TILE-1 emitter default (a multiple of
    // the 8x8x8 simdgroup fragment); the MSL materializer owns the byte layout.
    schedule.storage = "f16";
    schedule.accum = "f32";
    schedule.macroTileM = 32;
    schedule.macroTileN = 32;
    if (schedule.arch.empty())
      schedule.arch = "apple7";
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
  if (nvidia_sm120 &&
      ((lhsElement.isF16() && rhsElement.isF16()) ||
       (lhsElement.isBF16() && rhsElement.isBF16())) &&
      (outElement.isF32() || outElement.isF16())) {
    // Consumer Blackwell's owned MMA contract is warp-level m16n8k16 with
    // fp16 storage and fp32 accumulation.  The macro tile remains a launch
    // envelope; Tile/Target lowering owns fragment packing and repetition.
    schedule.storage = lhsElement.isBF16() ? "bf16" : "f16";
    schedule.accum = "f32";
    schedule.output = outElement.isF16() ? "f16" : "f32";
    schedule.tileM = 16;
    schedule.tileN = 8;
    schedule.tileK = 16;
    schedule.macroTileM = 128;
    schedule.macroTileN = 128;
    schedule.warps = 1;
    if (schedule.arch.empty())
      schedule.arch = "sm_120";
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
       Twine(schedule.rasterGroup) + ";epilogue=b" + Twine(schedule.bias) +
       ",a=" + schedule.activation + ",r=" + Twine(schedule.residual) +
       ",out=" + schedule.output + ";dynamic=" +
       Twine(schedule.dynamicM) + Twine(schedule.dynamicN) +
       Twine(schedule.dynamicK))
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
  // Apple GPU consumes the shared softmax launch tile through its f32 MSL ABI.
  // It has no scheduled reduction consumer yet, so it is admitted for softmax
  // only below.  The launch tile does not prescribe a workgroup for Apple (the
  // delegated MSL route owns its threadgroup), so workgroupSize stays 1.
  bool apple_gpu = schedule.target == "apple_gpu";
  if (!x86 && !rocm && !apple_gpu)
    return failure();

  StringRef opName = op->getName().getStringRef();
  if (opName == "tessera.softmax") {
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    if ((axisAttr && axisAttr.getInt() != -1) || input != output ||
        (x86 && schedule.storage != "f32") ||
        (rocm && schedule.storage != "f16" && schedule.storage != "f32") ||
        (apple_gpu && schedule.storage != "f32"))
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
      (x86 && axis != input.getRank() - 1) ||
      // Apple's synthesized reduce kernel gives one thread per row and folds
      // over the trailing extent, so it expresses last-axis reductions only.
      // An interior axis (inner != 1) fails closed rather than being reordered
      // into something the program did not ask for.
      (apple_gpu && axis != input.getRank() - 1))
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

struct FFTSchedule {
  StringRef target;
  StringRef arch;
  StringRef mode;
  StringRef radixPolicy;
  StringRef strategy;
  StringRef algorithm;
  StringRef realTransformPolicy;
  StringRef hermitianLayout;
  StringRef hermitianWeight;
  StringRef normalization;
  StringRef kernelFamily;
  StringRef workspacePolicy;
  StringRef residency;
  StringRef twiddlePolicy;
  SmallVector<int64_t> inputShape;
  SmallVector<int64_t> outputShape;
  SmallVector<int64_t> radixSequence;
  int64_t axis = -1;
  int64_t length = 0;
  int64_t physicalLength = 0;
  int64_t batch = 0;
  bool inverse = false;
  double scale = 1.0;
  int64_t bluesteinM = 0;
  int64_t workspaceElems = 0;
  int64_t workgroupSize = 1;
};

static int64_t nextPowerOfTwo(int64_t value) {
  int64_t result = 1;
  while (result < value &&
         result <= (std::numeric_limits<int64_t>::max() / 2))
    result <<= 1;
  return result;
}

static bool isPowerOfTwo(int64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

static std::optional<SmallVector<int64_t>> mixedRadixSequence(int64_t value) {
  SmallVector<int64_t> stages;
  int64_t rest = value;
  while (rest % 4 == 0) { stages.push_back(4); rest /= 4; }
  while (rest % 2 == 0) { stages.push_back(2); rest /= 2; }
  for (int64_t radix = 3; radix <= 17; radix += 2)
    while (rest % radix == 0) {
      stages.push_back(radix);
      rest /= radix;
    }
  if (rest != 1) return std::nullopt;
  return stages;
}

static bool preferX86MixedRadix(int64_t length, ArrayRef<int64_t> stages) {
  if (length <= 8 || isPowerOfTwo(length) || stages.empty()) return false;
  int64_t padded = nextPowerOfTwo(2 * length - 1);
  int64_t log2Padded = 0;
  for (int64_t rest = padded; rest > 1; rest /= 2) ++log2Padded;
  long double directWork = static_cast<long double>(length);
  int64_t radixSum = 0;
  for (int64_t stage : stages) radixSum += stage;
  directWork *= radixSum;
  long double bluesteinWork =
      3.0L * static_cast<long double>(padded) * log2Padded;
  return 5.0L * directWork <= 2.0L * bluesteinWork;
}

static FailureOr<FFTSchedule> getFFTSchedule(Operation *op) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module || op->getNumOperands() != 1 || op->getNumResults() != 1)
    return failure();
  auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!input || !output || input.getRank() < 1 || !input.hasStaticShape() ||
      !output.hasStaticShape() || input.getRank() != output.getRank())
    return failure();

  FFTSchedule schedule;
  schedule.target = moduleString(module, "tessera.target", "target");
  schedule.arch = moduleString(module, "tessera.arch", "arch");
  bool x86 = schedule.target == "x86" || schedule.arch.contains("avx512") ||
             schedule.arch.contains("zen5");
  bool rocm = schedule.arch.contains("gfx1151");
  if (!x86 && !rocm) return failure();
  schedule.radixPolicy = rocm ? "mixed_radix" : "radix2";
  schedule.algorithm = rocm ? "stockham_autosort" : "cooley_tukey_dit";
  schedule.residency = rocm ? "persistent_device_plan" : "host_inplace";
  schedule.twiddlePolicy =
      rocm ? "device_sincos_per_butterfly" : "thread_local_cached_f32";
  schedule.kernelFamily =
      rocm ? "gfx1151_stockham_bluestein_v6" : "zen5_avx512_fft_v4";
  schedule.workgroupSize = rocm ? 256 : 1;
  schedule.inputShape.assign(input.getShape().begin(), input.getShape().end());
  schedule.outputShape.assign(output.getShape().begin(), output.getShape().end());

  auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
  schedule.axis = axisAttr ? axisAttr.getInt() : -1;
  if (schedule.axis < 0) schedule.axis += input.getRank();
  if (schedule.axis < 0 || schedule.axis >= input.getRank()) return failure();
  auto norm = op->getAttrOfType<StringAttr>("norm");
  auto normalization = op->getAttrOfType<StringAttr>("normalization");
  schedule.normalization = normalization ? normalization.getValue() :
                                           norm ? norm.getValue() : "backward";
  if (schedule.normalization != "backward" &&
      schedule.normalization != "forward" &&
      schedule.normalization != "ortho")
    return failure();
  auto weight = op->getAttrOfType<StringAttr>("hermitian_weight");
  schedule.hermitianWeight = weight ? weight.getValue() : "none";
  if (schedule.hermitianWeight != "none" &&
      schedule.hermitianWeight != "half_interior" &&
      schedule.hermitianWeight != "double_interior")
    return failure();

  StringRef name = op->getName().getStringRef();
  auto isComplexF32 = [](Type type) {
    auto complex = dyn_cast<ComplexType>(type);
    return complex && complex.getElementType().isF32();
  };
  if (name == "tessera.fft" || name == "tessera.ifft") {
    if (!isComplexF32(input.getElementType()) ||
        !isComplexF32(output.getElementType()))
      return failure();
    schedule.mode = "c2c";
    schedule.inverse = name == "tessera.ifft";
    schedule.length = input.getDimSize(schedule.axis);
    if (input != output) return failure();
  } else if (name == "tessera.rfft") {
    if (!input.getElementType().isF32() ||
        !isComplexF32(output.getElementType()))
      return failure();
    schedule.mode = "r2c";
    schedule.length = input.getDimSize(schedule.axis);
    if (output.getDimSize(schedule.axis) != schedule.length / 2 + 1)
      return failure();
  } else if (name == "tessera.irfft") {
    if (!isComplexF32(input.getElementType()) ||
        !output.getElementType().isF32())
      return failure();
    schedule.mode = "c2r";
    schedule.inverse = true;
    auto lengthAttr = op->getAttrOfType<IntegerAttr>("logical_length");
    if (!lengthAttr) lengthAttr = op->getAttrOfType<IntegerAttr>("n");
    schedule.length =
        lengthAttr ? lengthAttr.getInt() : output.getDimSize(schedule.axis);
    if (output.getDimSize(schedule.axis) != schedule.length ||
        input.getDimSize(schedule.axis) != schedule.length / 2 + 1)
      return failure();
  } else {
    return failure();
  }
  if (auto logicalLength =
          op->getAttrOfType<IntegerAttr>("logical_length")) {
    if (logicalLength.getInt() != schedule.length) return failure();
  }
  if (schedule.length <= 0) return failure();
  for (int64_t dimension = 0; dimension < input.getRank(); ++dimension) {
    if (dimension == schedule.axis) continue;
    if (input.getDimSize(dimension) != output.getDimSize(dimension))
      return failure();
    schedule.batch = schedule.batch == 0
                         ? input.getDimSize(dimension)
                         : schedule.batch * input.getDimSize(dimension);
  }
  if (input.getRank() == 1) schedule.batch = 1;
  schedule.scale = schedule.normalization == "ortho"
                       ? 1.0 / std::sqrt(static_cast<double>(schedule.length))
                   : schedule.normalization == "forward"
                       ? (schedule.inverse ? 1.0
                                           : 1.0 / schedule.length)
                       : (schedule.inverse ? 1.0 / schedule.length : 1.0);

  const bool realMode = schedule.mode == "r2c" || schedule.mode == "c2r";
  const int64_t candidatePhysicalLength =
      realMode && schedule.length % 2 == 0 ? schedule.length / 2
                                           : schedule.length;
  auto candidateStages = mixedRadixSequence(candidatePhysicalLength);
  const bool packedReal =
      realMode && schedule.length % 2 == 0 &&
      (rocm || isPowerOfTwo(candidatePhysicalLength) ||
       (candidateStages &&
        preferX86MixedRadix(candidatePhysicalLength, *candidateStages)));
  schedule.physicalLength =
      packedReal ? candidatePhysicalLength : schedule.length;
  schedule.realTransformPolicy =
      packedReal ? "packed_even_n2_hermitian_v1"
                 : realMode ? "full_complex_hermitian_fallback"
                            : "not_applicable";
  schedule.hermitianLayout =
      realMode ? "half_spectrum_nyquist_explicit" : "full_complex";

  if (rocm) {
    auto stages = mixedRadixSequence(schedule.physicalLength);
    if (stages) {
      schedule.strategy = "mixed_radix";
      schedule.radixSequence = *stages;
      schedule.workspaceElems = schedule.physicalLength;
      schedule.workspacePolicy = "persistent_plan_n";
    } else {
      schedule.strategy = "bluestein";
      schedule.bluesteinM = nextPowerOfTwo(2 * schedule.physicalLength - 1);
      schedule.workspaceElems = 4 * schedule.bluesteinM;
      schedule.workspacePolicy = "persistent_plan_4m";
      schedule.twiddlePolicy = "persistent_device_chirp_fft";
    }
    if (schedule.strategy != "bluestein" && schedule.physicalLength <= 1024 &&
        isPowerOfTwo(schedule.physicalLength))
      schedule.residency = packedReal
                               ? "persistent_device_plan_fused_lds_hermitian_batch"
                               : "persistent_device_plan_fused_lds_batch";
  } else {
    auto stages = mixedRadixSequence(schedule.physicalLength);
    if (stages && preferX86MixedRadix(schedule.physicalLength, *stages)) {
      schedule.radixPolicy = "mixed_radix";
      schedule.strategy = "mixed_radix";
      schedule.algorithm = "stockham_autosort";
      schedule.radixSequence = *stages;
      schedule.workspaceElems = 2 * schedule.physicalLength;
      schedule.workspacePolicy = "thread_local_2n";
      schedule.residency = "host_thread_local_ping_pong";
    } else if (isPowerOfTwo(schedule.physicalLength)) {
      schedule.strategy = "radix2";
      for (int64_t rest = schedule.physicalLength; rest > 1; rest /= 2)
        schedule.radixSequence.push_back(2);
      schedule.workspacePolicy = "inplace_no_scratch";
    } else if (schedule.physicalLength <= 8) {
      schedule.strategy = "dft";
      schedule.workspacePolicy = "inplace_no_scratch";
    } else {
      schedule.strategy = "bluestein";
      schedule.bluesteinM = nextPowerOfTwo(2 * schedule.physicalLength - 1);
      schedule.workspaceElems = schedule.bluesteinM;
      schedule.workspacePolicy = "host_temporary_m";
    }
  }
  return schedule;
}

static std::string fftScheduleDigest(const FFTSchedule &schedule) {
  auto shapeText = [](ArrayRef<int64_t> shape) {
    std::string value;
    for (int64_t dim : shape)
      value += (value.empty() ? "" : "x") + Twine(dim).str();
    return value;
  };
  std::string radix;
  for (int64_t stage : schedule.radixSequence)
    radix += (radix.empty() ? "" : ",") + Twine(stage).str();
  std::string contract =
      (Twine("family=fft;target=") + schedule.target + ";arch=" +
       schedule.arch + ";mode=" + schedule.mode + ";input=" +
       shapeText(schedule.inputShape) + ";output=" +
       shapeText(schedule.outputShape) + ";axis=" + Twine(schedule.axis) +
       ";length=" + Twine(schedule.length) + ";batch=" +
       Twine(schedule.batch) + ";physical_length=" +
       Twine(schedule.physicalLength) + ";real_transform_policy=" +
       schedule.realTransformPolicy + ";hermitian_layout=" +
       schedule.hermitianLayout + ";hermitian_weight=" +
       schedule.hermitianWeight + ";inverse=" +
       Twine(schedule.inverse ? 1 : 0) + ";normalization=" +
       schedule.normalization + ";scale=" +
       std::to_string(static_cast<float>(schedule.scale)) +
       ";storage=complex64_interleaved_f32;accum=f32;radix_policy=" +
       schedule.radixPolicy + ";strategy=" + schedule.strategy +
       ";algorithm=" + schedule.algorithm +
       ";radix=" + radix + ";bluestein_m=" +
       Twine(schedule.bluesteinM) + ";workspace=" +
       Twine(schedule.workspaceElems) + ";workspace_policy=" +
       schedule.workspacePolicy + ";residency=" + schedule.residency +
       ";twiddle=interleaved_f32;twiddle_policy=" + schedule.twiddlePolicy +
       ";deterministic=1;kernel=" +
       schedule.kernelFamily + ";workgroup=" + Twine(schedule.workgroupSize))
          .str();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(contract)),
                     /*LowerCase=*/true);
}

static std::string spectralBackwardTypeSignature(ValueRange values) {
  std::string signature;
  llvm::raw_string_ostream stream(signature);
  llvm::interleave(
      values, stream,
      [&](Value value) { value.getType().print(stream); }, ",");
  stream.flush();
  return signature;
}

static FailureOr<std::string> spectralBackwardDigest(Operation *op) {
  auto stringAttr = [&](StringRef name) {
    return op->getAttrOfType<StringAttr>(name);
  };
  auto kind = stringAttr("kind");
  auto target = stringAttr("target");
  auto arch = stringAttr("arch");
  auto normalization = stringAttr("normalization");
  auto layout = stringAttr("spectrum_layout");
  auto padMode = stringAttr("pad_mode");
  auto lineage = stringAttr("mutation_lineage");
  if (!kind || !target || !arch || !normalization || !layout || !padMode ||
      !lineage)
    return failure();
  auto integer = [&](StringRef name, int64_t fallback) {
    auto attr = op->getAttrOfType<IntegerAttr>(name);
    return attr ? attr.getInt() : fallback;
  };
  auto boolean = [&](StringRef name, bool fallback) {
    auto attr = op->getAttrOfType<BoolAttr>(name);
    return attr ? attr.getValue() : fallback;
  };
  auto inputSignature = stringAttr("input_signature");
  auto outputSignature = stringAttr("output_signature");
  std::string inputs = inputSignature
                           ? inputSignature.getValue().str()
                           : spectralBackwardTypeSignature(op->getOperands());
  std::string outputs = outputSignature
                            ? outputSignature.getValue().str()
                            : spectralBackwardTypeSignature(op->getResults());
  std::string contract =
      (Twine("schema=tessera.spectral_backward.v1;kind=") + kind.getValue() +
       ";target=" + target.getValue() + ";arch=" + arch.getValue() +
       ";inputs=" + inputs + ";outputs=" + outputs +
       ";axis=" + Twine(integer("axis", -1)) + ";logical_length=" +
       Twine(integer("logical_length", -1)) + ";normalization=" +
       normalization.getValue() + ";spectrum_layout=" + layout.getValue() +
       ";hop=" + Twine(integer("hop", -1)) + ";center=" +
       Twine(boolean("center", false) ? 1 : 0) + ";onesided=" +
       Twine(boolean("onesided", true) ? 1 : 0) + ";pad_mode=" +
       padMode.getValue() + ";output_length=" +
       Twine(integer("output_length", -1)) + ";mutation_lineage=" +
       lineage.getValue())
          .str();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(contract)),
                     /*LowerCase=*/true);
}

static FailureOr<std::pair<std::string, std::string>>
esLowRankContract(Operation *op, StringRef architecture) {
  if (op->getName().getStringRef() != "tessera.es_low_rank_correction" ||
      op->getNumOperands() != 3 || op->getNumResults() != 1)
    return failure();
  auto x = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto members = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto key = dyn_cast<RankedTensorType>(op->getOperand(2).getType());
  auto result = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto outDim = op->getAttrOfType<IntegerAttr>("out_dim");
  auto rank = op->getAttrOfType<IntegerAttr>("rank");
  auto sigma = op->getAttrOfType<FloatAttr>("sigma");
  if (!x || !members || !key || !result || !x.hasStaticShape() ||
      !members.hasStaticShape() || !key.hasStaticShape() ||
      !result.hasStaticShape() || !x.getElementType().isF32() || !outDim ||
      !rank || !sigma)
    return failure();
  int64_t rows = 1;
  for (int64_t dim : x.getShape().drop_front().drop_back()) rows *= dim;
  int64_t epoch = 0;
  if (auto attr = op->getAttrOfType<IntegerAttr>("epoch")) epoch = attr.getInt();
  bool antithetic = true;
  if (auto attr = op->getAttrOfType<BoolAttr>("antithetic"))
    antithetic = attr.getValue();
  StringRef score = "gaussian";
  if (auto attr = op->getAttrOfType<StringAttr>("score")) score = attr.getValue();
  std::string sigmaText = std::to_string(sigma.getValueAsDouble());
  std::string payload =
      (Twine("schema=tessera.es_low_rank_correction.v1;arch=") + architecture +
       ";" +
       "population=" + Twine(x.getDimSize(0)) + ";rows_per_member=" +
       Twine(rows) + ";in_dim=" + Twine(x.getShape().back()) + ";out_dim=" +
       Twine(outDim.getInt()) + ";rank=" + Twine(rank.getInt()) + ";epoch=" +
       Twine(epoch) + ";sigma=" + sigmaText +
       ";antithetic=" + Twine(antithetic ? 1 : 0) + ";score=" + score +
       ";rng=splitmix64-philox4x32-boxmuller;version=1;storage=f32;accum=f32;" +
       "lineage=x/member_ids/key-read-only;correction-fresh")
          .str();
  std::string hash = llvm::toHex(
      llvm::SHA256::hash(llvm::arrayRefFromStringRef(payload)),
      /*LowerCase=*/true);
  return std::make_pair(std::move(payload), std::move(hash));
}

struct AttentionSchedule {
  StringRef target;
  StringRef arch;
  StringRef storage;
  StringRef accum = "f32";
  SmallVector<int64_t> qShape;
  SmallVector<int64_t> kShape;
  SmallVector<int64_t> vShape;
  SmallVector<int64_t> outputShape;
  int64_t batch;
  int64_t queryHeads;
  int64_t kvHeads;
  int64_t queryRows;
  int64_t keyRows;
  int64_t headDim;
  int64_t valueDim;
  double scale;
  bool causal;
  bool bias = false;
  int64_t windowLeft;
  int64_t windowRight;
  double softcap;
  double dropoutP;
  int64_t dropoutSeed;
  int64_t tileQ;
  int64_t tileKV = 16;
  int64_t workgroupSize;
  StringRef recurrence = "rank4_batch_query_head_kv_online_softmax_v1";
  StringRef backwardLsePolicy;
  StringRef backwardLseSelection;
};

struct DepthAttentionSchedule {
  StringRef target;
  StringRef arch;
  StringRef storage;
  StringRef softmax = "f32";
  StringRef accum = "f32";
  int64_t sourceCount;
  int64_t rows;
  int64_t width;
  double eps;
  int64_t sourceTile;
  int64_t workgroupSize;
  StringRef statisticsRecurrence = "rms_key_online_softmax_stats_v1";
  StringRef mergeRecurrence = "max_shifted_pairwise_merge_v1";
  bool reassociable = true;
};

struct TridiagonalSchedule {
  StringRef target;
  StringRef arch;
  StringRef storage = "f32";
  StringRef accum = "f64";
  int64_t batch;
  int64_t systemSize;
  int64_t workgroupSize;
  StringRef algorithm;
};

static FailureOr<TridiagonalSchedule>
getTridiagonalSchedule(Operation *op) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module || op->getName().getStringRef() !=
                     "tessera.tridiagonal_solve" ||
      op->getNumOperands() != 4 || op->getNumResults() != 1)
    return failure();
  auto dl = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto diagonal = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto du = dyn_cast<RankedTensorType>(op->getOperand(2).getType());
  auto rhs = dyn_cast<RankedTensorType>(op->getOperand(3).getType());
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!dl || !diagonal || !du || !rhs || !output || !dl.hasStaticShape() ||
      !diagonal.hasStaticShape() || !du.hasStaticShape() ||
      !rhs.hasStaticShape() || !output.hasStaticShape() || dl.getRank() != 1 ||
      diagonal.getRank() != 1 || du.getRank() != 1 || rhs.getRank() < 1 ||
      dl.getShape() != diagonal.getShape() || du.getShape() != diagonal.getShape() ||
      rhs.getShape() != output.getShape() ||
      rhs.getShape().back() != diagonal.getDimSize(0) ||
      dl.getElementType() != diagonal.getElementType() ||
      du.getElementType() != diagonal.getElementType() ||
      rhs.getElementType() != diagonal.getElementType() ||
      output.getElementType() != diagonal.getElementType() ||
      !diagonal.getElementType().isF32())
    return failure();

  TridiagonalSchedule schedule;
  schedule.target = moduleString(module, "tessera.target", "target");
  schedule.arch = moduleString(module, "tessera.arch", "arch");
  bool x86 = schedule.target == "x86" &&
             (schedule.arch.contains("avx512") || schedule.arch.contains("zen5"));
  bool rocm = schedule.target == "rocm" && schedule.arch.contains("gfx1151");
  if (!x86 && !rocm)
    return failure();
  schedule.systemSize = diagonal.getDimSize(0);
  schedule.batch = 1;
  for (int64_t extent : rhs.getShape().drop_back())
    schedule.batch *= extent;
  if (schedule.systemSize <= 0 || schedule.batch <= 0)
    return failure();
  if (rocm && (schedule.systemSize > 256 ||
               !llvm::isPowerOf2_64(schedule.systemSize)))
    return failure();
  schedule.workgroupSize = rocm ? schedule.systemSize : 1;
  schedule.algorithm = rocm ? "cooperative_lds_pcr_v1"
                            : "batch_vector_thomas_v1";
  return schedule;
}

static std::string
tridiagonalScheduleDigest(const TridiagonalSchedule &schedule) {
  std::string payload =
      (Twine("schema=tessera.tridiagonal.schedule.v1;target=") +
       schedule.target + ";arch=" + schedule.arch + ";batch=" +
       Twine(schedule.batch) + ";n=" + Twine(schedule.systemSize) +
       ";storage=" + schedule.storage + ";accum=" + schedule.accum +
       ";algorithm=" + schedule.algorithm + ";workgroup=" +
       Twine(schedule.workgroupSize))
          .str();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(payload)),
                     /*LowerCase=*/true);
}

struct CoalitionButterflySchedule {
  StringRef target;
  StringRef arch;
  StringRef storage = "f32";
  StringRef accum = "f64";
  int64_t batch;
  int64_t latticeSize;
  int64_t players;
  int64_t half;
  int64_t sign;
  int64_t workgroupSize;
  StringRef stageOrder = "ascending_bit_yates_v1";
};

static FailureOr<CoalitionButterflySchedule>
getCoalitionButterflySchedule(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name != "tessera.game_subset_zeta" &&
      name != "tessera.game_subset_mobius" &&
      name != "tessera.game_superset_zeta" &&
      name != "tessera.game_superset_mobius")
    return failure();
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module || op->getNumOperands() != 1 || op->getNumResults() != 1)
    return failure();
  auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!input || !output || !input.hasStaticShape() || !output.hasStaticShape() ||
      input.getShape() != output.getShape() || input.getRank() < 1 ||
      !input.getElementType().isF32() ||
      input.getElementType() != output.getElementType())
    return failure();
  int64_t size = input.getShape().back();
  if (size <= 0 || !llvm::isPowerOf2_64(size))
    return failure();
  CoalitionButterflySchedule schedule;
  schedule.target = moduleString(module, "tessera.target", "target");
  schedule.arch = moduleString(module, "tessera.arch", "arch");
  bool x86 = schedule.target == "x86" &&
             (schedule.arch.contains("avx512") || schedule.arch.contains("zen5"));
  bool rocm = schedule.target == "rocm" && schedule.arch.contains("gfx1151");
  if (!x86 && !rocm)
    return failure();
  // The physical gfx1151 consumer retains the complete fp64 lattice in LDS.
  // Fail before artifact creation when that state cannot fit in 64 KiB/CU.
  if (rocm && size > (64 * 1024) / static_cast<int64_t>(sizeof(double)))
    return failure();
  schedule.batch = 1;
  for (int64_t extent : input.getShape().drop_back())
    schedule.batch *= extent;
  schedule.latticeSize = size;
  schedule.players = llvm::Log2_64(size);
  schedule.half = name.contains("subset") ? 1 : 0;
  schedule.sign = name.contains("mobius") ? -1 : 1;
  schedule.workgroupSize = rocm ? std::min<int64_t>(256, size) : 1;
  return schedule;
}

static std::string coalitionButterflyDigest(
    const CoalitionButterflySchedule &schedule) {
  std::string payload =
      (Twine("schema=tessera.coalition_butterfly.schedule.v1;target=") +
       schedule.target + ";arch=" + schedule.arch + ";batch=" +
       Twine(schedule.batch) + ";size=" + Twine(schedule.latticeSize) +
       ";players=" + Twine(schedule.players) + ";half=" +
       Twine(schedule.half) + ";sign=" + Twine(schedule.sign) +
       ";storage=" + schedule.storage + ";accum=" + schedule.accum +
       ";stage_order=" + schedule.stageOrder + ";workgroup=" +
       Twine(schedule.workgroupSize))
          .str();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(payload)),
                     /*LowerCase=*/true);
}

static FailureOr<DepthAttentionSchedule>
getDepthAttentionSchedule(Operation *op) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module || op->getName().getStringRef() != "tessera.depth_attn" ||
      op->getNumOperands() != 2 || op->getNumResults() != 1)
    return failure();
  auto query = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto sources = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!query || !sources || !output || !query.hasStaticShape() ||
      !sources.hasStaticShape() || !output.hasStaticShape() ||
      query.getRank() != 1 || sources.getRank() < 2 ||
      output.getRank() != sources.getRank() - 1 ||
      query.getElementType() != sources.getElementType() ||
      output.getElementType() != sources.getElementType() ||
      query.getDimSize(0) != sources.getShape().back() ||
      output.getShape() != sources.getShape().drop_front())
    return failure();

  DepthAttentionSchedule schedule;
  schedule.target = moduleString(module, "tessera.target", "target");
  schedule.arch = moduleString(module, "tessera.arch", "arch");
  bool x86 = schedule.target == "x86" &&
             (schedule.arch.contains("avx512") || schedule.arch.contains("zen5"));
  bool rocm = schedule.target == "rocm" && schedule.arch.contains("gfx1151");
  if (!x86 && !rocm)
    return failure();
  schedule.storage = storageName(sources.getElementType());
  // Phase 4 freezes a portable all-f32 artifact. Reduced storage becomes a
  // separate evidence-backed contract rather than an implicit target choice.
  if (schedule.storage != "f32")
    return failure();
  auto eps = op->getAttrOfType<FloatAttr>("eps");
  auto policy = op->getAttrOfType<DictionaryAttr>("numeric_policy");
  if (!eps || !eps.getValue().isFinite() || eps.getValueAsDouble() <= 0.0 ||
      !policy)
    return failure();
  auto policyString = [&](StringRef name) -> StringRef {
    auto value = dyn_cast_or_null<StringAttr>(policy.get(name));
    return value ? value.getValue() : StringRef();
  };
  if (policyString("storage") != "fp32" ||
      policyString("softmax") != "fp32" ||
      policyString("accum") != "fp32")
    return failure();
  schedule.sourceCount = sources.getDimSize(0);
  schedule.width = sources.getShape().back();
  schedule.rows = 1;
  for (int64_t dim : sources.getShape().drop_front().drop_back())
    schedule.rows *= dim;
  if (schedule.sourceCount <= 0 || schedule.rows <= 0 || schedule.width <= 0)
    return failure();
  schedule.eps = eps.getValueAsDouble();
  schedule.sourceTile = std::min<int64_t>(schedule.sourceCount, 16);
  schedule.workgroupSize = rocm ? 256 : 1;
  return schedule;
}

static std::string
depthAttentionScheduleDigest(const DepthAttentionSchedule &schedule) {
  std::string payload =
      (Twine("schema=tessera.depth_attention.schedule.v1;target=") +
       schedule.target + ";arch=" + schedule.arch + ";sources=" +
       Twine(schedule.sourceCount) + ";rows=" + Twine(schedule.rows) +
       ";width=" + Twine(schedule.width) + ";storage=" + schedule.storage +
       ";softmax=" + schedule.softmax + ";accum=" + schedule.accum +
       ";eps=" + std::to_string(schedule.eps) + ";source_tile=" +
       Twine(schedule.sourceTile) + ";workgroup=" +
       Twine(schedule.workgroupSize) + ";statistics=" +
       schedule.statisticsRecurrence + ";merge=" + schedule.mergeRecurrence +
       ";reassociable=1")
          .str();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(payload)),
                     /*LowerCase=*/true);
}

static FailureOr<AttentionSchedule> getAttentionSchedule(Operation *op) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module || op->getName().getStringRef() != "tessera.flash_attn" ||
      (op->getNumOperands() != 3 && op->getNumOperands() != 4) ||
      op->getNumResults() != 1)
    return failure();
  auto q = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto k = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto v = dyn_cast<RankedTensorType>(op->getOperand(2).getType());
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!q || !k || !v || !output || q.getRank() != 4 || k.getRank() != 4 ||
      v.getRank() != 4 || output.getRank() != 4 || !q.hasStaticShape() ||
      !k.hasStaticShape() || !v.hasStaticShape() || !output.hasStaticShape())
    return failure();

  AttentionSchedule schedule;
  schedule.target = moduleString(module, "tessera.target", "target");
  schedule.arch = moduleString(module, "tessera.arch", "arch");
  bool x86 = schedule.target == "x86" &&
             (schedule.arch.contains("avx512") || schedule.arch.contains("zen5"));
  bool rocm = schedule.target == "rocm" && schedule.arch.contains("gfx1151");
  // Apple GPU consumes the shared rank-4 FORWARD attention launch tile through
  // its status-returning GQA MSL ABI
  // (tessera_apple_gpu_flash_attn_variant_f32_status).  That ABI is f32-only,
  // takes ONE symmetric window extent, and requires head_dim == value_dim;
  // those bounds are enforced below.  Attention BACKWARD has no Apple scheduled
  // consumer and is gated separately.
  bool apple_gpu = schedule.target == "apple_gpu" && schedule.arch == "apple7";
  if (!x86 && !rocm && !apple_gpu)
    return failure();
  schedule.batch = q.getDimSize(0);
  schedule.queryHeads = q.getDimSize(1);
  schedule.queryRows = q.getDimSize(2);
  schedule.headDim = q.getDimSize(3);
  schedule.kvHeads = k.getDimSize(1);
  schedule.keyRows = k.getDimSize(2);
  schedule.valueDim = v.getDimSize(3);
  if (schedule.batch <= 0 || schedule.queryHeads <= 0 ||
      schedule.kvHeads <= 0 || schedule.queryRows <= 0 ||
      schedule.keyRows <= 0 || schedule.headDim <= 0 ||
      schedule.valueDim <= 0 || schedule.queryHeads % schedule.kvHeads != 0 ||
      k.getDimSize(0) != schedule.batch || v.getDimSize(0) != schedule.batch ||
      v.getDimSize(1) != schedule.kvHeads ||
      k.getDimSize(3) != schedule.headDim ||
      v.getDimSize(2) != schedule.keyRows ||
      output.getShape() !=
          ArrayRef<int64_t>({schedule.batch, schedule.queryHeads,
                             schedule.queryRows, schedule.valueDim}) ||
      !output.getElementType().isF32())
    return failure();
  Type qElement = q.getElementType();
  if (k.getElementType() != qElement || v.getElementType() != qElement)
    return failure();
  schedule.storage = storageName(qElement);
  if ((x86 && schedule.storage != "f32") ||
      (rocm && (schedule.storage != "f16" && schedule.storage != "bf16")) ||
      (rocm && (schedule.headDim != schedule.valueDim ||
                schedule.headDim % 16 != 0)) ||
      // The Apple GQA MSL ABI is f32-only, shares one head/value dim, and
      // rejects D > 256 before submission.
      (apple_gpu && (schedule.storage != "f32" ||
                     schedule.headDim != schedule.valueDim ||
                     schedule.headDim > 256)))
    return failure();
  if (op->getNumOperands() == 4) {
    auto bias = dyn_cast<RankedTensorType>(op->getOperand(3).getType());
    if (!bias || !bias.hasStaticShape() || !bias.getElementType().isF32() ||
        bias.getShape() !=
            ArrayRef<int64_t>({schedule.batch, schedule.queryHeads,
                               schedule.queryRows, schedule.keyRows}))
      return failure();
    schedule.bias = true;
  }
  auto scale = op->getAttrOfType<FloatAttr>("scale");
  auto causal = op->getAttrOfType<BoolAttr>("causal");
  auto windowLeft = op->getAttrOfType<IntegerAttr>("window_left");
  auto windowRight = op->getAttrOfType<IntegerAttr>("window_right");
  auto softcap = op->getAttrOfType<FloatAttr>("softcap");
  auto dropoutP = op->getAttrOfType<FloatAttr>("dropout_p");
  auto dropoutSeed = op->getAttrOfType<IntegerAttr>("dropout_seed");
  if (!scale || !scale.getValue().isFinite() ||
      scale.getValueAsDouble() <= 0.0 || !causal || !windowLeft ||
      !windowRight || windowLeft.getInt() < -1 || windowRight.getInt() < -1 ||
      !softcap || !softcap.getValue().isFinite() ||
      softcap.getValueAsDouble() < 0.0 || !dropoutP ||
      !dropoutP.getValue().isFinite() || dropoutP.getValueAsDouble() < 0.0 ||
      dropoutP.getValueAsDouble() >= 1.0 || !dropoutSeed)
    return failure();
  if (x86 && (windowLeft.getInt() != windowRight.getInt() ||
              dropoutP.getValueAsDouble() != 0.0))
    return failure();
  if (rocm && !((windowLeft.getInt() == -1 && windowRight.getInt() == -1) ||
                (causal.getValue() && windowLeft.getInt() >= 0 &&
                 windowRight.getInt() == 0)))
    return failure();
  // Apple's MSL non-causal window is a *symmetric half-window*
  // (window_size/2 per side), which is not the shared window_left/window_right
  // semantics, so a live window would compute a different mask than requested.
  // Bound this slice to the unwindowed contract (causal is still supported) and
  // reject dropout, which the ABI does not carry.  Windowed Apple attention
  // needs its own contract and oracle before admission.
  if (apple_gpu && (windowLeft.getInt() != -1 || windowRight.getInt() != -1 ||
                    dropoutP.getValueAsDouble() != 0.0))
    return failure();
  schedule.scale = static_cast<double>(
      static_cast<float>(scale.getValueAsDouble()));
  schedule.causal = causal.getValue();
  schedule.windowLeft = windowLeft.getInt();
  schedule.windowRight = windowRight.getInt();
  schedule.softcap = static_cast<double>(
      static_cast<float>(softcap.getValueAsDouble()));
  schedule.dropoutP = static_cast<double>(
      static_cast<float>(dropoutP.getValueAsDouble()));
  schedule.dropoutSeed = dropoutSeed.getInt();
  schedule.tileQ = schedule.queryRows;
  schedule.workgroupSize = rocm ? 256 : 1;
  // Apple's attention backward recomputes m/l per query row and its ABI takes
  // no LSE buffer (APPLE-ATTN-STREAM-1), so it owns an explicit recompute
  // policy rather than inheriting x86's save_lse or the gfx1151 threshold.
  schedule.backwardLsePolicy = apple_gpu   ? "apple7_recompute"
                               : x86       ? "save_lse"
                                           : "gfx1151_auto_128";
  schedule.backwardLseSelection =
      apple_gpu ? "recompute"
                : (x86 || schedule.queryRows >= 128) ? "saved" : "recompute";
  schedule.qShape.assign(q.getShape().begin(), q.getShape().end());
  schedule.kShape.assign(k.getShape().begin(), k.getShape().end());
  schedule.vShape.assign(v.getShape().begin(), v.getShape().end());
  schedule.outputShape.assign(output.getShape().begin(), output.getShape().end());
  return schedule;
}

static std::string attentionScheduleDigest(const AttentionSchedule &schedule) {
  std::string contract =
      (Twine("family=attention;target=") + schedule.target +
       ";arch=" + schedule.arch + ";shape=" + Twine(schedule.batch) + "x" +
       Twine(schedule.queryHeads) + "x" + Twine(schedule.kvHeads) + "x" +
       Twine(schedule.queryRows) + "x" + Twine(schedule.keyRows) + "x" +
       Twine(schedule.headDim) + "x" + Twine(schedule.valueDim) +
       ";storage=" + schedule.storage + ";accum=" + schedule.accum +
       ";scale=" + std::to_string(schedule.scale) +
       ";causal=" + Twine(schedule.causal ? 1 : 0) +
       ";bias=" + Twine(schedule.bias ? 1 : 0) +
       ";window=" + Twine(schedule.windowLeft) + ":" +
       Twine(schedule.windowRight) + ";softcap=" +
       std::to_string(schedule.softcap) + ";dropout=" +
       std::to_string(schedule.dropoutP) + ";seed=" +
       Twine(schedule.dropoutSeed) + ";tile=" + Twine(schedule.tileQ) + "x" +
       Twine(schedule.tileKV) + ";workgroup=" +
       Twine(schedule.workgroupSize) + ";recurrence=" + schedule.recurrence +
       ";backward_lse_policy=" + schedule.backwardLsePolicy +
       ";backward_lse_selection=" + schedule.backwardLseSelection)
          .str();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(contract)),
                     /*LowerCase=*/true);
}

struct AttentionBackwardSchedule {
  StringRef target;
  StringRef arch;
  StringRef storage;
  StringRef accum = "f32";
  int64_t batch;
  int64_t queryHeads;
  int64_t kvHeads;
  int64_t queryRows;
  int64_t keyRows;
  int64_t headDim;
  int64_t valueDim;
  double scale;
  bool causal;
  bool bias = false;
  int64_t windowLeft;
  int64_t windowRight;
  double softcap;
  double dropoutP;
  int64_t dropoutSeed;
  int64_t queryBlock;
  int64_t keyBlock;
  int64_t splitCount;
  int64_t workspaceBytes;
  int64_t workgroupSize;
  StringRef recurrence = "tensor_dq_split_dkdv_fixed_reduce_v1";
  StringRef lseCheckpointPolicy;
  StringRef lseCheckpointSelection;
};

static int64_t alignAttentionWorkspace(int64_t value) {
  return ((value + 255) / 256) * 256;
}

static int64_t attentionBackwardWorkspaceBytes(
    const AttentionBackwardSchedule &schedule) {
  int64_t offset = 0;
  auto add = [&](int64_t bytes) {
    offset = alignAttentionWorkspace(offset);
    offset += bytes;
  };
  add(schedule.batch * schedule.queryHeads * schedule.queryRows *
      schedule.valueDim * 4);
  add(schedule.batch * schedule.queryHeads * schedule.queryRows * 4);
  add(schedule.batch * schedule.queryHeads * schedule.queryRows * 4);
  add((schedule.splitCount - 1) * schedule.batch * schedule.kvHeads *
      schedule.keyRows * schedule.headDim * 4);
  add((schedule.splitCount - 1) * schedule.batch * schedule.kvHeads *
      schedule.keyRows * schedule.valueDim * 4);
  return alignAttentionWorkspace(offset);
}

static FailureOr<AttentionBackwardSchedule>
getAttentionBackwardSchedule(Operation *op) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module || op->getName().getStringRef() != "tessera_attn.backward" ||
      op->getNumOperands() != 5 || op->getNumResults() != 3)
    return failure();
  auto dO = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto q = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto k = dyn_cast<RankedTensorType>(op->getOperand(2).getType());
  auto v = dyn_cast<RankedTensorType>(op->getOperand(3).getType());
  auto bias = dyn_cast<RankedTensorType>(op->getOperand(4).getType());
  auto dQ = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto dK = dyn_cast<RankedTensorType>(op->getResult(1).getType());
  auto dV = dyn_cast<RankedTensorType>(op->getResult(2).getType());
  if (!dO || !q || !k || !v || !bias || !dQ || !dK || !dV ||
      !dO.hasStaticShape() || !q.hasStaticShape() || !k.hasStaticShape() ||
      !v.hasStaticShape() || !bias.hasStaticShape() ||
      !dQ.hasStaticShape() || !dK.hasStaticShape() ||
      !dV.hasStaticShape() ||
      dO.getRank() != 4 || q.getRank() != 4 || k.getRank() != 4 ||
      v.getRank() != 4 || dQ.getRank() != 4 || dK.getRank() != 4 ||
      dV.getRank() != 4)
    return failure();

  AttentionBackwardSchedule schedule;
  schedule.target = moduleString(module, "tessera.target", "target");
  schedule.arch = moduleString(module, "tessera.arch", "arch");
  bool x86 = schedule.target == "x86" &&
             (schedule.arch.contains("avx512") || schedule.arch.contains("zen5"));
  bool rocm = schedule.target == "rocm" && schedule.arch.contains("gfx1151");
  // Apple GPU consumes the shared rank-4 BACKWARD launch tile through its
  // status-returning MSL VJP ABI (tessera_apple_gpu_flash_attn_bwd_variant_*
  // _status), which owns dQ / split-dK/dV / fixed-order reduction routes.
  // Same envelope as the Apple forward gate; enforced below.
  bool apple_gpu = schedule.target == "apple_gpu" && schedule.arch == "apple7";
  if (!x86 && !rocm && !apple_gpu)
    return failure();
  schedule.batch = q.getDimSize(0);
  schedule.queryHeads = q.getDimSize(1);
  schedule.queryRows = q.getDimSize(2);
  schedule.headDim = q.getDimSize(3);
  schedule.kvHeads = k.getDimSize(1);
  schedule.keyRows = k.getDimSize(2);
  schedule.valueDim = v.getDimSize(3);
  SmallVector<int64_t, 4> expectedDO{schedule.batch, schedule.queryHeads,
                                     schedule.queryRows, schedule.valueDim};
  if (schedule.batch <= 0 || schedule.queryHeads <= 0 ||
      schedule.kvHeads <= 0 || schedule.queryRows <= 0 ||
      schedule.keyRows <= 0 || schedule.headDim <= 0 ||
      schedule.valueDim <= 0 || schedule.queryHeads % schedule.kvHeads != 0 ||
      k.getDimSize(0) != schedule.batch || v.getDimSize(0) != schedule.batch ||
      v.getDimSize(1) != schedule.kvHeads ||
      k.getDimSize(3) != schedule.headDim ||
      v.getDimSize(2) != schedule.keyRows ||
      dO.getShape() != ArrayRef<int64_t>(expectedDO) ||
      dQ.getShape() != q.getShape() || dK.getShape() != k.getShape() ||
      dV.getShape() != v.getShape() || !dQ.getElementType().isF32() ||
      !dK.getElementType().isF32() || !dV.getElementType().isF32())
    return failure();
  Type storageType = q.getElementType();
  if (dO.getElementType() != storageType || k.getElementType() != storageType ||
      v.getElementType() != storageType)
    return failure();
  schedule.storage = storageName(storageType);
  if ((x86 && schedule.storage != "f32") ||
      (rocm && schedule.storage != "f16" && schedule.storage != "bf16") ||
      (rocm && (schedule.headDim != schedule.valueDim ||
                schedule.headDim % 16 != 0)) ||
      // The Apple VJP ABI shares one head/value dim and rejects D > 256.  It
      // accepts f32/f16/bf16 storage; dQ/dK/dV are always f32, which the
      // shared contract already requires above.
      (apple_gpu && ((schedule.storage != "f32" && schedule.storage != "f16" &&
                      schedule.storage != "bf16") ||
                     schedule.headDim != schedule.valueDim ||
                     schedule.headDim > 256)))
    return failure();
  schedule.bias = bias.getRank() == 4;
  if (schedule.bias) {
    SmallVector<int64_t, 4> expectedBias{
        schedule.batch, schedule.queryHeads, schedule.queryRows,
        schedule.keyRows};
    if (bias.getShape() != ArrayRef<int64_t>(expectedBias))
      return failure();
  }

  auto scale = op->getAttrOfType<FloatAttr>("scale");
  auto causal = op->getAttrOfType<BoolAttr>("causal");
  auto windowLeft = op->getAttrOfType<IntegerAttr>("window_left");
  auto windowRight = op->getAttrOfType<IntegerAttr>("window_right");
  auto softcap = op->getAttrOfType<FloatAttr>("softcap");
  auto dropoutP = op->getAttrOfType<FloatAttr>("dropout_p");
  auto dropoutSeed = op->getAttrOfType<IntegerAttr>("dropout_seed");
  auto queryBlock = op->getAttrOfType<IntegerAttr>("query_block");
  auto keyBlock = op->getAttrOfType<IntegerAttr>("key_block");
  auto splitCount = op->getAttrOfType<IntegerAttr>("split_count");
  if (!scale || !scale.getValue().isFinite() || scale.getValueAsDouble() <= 0.0 ||
      !causal || !windowLeft || !windowRight || windowLeft.getInt() < -1 ||
      windowRight.getInt() < -1 || !softcap || !softcap.getValue().isFinite() ||
      softcap.getValueAsDouble() < 0.0 || !dropoutP ||
      !dropoutP.getValue().isFinite() || dropoutP.getValueAsDouble() < 0.0 ||
      dropoutP.getValueAsDouble() >= 1.0 || !dropoutSeed || !queryBlock ||
      !keyBlock || !splitCount || queryBlock.getInt() <= 0 ||
      keyBlock.getInt() <= 0 || splitCount.getInt() != 2)
    return failure();
  if (x86 && (windowLeft.getInt() != windowRight.getInt() ||
              dropoutP.getValueAsDouble() != 0.0))
    return failure();
  if (rocm && !((windowLeft.getInt() == -1 && windowRight.getInt() == -1) ||
                (causal.getValue() && windowLeft.getInt() >= 0 &&
                 windowRight.getInt() == 0)))
    return failure();
  // Same bound as the Apple forward gate: the MSL non-causal window is a
  // symmetric half-window, not the shared window_left/window_right semantics,
  // so a live window fails closed rather than computing a different mask.
  if (apple_gpu && (windowLeft.getInt() != -1 || windowRight.getInt() != -1 ||
                    dropoutP.getValueAsDouble() != 0.0))
    return failure();
  schedule.scale = static_cast<double>(static_cast<float>(scale.getValueAsDouble()));
  schedule.causal = causal.getValue();
  schedule.windowLeft = windowLeft.getInt();
  schedule.windowRight = windowRight.getInt();
  schedule.softcap = static_cast<double>(static_cast<float>(softcap.getValueAsDouble()));
  schedule.dropoutP = static_cast<double>(static_cast<float>(dropoutP.getValueAsDouble()));
  schedule.dropoutSeed = dropoutSeed.getInt();
  schedule.queryBlock = queryBlock.getInt();
  schedule.keyBlock = keyBlock.getInt();
  schedule.splitCount = splitCount.getInt();
  schedule.workgroupSize = rocm ? 256 : 1;
  // Apple's backward recomputes m/l per query row and its ABI takes no LSE
  // buffer (APPLE-ATTN-STREAM-1), so it declares its own policy rather than
  // inheriting x86's save_lse or the gfx1151 threshold.
  schedule.lseCheckpointPolicy = apple_gpu ? "apple7_recompute"
                                 : x86     ? "save_lse"
                                           : "gfx1151_auto_128";
  schedule.lseCheckpointSelection =
      apple_gpu ? "recompute"
      : (x86 || std::max(schedule.queryRows, schedule.keyRows) >= 128)
          ? "saved"
          : "recompute";
  Operation *function = op->getParentOp();
  auto checkpoint = function
      ? function->getAttrOfType<StringAttr>("tessera.lse_checkpoint")
      : StringAttr();
  if (!checkpoint || checkpoint.getValue() != schedule.lseCheckpointSelection)
    return failure();
  schedule.workspaceBytes = attentionBackwardWorkspaceBytes(schedule);
  return schedule;
}

static std::string
attentionBackwardScheduleDigest(const AttentionBackwardSchedule &schedule) {
  std::string contract =
      (Twine("family=attention_backward;target=") + schedule.target +
       ";arch=" + schedule.arch + ";shape=" + Twine(schedule.batch) + "x" +
       Twine(schedule.queryHeads) + "x" + Twine(schedule.kvHeads) + "x" +
       Twine(schedule.queryRows) + "x" + Twine(schedule.keyRows) + "x" +
       Twine(schedule.headDim) + "x" + Twine(schedule.valueDim) +
       ";storage=" + schedule.storage + ";accum=f32;scale=" +
       std::to_string(schedule.scale) + ";causal=" +
       Twine(schedule.causal ? 1 : 0) + ";bias=" +
       Twine(schedule.bias ? 1 : 0) + ";window=" +
       Twine(schedule.windowLeft) + ":" + Twine(schedule.windowRight) +
       ";softcap=" + std::to_string(schedule.softcap) + ";dropout=" +
       std::to_string(schedule.dropoutP) + ";seed=" +
       Twine(schedule.dropoutSeed) + ";blocks=" +
       Twine(schedule.queryBlock) + "x" + Twine(schedule.keyBlock) +
       ";splits=" + Twine(schedule.splitCount) + ";reduce=0,1;workspace=" +
       Twine(schedule.workspaceBytes) + ";workgroup=" +
       Twine(schedule.workgroupSize) + ";recurrence=" + schedule.recurrence +
       ";lse_policy=" + schedule.lseCheckpointPolicy + ";lse_selection=" +
       schedule.lseCheckpointSelection)
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
           "for bounded static x86-f32, ROCm-f16/f32, and Apple-GPU-f32 Graph "
           "matmul";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<schedule::ScheduleDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod.getContext());

    SmallVector<Operation *> tridiagonalSolves;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.tridiagonal_solve")
        tridiagonalSolves.push_back(op);
    });
    for (Operation *op : tridiagonalSolves) {
      FailureOr<TridiagonalSchedule> selected = getTridiagonalSchedule(op);
      if (failed(selected)) {
        op->emitError(
            "REF-TIER-PHYS-1 requires a static fp32 tridiagonal system on "
            "Zen 5 AVX-512, or a power-of-two N<=256 system on gfx1151");
        return signalPassFailure();
      }
      std::string digest = tridiagonalScheduleDigest(*selected);
      op->setAttr("schedule.artifact_hash", builder.getStringAttr(digest));
      builder.setInsertionPointAfter(op);
      OperationState state(op->getLoc(), "schedule.tridiagonal_solve");
      state.addOperands(op->getResult(0));
      state.addTypes(op->getResult(0).getType());
      state.addAttribute("artifact_hash", builder.getStringAttr(digest));
      state.addAttribute("arch", builder.getStringAttr(selected->arch));
      state.addAttribute("storage", builder.getStringAttr(selected->storage));
      state.addAttribute("accum", builder.getStringAttr(selected->accum));
      state.addAttribute("batch", builder.getI64IntegerAttr(selected->batch));
      state.addAttribute(
          "system_size", builder.getI64IntegerAttr(selected->systemSize));
      state.addAttribute(
          "workgroup_size", builder.getI64IntegerAttr(selected->workgroupSize));
      state.addAttribute("algorithm", builder.getStringAttr(selected->algorithm));
      Operation *scheduled = builder.create(state);
      for (OpOperand &use :
           llvm::make_early_inc_range(op->getResult(0).getUses()))
        if (use.getOwner() != scheduled)
          use.set(scheduled->getResult(0));

      builder.setInsertionPointAfter(scheduled);
      OperationState artifact(op->getLoc(), "schedule.artifact");
      artifact.addAttribute("hash", builder.getStringAttr(digest));
      artifact.addAttribute("arch", builder.getStringAttr(selected->arch));
      artifact.addAttribute(
          "shape_key",
          builder.getStringAttr((Twine("family=tridiagonal_solve;batch=") +
                                 Twine(selected->batch) + ";n=" +
                                 Twine(selected->systemSize))
                                    .str()));
      artifact.addAttribute(
          "tile",
          builder.getDictionaryAttr({builder.getNamedAttr(
              "workgroup_size",
              builder.getI64IntegerAttr(selected->workgroupSize))}));
      artifact.addAttribute(
          "numeric_policy",
          builder.getStringAttr((Twine(selected->storage) + "->" +
                                 selected->accum + ";algorithm=" +
                                 selected->algorithm)
                                    .str()));
      builder.create(artifact);
    }

    SmallVector<Operation *> coalitionButterflies;
    mod.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tessera.game_subset_zeta" ||
          name == "tessera.game_subset_mobius" ||
          name == "tessera.game_superset_zeta" ||
          name == "tessera.game_superset_mobius")
        coalitionButterflies.push_back(op);
    });
    for (Operation *op : coalitionButterflies) {
      FailureOr<CoalitionButterflySchedule> selected =
          getCoalitionButterflySchedule(op);
      if (failed(selected)) {
        op->emitError(
            "REF-TIER-PHYS-1 requires a static power-of-two fp32 coalition "
            "axis on Zen 5 AVX-512, or an axis no larger than the 64 KiB "
            "gfx1151 fp64 LDS envelope");
        return signalPassFailure();
      }
      std::string digest = coalitionButterflyDigest(*selected);
      op->setAttr("schedule.artifact_hash", builder.getStringAttr(digest));
      builder.setInsertionPointAfter(op);
      OperationState state(op->getLoc(), "schedule.coalition_butterfly");
      state.addOperands(op->getResult(0));
      state.addTypes(op->getResult(0).getType());
      state.addAttribute("artifact_hash", builder.getStringAttr(digest));
      state.addAttribute("arch", builder.getStringAttr(selected->arch));
      state.addAttribute("storage", builder.getStringAttr(selected->storage));
      state.addAttribute("accum", builder.getStringAttr(selected->accum));
      state.addAttribute("batch", builder.getI64IntegerAttr(selected->batch));
      state.addAttribute(
          "lattice_size", builder.getI64IntegerAttr(selected->latticeSize));
      state.addAttribute("players",
                         builder.getI64IntegerAttr(selected->players));
      state.addAttribute("half", builder.getI64IntegerAttr(selected->half));
      state.addAttribute("sign", builder.getI64IntegerAttr(selected->sign));
      state.addAttribute(
          "workgroup_size", builder.getI64IntegerAttr(selected->workgroupSize));
      state.addAttribute("stage_order",
                         builder.getStringAttr(selected->stageOrder));
      Operation *scheduled = builder.create(state);
      for (OpOperand &use :
           llvm::make_early_inc_range(op->getResult(0).getUses()))
        if (use.getOwner() != scheduled)
          use.set(scheduled->getResult(0));

      builder.setInsertionPointAfter(scheduled);
      OperationState artifact(op->getLoc(), "schedule.artifact");
      artifact.addAttribute("hash", builder.getStringAttr(digest));
      artifact.addAttribute("arch", builder.getStringAttr(selected->arch));
      artifact.addAttribute(
          "shape_key",
          builder.getStringAttr((Twine("family=coalition_butterfly;batch=") +
                                 Twine(selected->batch) + ";size=" +
                                 Twine(selected->latticeSize) + ";half=" +
                                 Twine(selected->half) + ";sign=" +
                                 Twine(selected->sign))
                                    .str()));
      artifact.addAttribute(
          "tile",
          builder.getDictionaryAttr({builder.getNamedAttr(
              "workgroup_size",
              builder.getI64IntegerAttr(selected->workgroupSize))}));
      artifact.addAttribute(
          "numeric_policy",
          builder.getStringAttr((Twine(selected->storage) + "->" +
                                 selected->accum + ";stage_order=" +
                                 selected->stageOrder)
                                    .str()));
      builder.create(artifact);
    }

    SmallVector<Operation *> matmuls;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.matmul")
        matmuls.push_back(op);
    });
    for (Operation *op : matmuls) {
      FailureOr<MatmulSchedule> selected = getMatmulSchedule(op);
      if (failed(selected)) {
        op->emitError("E2E-REAL-2 Graph->Schedule requires static rank-2 "
                      "x86 f32->f32, ROCm f16->f32, NVIDIA f16/bf16->f32, "
                      "or Apple-GPU f32->f32 "
                      "matmul with no transpose");
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
      state.addAttribute("bias", builder.getBoolAttr(selected->bias));
      state.addAttribute("activation",
                         builder.getStringAttr(selected->activation));
      state.addAttribute("residual", builder.getBoolAttr(selected->residual));
      state.addAttribute("output", builder.getStringAttr(selected->output));
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

    SmallVector<Operation *> ffts;
    mod.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tessera.fft" || name == "tessera.ifft" ||
          name == "tessera.rfft" || name == "tessera.irfft")
        ffts.push_back(op);
    });
    for (Operation *op : ffts) {
      FailureOr<FFTSchedule> selected = getFFTSchedule(op);
      if (failed(selected)) {
        op->emitError(
            "E2E-REAL-FFT Graph->Schedule requires a static f32-pair FFT on "
            "Zen 5 or gfx1151 with an explicit supported spectral contract");
        return signalPassFailure();
      }
      std::string digest = fftScheduleDigest(*selected);
      op->setAttr("schedule.artifact_hash", builder.getStringAttr(digest));
      builder.setInsertionPointAfter(op);
      OperationState state(op->getLoc(), "schedule.fft");
      state.addOperands(op->getResult(0));
      state.addTypes(op->getResult(0).getType());
      state.addAttribute("artifact_hash", builder.getStringAttr(digest));
      state.addAttribute("arch", builder.getStringAttr(selected->arch));
      state.addAttribute("mode", builder.getStringAttr(selected->mode));
      state.addAttribute("axis", builder.getI64IntegerAttr(selected->axis));
      state.addAttribute("length", builder.getI64IntegerAttr(selected->length));
      state.addAttribute("physical_length",
                         builder.getI64IntegerAttr(selected->physicalLength));
      state.addAttribute("batch", builder.getI64IntegerAttr(selected->batch));
      state.addAttribute("real_transform_policy",
                         builder.getStringAttr(selected->realTransformPolicy));
      state.addAttribute("hermitian_layout",
                         builder.getStringAttr(selected->hermitianLayout));
      state.addAttribute("hermitian_weight",
                         builder.getStringAttr(selected->hermitianWeight));
      state.addAttribute("inverse", builder.getBoolAttr(selected->inverse));
      state.addAttribute("normalization",
                         builder.getStringAttr(selected->normalization));
      state.addAttribute("scale", builder.getF32FloatAttr(selected->scale));
      state.addAttribute("storage",
                         builder.getStringAttr("complex64_interleaved_f32"));
      state.addAttribute("accum", builder.getStringAttr("f32"));
      state.addAttribute("radix_policy",
                         builder.getStringAttr(selected->radixPolicy));
      state.addAttribute("strategy", builder.getStringAttr(selected->strategy));
      state.addAttribute("algorithm", builder.getStringAttr(selected->algorithm));
      state.addAttribute("radix_sequence",
                         builder.getDenseI64ArrayAttr(selected->radixSequence));
      state.addAttribute("bluestein_m",
                         builder.getI64IntegerAttr(selected->bluesteinM));
      state.addAttribute("workspace_elems",
                         builder.getI64IntegerAttr(selected->workspaceElems));
      state.addAttribute("workspace_policy",
                         builder.getStringAttr(selected->workspacePolicy));
      state.addAttribute("residency", builder.getStringAttr(selected->residency));
      state.addAttribute("twiddle_layout",
                         builder.getStringAttr("interleaved_f32"));
      state.addAttribute("twiddle_policy",
                         builder.getStringAttr(selected->twiddlePolicy));
      state.addAttribute("deterministic", builder.getBoolAttr(true));
      state.addAttribute("kernel_family",
                         builder.getStringAttr(selected->kernelFamily));
      state.addAttribute("workgroup_size",
                         builder.getI64IntegerAttr(selected->workgroupSize));
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
          builder.getStringAttr(
              (Twine("family=fft;mode=") + selected->mode + ";batch=" +
               Twine(selected->batch) + ";length=" + Twine(selected->length))
                  .str()));
      artifactState.addAttribute(
          "tile", builder.getDictionaryAttr({
                      builder.getNamedAttr(
                          "workgroup_size",
                          builder.getI64IntegerAttr(selected->workgroupSize)),
                      builder.getNamedAttr(
                          "workspace_elems",
                          builder.getI64IntegerAttr(selected->workspaceElems)),
                      builder.getNamedAttr(
                          "algorithm",
                          builder.getStringAttr(selected->algorithm)),
                      builder.getNamedAttr(
                          "workspace_policy",
                          builder.getStringAttr(selected->workspacePolicy)),
                      builder.getNamedAttr(
                          "residency",
                          builder.getStringAttr(selected->residency)),
                      builder.getNamedAttr(
                          "twiddle_policy",
                          builder.getStringAttr(selected->twiddlePolicy)),
                  }));
      artifactState.addAttribute(
          "numeric_policy",
          builder.getStringAttr(
              (Twine("complex64_interleaved_f32->f32;") +
               selected->strategy + ";" + selected->normalization)
                  .str()));
      builder.create(artifactState);
    }

    SmallVector<Operation *> spectralBackwards;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.spectral_backward")
        spectralBackwards.push_back(op);
    });
    for (Operation *op : spectralBackwards) {
      ModuleOp module = op->getParentOfType<ModuleOp>();
      StringRef configuredTarget = moduleString(module, "tessera.target", "target");
      StringRef configuredArch = moduleString(module, "tessera.arch", "arch");
      const bool x86 = configuredTarget == "x86" ||
                       configuredArch.contains("avx512") ||
                       configuredArch.contains("zen5");
      const bool rocm = configuredArch.contains("gfx1151");
      if (!x86 && !rocm) {
        op->emitError(
            "AD-TSOL-SPECTRAL-1 compound adjoints require an exact Zen 5 or gfx1151 profile");
        return signalPassFailure();
      }
      for (Type type : op->getOperandTypes()) {
        auto tensor = dyn_cast<RankedTensorType>(type);
        if (!tensor || !tensor.hasStaticShape()) {
          op->emitError("compound spectral adjoint scheduling requires static tensors");
          return signalPassFailure();
        }
      }
      op->setAttr("target", builder.getStringAttr(x86 ? "x86" : "rocm"));
      op->setAttr("arch", builder.getStringAttr(x86 ? "zen5-avx512" : "gfx1151"));
      op->setAttr("mutation_lineage",
                  builder.getStringAttr("inputs_immutable_outputs_fresh_v1"));
      if (!op->hasAttr("normalization"))
        op->setAttr("normalization", builder.getStringAttr("backward"));
      if (!op->hasAttr("spectrum_layout"))
        op->setAttr("spectrum_layout",
                    builder.getStringAttr(
                        op->getAttrOfType<StringAttr>("kind").getValue() ==
                                "tessera.spectral_filter"
                            ? "full_complex"
                            : "half_spectrum_nyquist_explicit"));
      if (!op->hasAttr("center"))
        op->setAttr("center", builder.getBoolAttr(false));
      if (!op->hasAttr("onesided"))
        op->setAttr("onesided", builder.getBoolAttr(true));
      if (!op->hasAttr("pad_mode"))
        op->setAttr("pad_mode", builder.getStringAttr("constant"));
      op->setAttr(
          "input_signature",
          builder.getStringAttr(spectralBackwardTypeSignature(op->getOperands())));
      op->setAttr(
          "output_signature",
          builder.getStringAttr(spectralBackwardTypeSignature(op->getResults())));
      FailureOr<std::string> digest = spectralBackwardDigest(op);
      if (failed(digest)) {
        op->emitError("compound spectral adjoint has an incomplete semantic contract");
        return signalPassFailure();
      }
      op->setAttr("schedule.artifact_hash", builder.getStringAttr(*digest));
      builder.setInsertionPointAfter(op);
      OperationState state(op->getLoc(), "schedule.spectral_backward");
      state.addOperands(op->getResults());
      state.addTypes(op->getResultTypes());
      state.addAttribute("artifact_hash", builder.getStringAttr(*digest));
      for (StringRef name : {"target", "arch", "kind", "axis",
                             "logical_length", "normalization",
                             "spectrum_layout", "hop", "center", "onesided",
                             "pad_mode", "output_length",
                             "input_signature", "output_signature",
                             "mutation_lineage"})
        if (Attribute attr = op->getAttr(name)) state.addAttribute(name, attr);
      Operation *scheduled = builder.create(state);
      for (auto [original, replacement] :
           llvm::zip(op->getResults(), scheduled->getResults()))
        for (OpOperand &use : llvm::make_early_inc_range(original.getUses()))
          if (use.getOwner() != scheduled) use.set(replacement);

      builder.setInsertionPointAfter(scheduled);
      OperationState artifactState(op->getLoc(), "schedule.artifact");
      artifactState.addAttribute("hash", builder.getStringAttr(*digest));
      artifactState.addAttribute("arch", op->getAttr("arch"));
      artifactState.addAttribute(
          "shape_key",
          builder.getStringAttr((Twine("family=spectral_backward;kind=") +
                                 op->getAttrOfType<StringAttr>("kind").getValue())
                                    .str()));
      artifactState.addAttribute(
          "tile", builder.getDictionaryAttr({builder.getNamedAttr(
                      "output_count",
                      builder.getI64IntegerAttr(op->getNumResults()))}));
      artifactState.addAttribute(
          "numeric_policy",
          builder.getStringAttr((Twine("adjoint;") +
                                 op->getAttrOfType<StringAttr>("normalization")
                                     .getValue())
                                    .str()));
      builder.create(artifactState);
    }

    SmallVector<Operation *> esCorrections;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.es_low_rank_correction")
        esCorrections.push_back(op);
    });
    for (Operation *op : esCorrections) {
      ModuleOp module = op->getParentOfType<ModuleOp>();
      StringRef configuredArch = moduleString(module, "tessera.arch", "arch");
      if (configuredArch != "gfx1151" && configuredArch != "zen5-avx512") {
        op->emitError("EGGROLL W2 is fail-closed outside exact gfx1151 and Zen 5 AVX-512 profiles");
        return signalPassFailure();
      }
      FailureOr<std::pair<std::string, std::string>> contract =
          esLowRankContract(op, configuredArch);
      if (failed(contract)) {
        op->emitError("EGGROLL W2 requires static f32 rank-1 Graph operands");
        return signalPassFailure();
      }
      auto x = cast<RankedTensorType>(op->getOperand(0).getType());
      int64_t rows = 1;
      for (int64_t dim : x.getShape().drop_front().drop_back()) rows *= dim;
      auto integer = [&](StringRef name, int64_t fallback) {
        auto attr = op->getAttrOfType<IntegerAttr>(name);
        return attr ? attr.getInt() : fallback;
      };
      auto boolean = [&](StringRef name, bool fallback) {
        auto attr = op->getAttrOfType<BoolAttr>(name);
        return attr ? attr.getValue() : fallback;
      };
      op->setAttr("schedule.artifact_hash",
                  builder.getStringAttr(contract->second));
      builder.setInsertionPointAfter(op);
      OperationState state(op->getLoc(), "schedule.es_low_rank_correction");
      state.addOperands(op->getResult(0));
      state.addTypes(op->getResult(0).getType());
      state.addAttribute("artifact_hash", builder.getStringAttr(contract->second));
      state.addAttribute("lineage_payload", builder.getStringAttr(contract->first));
      state.addAttribute("arch", builder.getStringAttr(configuredArch));
      state.addAttribute("population", builder.getI64IntegerAttr(x.getDimSize(0)));
      state.addAttribute("rows_per_member", builder.getI64IntegerAttr(rows));
      state.addAttribute("in_dim", builder.getI64IntegerAttr(x.getShape().back()));
      state.addAttribute("out_dim",
                         builder.getI64IntegerAttr(integer("out_dim", -1)));
      state.addAttribute("rank", builder.getI64IntegerAttr(integer("rank", -1)));
      state.addAttribute("epoch", builder.getI64IntegerAttr(integer("epoch", 0)));
      state.addAttribute("sigma", builder.getF32FloatAttr(
          static_cast<float>(op->getAttrOfType<FloatAttr>("sigma")
                                 .getValueAsDouble())));
      state.addAttribute("antithetic",
                         builder.getBoolAttr(boolean("antithetic", true)));
      state.addAttribute("score", builder.getStringAttr("gaussian"));
      state.addAttribute("rng_algorithm", builder.getStringAttr(
          "splitmix64-philox4x32-boxmuller"));
      state.addAttribute("rng_version", builder.getI64IntegerAttr(1));
      state.addAttribute("storage", builder.getStringAttr("f32"));
      state.addAttribute("accum", builder.getStringAttr("f32"));
      state.addAttribute("workgroup_size", builder.getI64IntegerAttr(
          configuredArch == "gfx1151" ? 256 : 1));
      Operation *scheduled = builder.create(state);
      for (OpOperand &use :
           llvm::make_early_inc_range(op->getResult(0).getUses()))
        if (use.getOwner() != scheduled) use.set(scheduled->getResult(0));

      builder.setInsertionPointAfter(scheduled);
      OperationState artifactState(op->getLoc(), "schedule.artifact");
      artifactState.addAttribute("hash", builder.getStringAttr(contract->second));
      artifactState.addAttribute("arch", builder.getStringAttr(configuredArch));
      artifactState.addAttribute(
          "shape_key", builder.getStringAttr(
              (Twine("family=es_low_rank_correction;p=") +
               Twine(x.getDimSize(0)) + ";in=" + Twine(x.getShape().back()) +
               ";out=" + Twine(integer("out_dim", -1)) + ";rank=1")
                  .str()));
      artifactState.addAttribute(
          "tile", builder.getDictionaryAttr({
                      builder.getNamedAttr(
                          "workgroup_size",
                          builder.getI64IntegerAttr(
                              configuredArch == "gfx1151" ? 256 : 1)),
                      builder.getNamedAttr("rng_version",
                                           builder.getI64IntegerAttr(1)),
                  }));
      artifactState.addAttribute(
          "numeric_policy", builder.getStringAttr("f32->f32;rank1"));
      builder.create(artifactState);
    }

    SmallVector<Operation *> depthAttentions;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.depth_attn")
        depthAttentions.push_back(op);
    });
    for (Operation *op : depthAttentions) {
      FailureOr<DepthAttentionSchedule> selected =
          getDepthAttentionSchedule(op);
      if (failed(selected)) {
        op->emitError(
            "BLOCK-ATTNRES-1 Graph->Schedule requires a static all-f32 "
            "depth-attention contract on gfx1151 or Zen 5 AVX-512");
        return signalPassFailure();
      }
      std::string digest = depthAttentionScheduleDigest(*selected);
      op->setAttr("schedule.artifact_hash", builder.getStringAttr(digest));
      builder.setInsertionPointAfter(op);
      auto scheduledOp = builder.create<schedule::DepthAttentionOp>(
          op->getLoc(), op->getResult(0).getType(), op->getResult(0),
          builder.getStringAttr(digest), builder.getStringAttr(selected->arch),
          builder.getStringAttr(selected->storage),
          builder.getStringAttr(selected->softmax),
          builder.getStringAttr(selected->accum),
          builder.getF64FloatAttr(selected->eps),
          builder.getI64IntegerAttr(selected->sourceCount),
          builder.getI64IntegerAttr(selected->rows),
          builder.getI64IntegerAttr(selected->width),
          builder.getI64IntegerAttr(selected->sourceTile),
          builder.getI64IntegerAttr(selected->workgroupSize),
          builder.getStringAttr(selected->statisticsRecurrence),
          builder.getStringAttr(selected->mergeRecurrence),
          builder.getBoolAttr(selected->reassociable));
      Operation *scheduled = scheduledOp.getOperation();
      for (OpOperand &use :
           llvm::make_early_inc_range(op->getResult(0).getUses()))
        if (use.getOwner() != scheduled)
          use.set(scheduled->getResult(0));

      builder.setInsertionPointAfter(scheduled);
      OperationState artifactState(op->getLoc(), "schedule.artifact");
      artifactState.addAttribute("hash", builder.getStringAttr(digest));
      artifactState.addAttribute("arch", builder.getStringAttr(selected->arch));
      artifactState.addAttribute(
          "shape_key",
          builder.getStringAttr(
              (Twine("family=depth_attention;sources=") +
               Twine(selected->sourceCount) + ";rows=" +
               Twine(selected->rows) + ";width=" + Twine(selected->width) +
               ";storage=" + selected->storage)
                  .str()));
      artifactState.addAttribute(
          "tile", builder.getDictionaryAttr({
                      builder.getNamedAttr(
                          "source_tile",
                          builder.getI64IntegerAttr(selected->sourceTile)),
                      builder.getNamedAttr(
                          "workgroup_size",
                          builder.getI64IntegerAttr(selected->workgroupSize)),
                  }));
      artifactState.addAttribute(
          "numeric_policy",
          builder.getStringAttr(
              (Twine(selected->storage) + "->" + selected->softmax + "->" +
               selected->accum + ";eps=" + std::to_string(selected->eps) +
               ";merge=" + selected->mergeRecurrence)
                  .str()));
      builder.create(artifactState);
    }

    SmallVector<Operation *> attentions;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.flash_attn")
        attentions.push_back(op);
    });
    for (Operation *op : attentions) {
      FailureOr<AttentionSchedule> selected = getAttentionSchedule(op);
      if (failed(selected)) {
        op->emitError(
            "E2E-REAL-5A Graph->Schedule requires a supported static rank-4 "
            "Zen 5 f32 or gfx1151 f16/bf16 attention contract");
        return signalPassFailure();
      }
      std::string digest = attentionScheduleDigest(*selected);
      op->setAttr("schedule.artifact_hash", builder.getStringAttr(digest));
      builder.setInsertionPointAfter(op);
      auto scheduledOp = builder.create<schedule::AttentionOp>(
          op->getLoc(), op->getResult(0).getType(), op->getResult(0),
          builder.getStringAttr(digest), builder.getStringAttr(selected->arch),
          builder.getStringAttr(selected->storage),
          builder.getStringAttr(selected->accum),
          builder.getF32FloatAttr(selected->scale),
          builder.getBoolAttr(selected->causal),
          builder.getBoolAttr(selected->bias),
          builder.getI64IntegerAttr(selected->windowLeft),
          builder.getI64IntegerAttr(selected->windowRight),
          builder.getF32FloatAttr(selected->softcap),
          builder.getF32FloatAttr(selected->dropoutP),
          builder.getI64IntegerAttr(selected->dropoutSeed),
          builder.getI64IntegerAttr(selected->tileQ),
          builder.getI64IntegerAttr(selected->tileKV),
          builder.getI64IntegerAttr(selected->workgroupSize),
          builder.getStringAttr(selected->recurrence),
          builder.getStringAttr(selected->backwardLsePolicy),
          builder.getStringAttr(selected->backwardLseSelection));
      Operation *scheduled = scheduledOp.getOperation();
      for (OpOperand &use : llvm::make_early_inc_range(op->getResult(0).getUses()))
        if (use.getOwner() != scheduled)
          use.set(scheduled->getResult(0));

      builder.setInsertionPointAfter(scheduled);
      OperationState artifactState(op->getLoc(), "schedule.artifact");
      artifactState.addAttribute("hash", builder.getStringAttr(digest));
      artifactState.addAttribute("arch", builder.getStringAttr(selected->arch));
      artifactState.addAttribute(
          "shape_key",
          builder.getStringAttr(
              (Twine("family=attention;B=") + Twine(selected->batch) +
               ";Hq=" + Twine(selected->queryHeads) +
               ";Hkv=" + Twine(selected->kvHeads) +
               ";Sq=" + Twine(selected->queryRows) +
               ";Sk=" + Twine(selected->keyRows) +
               ";D=" + Twine(selected->headDim) +
               ";Dv=" + Twine(selected->valueDim) +
               ";storage=" + selected->storage)
                  .str()));
      artifactState.addAttribute(
          "tile", builder.getDictionaryAttr({
                      builder.getNamedAttr(
                          "tile_q", builder.getI64IntegerAttr(selected->tileQ)),
                      builder.getNamedAttr(
                          "tile_kv", builder.getI64IntegerAttr(selected->tileKV)),
                      builder.getNamedAttr(
                          "workgroup_size",
                          builder.getI64IntegerAttr(selected->workgroupSize)),
                  }));
      artifactState.addAttribute(
          "numeric_policy",
          builder.getStringAttr(
              (Twine(selected->storage) + "->" + selected->accum +
               ";backward_lse=" + selected->backwardLsePolicy + ":" +
               selected->backwardLseSelection)
                  .str()));
      builder.create(artifactState);
    }

    SmallVector<Operation *> attentionBackwards;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_attn.backward")
        attentionBackwards.push_back(op);
    });
    for (Operation *op : attentionBackwards) {
      FailureOr<AttentionBackwardSchedule> selected =
          getAttentionBackwardSchedule(op);
      if (failed(selected)) {
        op->emitError(
            "E2E-REAL-5B Graph->Schedule requires the canonical static "
            "tensor-valued Zen 5 f32 or gfx1151 f16/bf16 attention VJP");
        return signalPassFailure();
      }
      std::string digest = attentionBackwardScheduleDigest(*selected);
      op->setAttr("schedule.artifact_hash", builder.getStringAttr(digest));
      builder.setInsertionPointAfter(op);
      OperationState state(op->getLoc(), "schedule.attention_backward");
      state.addOperands(op->getResults());
      state.addTypes(op->getResultTypes());
      state.addAttribute("artifact_hash", builder.getStringAttr(digest));
      state.addAttribute("arch", builder.getStringAttr(selected->arch));
      state.addAttribute("storage", builder.getStringAttr(selected->storage));
      state.addAttribute("accum", builder.getStringAttr(selected->accum));
      state.addAttribute("scale", builder.getF32FloatAttr(selected->scale));
      state.addAttribute("causal", builder.getBoolAttr(selected->causal));
      state.addAttribute("bias", builder.getBoolAttr(selected->bias));
      state.addAttribute("window_left",
                         builder.getI64IntegerAttr(selected->windowLeft));
      state.addAttribute("window_right",
                         builder.getI64IntegerAttr(selected->windowRight));
      state.addAttribute("softcap", builder.getF32FloatAttr(selected->softcap));
      state.addAttribute("dropout_p",
                         builder.getF32FloatAttr(selected->dropoutP));
      state.addAttribute("dropout_seed",
                         builder.getI64IntegerAttr(selected->dropoutSeed));
      state.addAttribute("query_block",
                         builder.getI64IntegerAttr(selected->queryBlock));
      state.addAttribute("key_block",
                         builder.getI64IntegerAttr(selected->keyBlock));
      state.addAttribute("split_count",
                         builder.getI64IntegerAttr(selected->splitCount));
      state.addAttribute("reduction_order",
                         builder.getDenseI64ArrayAttr({0, 1}));
      state.addAttribute("workspace_bytes",
                         builder.getI64IntegerAttr(selected->workspaceBytes));
      state.addAttribute("workgroup_size",
                         builder.getI64IntegerAttr(selected->workgroupSize));
      state.addAttribute("recurrence",
                         builder.getStringAttr(selected->recurrence));
      state.addAttribute(
          "lse_checkpoint_policy",
          builder.getStringAttr(selected->lseCheckpointPolicy));
      state.addAttribute(
          "lse_checkpoint_selection",
          builder.getStringAttr(selected->lseCheckpointSelection));
      Operation *scheduled = builder.create(state);
      for (auto [source, replacement] :
           llvm::zip_equal(op->getResults(), scheduled->getResults()))
        for (OpOperand &use : llvm::make_early_inc_range(source.getUses()))
          if (use.getOwner() != scheduled)
            use.set(replacement);

      builder.setInsertionPointAfter(scheduled);
      OperationState artifactState(op->getLoc(), "schedule.artifact");
      artifactState.addAttribute("hash", builder.getStringAttr(digest));
      artifactState.addAttribute("arch", builder.getStringAttr(selected->arch));
      artifactState.addAttribute(
          "shape_key",
          builder.getStringAttr(
              (Twine("family=attention_backward;B=") +
               Twine(selected->batch) + ";Hq=" +
               Twine(selected->queryHeads) + ";Hkv=" +
               Twine(selected->kvHeads) + ";Sq=" +
               Twine(selected->queryRows) + ";Sk=" +
               Twine(selected->keyRows) + ";D=" +
               Twine(selected->headDim) + ";Dv=" +
               Twine(selected->valueDim) + ";storage=" + selected->storage)
                  .str()));
      artifactState.addAttribute(
          "tile", builder.getDictionaryAttr({
                      builder.getNamedAttr(
                          "query_block",
                          builder.getI64IntegerAttr(selected->queryBlock)),
                      builder.getNamedAttr(
                          "key_block",
                          builder.getI64IntegerAttr(selected->keyBlock)),
                      builder.getNamedAttr(
                          "split_count",
                          builder.getI64IntegerAttr(selected->splitCount)),
                      builder.getNamedAttr(
                          "workspace_bytes",
                          builder.getI64IntegerAttr(selected->workspaceBytes)),
                      builder.getNamedAttr(
                          "workgroup_size",
                          builder.getI64IntegerAttr(selected->workgroupSize)),
                  }));
      artifactState.addAttribute(
          "numeric_policy",
          builder.getStringAttr(
              (Twine(selected->storage) + "->f32;lse=" +
               selected->lseCheckpointPolicy + ":" +
               selected->lseCheckpointSelection)
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
    return "Revalidate registered content-addressed Schedule decisions against "
           "retained Graph producers and emit launch-level Tile carriers";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, LLVM::LLVMDialect, memref::MemRefDialect,
                    NVVM::NVVMDialect, scf::SCFDialect,
                    schedule::ScheduleDialect>();
    tile::registerTileDialect(registry);
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod.getContext());

    SmallVector<Operation *> scheduledTridiagonalSolves;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "schedule.tridiagonal_solve")
        scheduledTridiagonalSolves.push_back(op);
    });
    for (Operation *scheduled : scheduledTridiagonalSolves) {
      Operation *graph = scheduled->getOperand(0).getDefiningOp();
      FailureOr<TridiagonalSchedule> selected =
          graph ? getTridiagonalSchedule(graph)
                : FailureOr<TridiagonalSchedule>(failure());
      auto hash = scheduled->getAttrOfType<StringAttr>("artifact_hash");
      auto arch = scheduled->getAttrOfType<StringAttr>("arch");
      auto storage = scheduled->getAttrOfType<StringAttr>("storage");
      auto accum = scheduled->getAttrOfType<StringAttr>("accum");
      auto batch = scheduled->getAttrOfType<IntegerAttr>("batch");
      auto systemSize = scheduled->getAttrOfType<IntegerAttr>("system_size");
      auto workgroup = scheduled->getAttrOfType<IntegerAttr>("workgroup_size");
      auto algorithm = scheduled->getAttrOfType<StringAttr>("algorithm");
      if (!graph || failed(selected) || !hash || !arch || !storage || !accum ||
          !batch || !systemSize || !workgroup || !algorithm ||
          tridiagonalScheduleDigest(*selected) != hash.getValue() ||
          arch.getValue() != selected->arch ||
          storage.getValue() != selected->storage ||
          accum.getValue() != selected->accum ||
          batch.getInt() != selected->batch ||
          systemSize.getInt() != selected->systemSize ||
          workgroup.getInt() != selected->workgroupSize ||
          algorithm.getValue() != selected->algorithm) {
        scheduled->emitError(
            "tridiagonal Schedule contract was altered or lost its Graph subject");
        return signalPassFailure();
      }
      auto graphHash =
          graph->getAttrOfType<StringAttr>("schedule.artifact_hash");
      SmallVector<schedule::ArtifactOp> matchingArtifacts;
      mod.walk([&](schedule::ArtifactOp artifact) {
        if (artifact.getHash() == hash.getValue())
          matchingArtifacts.push_back(artifact);
      });
      if (!graphHash || graphHash.getValue() != hash.getValue() ||
          matchingArtifacts.size() != 1) {
        scheduled->emitError(
            "requires exactly one matching Graph hash and schedule.artifact");
        return signalPassFailure();
      }

      Location loc = scheduled->getLoc();
      builder.setInsertionPoint(scheduled);
      auto pointerType = LLVM::LLVMPointerType::get(&getContext());
      auto toPointer = [&](Value tensor) -> Value {
        auto type = cast<RankedTensorType>(tensor.getType());
        auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
        Value buffer =
            builder.create<bufferization::ToBufferOp>(loc, memrefType, tensor);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), index);
        return builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer);
      };
      SmallVector<Value> operands;
      for (Value value : graph->getOperands())
        operands.push_back(toPointer(value));
      auto outputType = cast<RankedTensorType>(graph->getResult(0).getType());
      auto outputMemref =
          MemRefType::get(outputType.getShape(), outputType.getElementType());
      Value outputBuffer = builder.create<memref::AllocOp>(loc, outputMemref);
      Value outputIndex =
          builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outputBuffer);
      Value outputInteger = builder.create<arith::IndexCastOp>(
          loc, builder.getI64Type(), outputIndex);
      operands.push_back(
          builder.create<LLVM::IntToPtrOp>(loc, pointerType, outputInteger));
      operands.push_back(builder.create<arith::ConstantIntOp>(
          loc, selected->batch, 64));
      operands.push_back(builder.create<arith::ConstantIntOp>(
          loc, selected->systemSize, 64));
      OperationState kernel(loc, "tile.tridiagonal_solve_kernel");
      kernel.addOperands(operands);
      kernel.addAttribute("arch", arch);
      kernel.addAttribute("storage", storage);
      kernel.addAttribute("accum", accum);
      kernel.addAttribute("algorithm", algorithm);
      kernel.addAttribute("workgroup_size", workgroup);
      kernel.addAttribute("tessera.schedule_hash", hash);
      builder.create(kernel);
      Value result = builder.create<bufferization::ToTensorOp>(
          loc, outputType, outputBuffer);
      scheduled->getResult(0).replaceAllUsesWith(result);
      scheduled->erase();
      if (graph->use_empty())
        graph->erase();
      for (schedule::ArtifactOp artifact : matchingArtifacts)
        artifact.erase();
    }

    SmallVector<Operation *> scheduledButterflies;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "schedule.coalition_butterfly")
        scheduledButterflies.push_back(op);
    });
    for (Operation *scheduled : scheduledButterflies) {
      Operation *graph = scheduled->getOperand(0).getDefiningOp();
      FailureOr<CoalitionButterflySchedule> selected =
          graph ? getCoalitionButterflySchedule(graph)
                : FailureOr<CoalitionButterflySchedule>(failure());
      auto hash = scheduled->getAttrOfType<StringAttr>("artifact_hash");
      auto arch = scheduled->getAttrOfType<StringAttr>("arch");
      auto storage = scheduled->getAttrOfType<StringAttr>("storage");
      auto accum = scheduled->getAttrOfType<StringAttr>("accum");
      auto batch = scheduled->getAttrOfType<IntegerAttr>("batch");
      auto latticeSize = scheduled->getAttrOfType<IntegerAttr>("lattice_size");
      auto players = scheduled->getAttrOfType<IntegerAttr>("players");
      auto half = scheduled->getAttrOfType<IntegerAttr>("half");
      auto sign = scheduled->getAttrOfType<IntegerAttr>("sign");
      auto workgroup = scheduled->getAttrOfType<IntegerAttr>("workgroup_size");
      auto stageOrder = scheduled->getAttrOfType<StringAttr>("stage_order");
      if (!graph || failed(selected) || !hash || !arch || !storage || !accum ||
          !batch || !latticeSize || !players || !half || !sign || !workgroup ||
          !stageOrder || coalitionButterflyDigest(*selected) != hash.getValue() ||
          arch.getValue() != selected->arch ||
          storage.getValue() != selected->storage ||
          accum.getValue() != selected->accum || batch.getInt() != selected->batch ||
          latticeSize.getInt() != selected->latticeSize ||
          players.getInt() != selected->players || half.getInt() != selected->half ||
          sign.getInt() != selected->sign ||
          workgroup.getInt() != selected->workgroupSize ||
          stageOrder.getValue() != selected->stageOrder) {
        scheduled->emitError(
            "coalition butterfly Schedule contract was altered or lost its Graph subject");
        return signalPassFailure();
      }
      auto graphHash =
          graph->getAttrOfType<StringAttr>("schedule.artifact_hash");
      SmallVector<schedule::ArtifactOp> matchingArtifacts;
      mod.walk([&](schedule::ArtifactOp artifact) {
        if (artifact.getHash() == hash.getValue())
          matchingArtifacts.push_back(artifact);
      });
      if (!graphHash || graphHash.getValue() != hash.getValue() ||
          matchingArtifacts.size() != 1) {
        scheduled->emitError(
            "requires exactly one matching Graph hash and schedule.artifact");
        return signalPassFailure();
      }
      Location loc = scheduled->getLoc();
      builder.setInsertionPoint(scheduled);
      auto pointerType = LLVM::LLVMPointerType::get(&getContext());
      auto inputType = cast<RankedTensorType>(graph->getOperand(0).getType());
      auto inputMemref =
          MemRefType::get(inputType.getShape(), inputType.getElementType());
      Value inputBuffer = builder.create<bufferization::ToBufferOp>(
          loc, inputMemref, graph->getOperand(0));
      Value inputIndex =
          builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, inputBuffer);
      Value inputInteger = builder.create<arith::IndexCastOp>(
          loc, builder.getI64Type(), inputIndex);
      Value inputPointer =
          builder.create<LLVM::IntToPtrOp>(loc, pointerType, inputInteger);
      auto outputType = cast<RankedTensorType>(graph->getResult(0).getType());
      auto outputMemref =
          MemRefType::get(outputType.getShape(), outputType.getElementType());
      Value outputBuffer = builder.create<memref::AllocOp>(loc, outputMemref);
      Value outputIndex =
          builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outputBuffer);
      Value outputInteger = builder.create<arith::IndexCastOp>(
          loc, builder.getI64Type(), outputIndex);
      Value outputPointer =
          builder.create<LLVM::IntToPtrOp>(loc, pointerType, outputInteger);
      Value batchValue = builder.create<arith::ConstantIntOp>(
          loc, selected->batch, 64);
      Value sizeValue = builder.create<arith::ConstantIntOp>(
          loc, selected->latticeSize, 64);
      OperationState kernel(loc, "tile.coalition_butterfly_kernel");
      kernel.addOperands(
          {inputPointer, outputPointer, batchValue, sizeValue});
      kernel.addAttribute("arch", arch);
      kernel.addAttribute("storage", storage);
      kernel.addAttribute("accum", accum);
      kernel.addAttribute("players", players);
      kernel.addAttribute("half", half);
      kernel.addAttribute("sign", sign);
      kernel.addAttribute("workgroup_size", workgroup);
      kernel.addAttribute("stage_order", stageOrder);
      kernel.addAttribute("tessera.schedule_hash", hash);
      builder.create(kernel);
      Value result = builder.create<bufferization::ToTensorOp>(
          loc, outputType, outputBuffer);
      scheduled->getResult(0).replaceAllUsesWith(result);
      scheduled->erase();
      if (graph->use_empty())
        graph->erase();
      for (schedule::ArtifactOp artifact : matchingArtifacts)
        artifact.erase();
    }

    SmallVector<schedule::MatmulOp> scheduledMatmuls;
    mod.walk([&](schedule::MatmulOp op) { scheduledMatmuls.push_back(op); });
    // A Graph-facing tensor function is not a CUDA launch ABI.  The canonical
    // SM120 consumer therefore emits one sibling LLVM kernel with the proven
    // A/B/D/M/N/K ABI, then removes the now-consumed tensor wrapper once all
    // schedule records have been checked.  Other backends retain their
    // existing in-function carriers.
    SmallVector<func::FuncOp> consumedNvidiaGraphFunctions;
    for (schedule::MatmulOp scheduled : scheduledMatmuls) {
      Operation *graph = scheduled.getSubject().getDefiningOp();
      if (!graph || graph->getName().getStringRef() != "tessera.matmul" ||
          graph->getNumOperands() < 2 || graph->getNumOperands() > 4 ||
          graph->getNumResults() != 1) {
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
          scheduled.getBias() != selected->bias ||
          scheduled.getActivation() != selected->activation ||
          scheduled.getResidual() != selected->residual ||
          scheduled.getOutput() != selected->output ||
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

      Location loc = scheduled.getLoc();

      if (selected->target == "nvidia_sm120") {
        // The shared scheduling library intentionally has no link-time
        // dependency on an optional target backend.  In a CUDA-enabled tool,
        // load the registered target dialect by namespace before constructing
        // its block-coordinate boundary operation.
        if (!getContext().getOrLoadDialect("tessera_nvidia")) {
          scheduled.emitError(
              "SM120 scheduled matmul requires the registered NVIDIA Target IR dialect");
          return signalPassFailure();
        }
        auto graphFunction = scheduled->getParentOfType<func::FuncOp>();
        if (!graphFunction) {
          scheduled.emitError("SM120 scheduled matmul requires a Graph func wrapper");
          return signalPassFailure();
        }
        // The explicit scheduled route uses the conservative crossover from
        // the retained SuperBear pruning packet: cases below 67.1M FLOPs are
        // tied, regress, or fail the low-variance gate; all measured 67.1M+
        // cases are low-variance wins. This is not a global selector promotion
        // (WSL cannot supply that authority); it prevents the canonical route
        // from choosing an unproven crossover.
        const __int128 sm120Work = static_cast<__int128>(2) * selected->m *
                                   selected->n * selected->k;
        const bool dynamicSm120 = selected->dynamicM || selected->dynamicN ||
                                  selected->dynamicK;
        const bool macroSm120Producer =
            selected->m >= 32 && selected->n >= 32 &&
            selected->k >= 16 &&
            (selected->storage == "f16" || selected->storage == "bf16") &&
            selected->accum == "f32" &&
            sm120Work >= 67108864;
        const bool fusedEpilogue = selected->bias || selected->residual ||
                                   selected->activation != "none";
        std::string epilogueSuffix;
        if (fusedEpilogue || selected->output == "f16")
          epilogueSuffix =
              (Twine("_fused_") + selected->storage + "_" +
               selected->activation + "_b" + Twine(selected->bias) + "_r" +
               Twine(selected->residual) +
               (selected->output == "f16" ? "_outf16" : ""))
                  .str();
        std::string kernelName =
            (graphFunction.getName() + epilogueSuffix +
             (macroSm120Producer ? "_macro_kernel" : "_kernel"))
                .str();
        if (SymbolTable::lookupSymbolIn(mod, kernelName)) {
          scheduled.emitError("SM120 scheduled matmul kernel symbol already exists");
          return signalPassFailure();
        }

        auto pointerType = LLVM::LLVMPointerType::get(&getContext());
        auto i64 = builder.getI64Type();
        SmallVector<Type> kernelInputs{pointerType, pointerType};
        if (selected->bias)
          kernelInputs.push_back(pointerType);
        if (selected->residual)
          kernelInputs.push_back(pointerType);
        kernelInputs.append({pointerType, i64, i64, i64});
        if (dynamicSm120)
          kernelInputs.append({i64, i64, i64});
        auto kernelType = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(&getContext()), kernelInputs, false);
        OpBuilder moduleBuilder(mod.getBody(), mod.getBody()->end());
        auto kernelFunction =
            LLVM::LLVMFuncOp::create(moduleBuilder, loc, kernelName, kernelType);
        kernelFunction->setAttr("nvvm.kernel", UnitAttr::get(&getContext()));
        Block *entry = kernelFunction.addEntryBlock(moduleBuilder);
        OpBuilder kernelBuilder(entry, entry->begin());
        auto mma = tile::TileMmaDescAttr::get(
            &getContext(), "auto", selected->tileM, selected->tileN,
            selected->tileK, selected->storage, selected->storage,
            selected->accum, "row_major", "col_major", 1);
        auto epilogue = tile::TileEpilogueAttr::get(
            &getContext(), selected->bias, selected->activation,
            selected->output);
        const bool typedSm120Producer = selected->m >= 16 && selected->m % 16 == 0 &&
                                        selected->n >= 8 && selected->n % 8 == 0 &&
                                        selected->k >= 16 && selected->k % 16 == 0 &&
                                        (selected->storage == "f16" ||
                                         selected->storage == "bf16") &&
                                        selected->accum == "f32" &&
                                        !fusedEpilogue &&
                                        selected->output == "f32";
        if (macroSm120Producer && selected->k % 8 == 0 && !fusedEpilogue &&
            selected->output == "f32") {
          // NVIDIA owns this physical reuse boundary. One 128-thread CTA
          // computes a 32x32 output tile: four warps own 2x2 16x16 quadrants,
          // and each warp emits two adjacent m16n8 MMA tiles. A[32,16] and
          // B[16,32] are staged once per K panel and protected by CTA barriers.
          OperationState macroState(loc, "tessera_nvidia.macro_cta_matmul");
          macroState.addOperands(entry->getArguments());
          macroState.addAttribute("arch", kernelBuilder.getStringAttr("sm_120"));
          macroState.addAttribute("cta_m", kernelBuilder.getI64IntegerAttr(32));
          macroState.addAttribute("cta_n", kernelBuilder.getI64IntegerAttr(32));
          macroState.addAttribute("tile_m", kernelBuilder.getI64IntegerAttr(16));
          macroState.addAttribute("tile_n", kernelBuilder.getI64IntegerAttr(8));
          macroState.addAttribute("tile_k", kernelBuilder.getI64IntegerAttr(16));
          macroState.addAttribute("warps", kernelBuilder.getI64IntegerAttr(4));
          macroState.addAttribute(
              "warp_ownership",
              kernelBuilder.getStringAttr("quadrant_2x2_two_n_tiles"));
          macroState.addAttribute(
              "storage", kernelBuilder.getStringAttr(selected->storage));
          macroState.addAttribute(
              "accum", kernelBuilder.getStringAttr(selected->accum));
          macroState.addAttribute(
              "staging",
              kernelBuilder.getStringAttr(
                  dynamicSm120 ? "masked_scalar_shared_ab_16bit"
                               : "cp_async_shared_ab_16bit"));
          macroState.addAttribute("stages",
                                  kernelBuilder.getI64IntegerAttr(
                                      dynamicSm120 ? 1 : 2));
          macroState.addAttribute(
              "completion",
              kernelBuilder.getStringAttr(
                  dynamicSm120 ? "cta_barrier"
                               : "wait_group_0_cta_barrier"));
          macroState.addAttribute(
              "bounds", kernelBuilder.getStringAttr("zero_fill_mnk_tail"));
          macroState.addAttribute(
              "grid_order", kernelBuilder.getStringAttr("column_major_xy"));
          macroState.addAttribute(
              "tessera.schedule_hash",
              kernelBuilder.getStringAttr(scheduled.getArtifactHash()));
          macroState.addAttribute(
              "tessera.macro_tile_m",
              kernelBuilder.getI64IntegerAttr(selected->macroTileM));
          macroState.addAttribute(
              "tessera.macro_tile_n",
              kernelBuilder.getI64IntegerAttr(selected->macroTileN));
          kernelBuilder.create(macroState);
        } else if (typedSm120Producer) {
          // The narrow, exact m16n8k16 seed uses the same proof-bearing
          // composed-layout carrier as the portable Tile path.  Larger
          // scheduled problems retain tile.matmul_kernel until their tiled
          // loop producer can preserve this fragment lineage end-to-end.
          auto tileType = tile::TileValueType::get(&getContext());
          auto typedMma = tile::TileMmaDescAttr::get(
              &getContext(), "mma_sync", selected->tileM, selected->tileN,
              selected->tileK, selected->storage, selected->storage,
              selected->accum, "row_major", "col_major", 1);
          SmallVector<StringAttr> laneReg{kernelBuilder.getStringAttr("laneid"),
                                          kernelBuilder.getStringAttr("reg")};
          auto aLayout = tile::TileLayoutAttr::get(
              &getContext(), {16, 16}, {16, 1}, laneReg, {}, {}, {}, 0,
              tile::TileSwizzleAttr());
          auto bLayout = tile::TileLayoutAttr::get(
              &getContext(), {16, 8}, {8, 1}, laneReg, {}, {}, {}, 0,
              tile::TileSwizzleAttr());
          auto aMemory = tile::TileMemoryLayoutAttr::get(
              &getContext(), "gmem", "row_major", dynamicSm120 ? 0 : selected->k);
          auto bMemory = tile::TileMemoryLayoutAttr::get(
              &getContext(), "gmem", "col_major", dynamicSm120 ? 0 : selected->k);
          auto dMemory = tile::TileMemoryLayoutAttr::get(
              &getContext(), "gmem", "row_major", dynamicSm120 ? 0 : selected->n);
          auto composed = [&](ArrayRef<int64_t> shape, ArrayRef<int64_t> strides) {
            auto leaf = [&](int64_t value) -> Attribute {
              return kernelBuilder.getI64IntegerAttr(value);
            };
            auto singleton = [&](int64_t value) {
              return kernelBuilder.getArrayAttr({leaf(value)});
            };
            auto basis = kernelBuilder.getArrayAttr({
                kernelBuilder.getArrayAttr({singleton(shape[0]), singleton(1)}),
                kernelBuilder.getArrayAttr({singleton(shape[1]), singleton(1)}),
            });
            return tile::TileComposedLayoutAttr::get(
                &getContext(), kernelBuilder.getArrayAttr({leaf(shape[0]), leaf(shape[1])}),
                kernelBuilder.getArrayAttr({leaf(strides[0]), leaf(strides[1])}),
                basis, {0, 0});
          };
          auto zero = arith::ConstantIntOp::create(kernelBuilder, loc, 0, 64);
          // The launch ABI maps grid.x to N/8 and grid.y to M/16.  Carry the
          // resulting logical origins into the shared views/materializations;
          // the physical packer may then use only the proven linear base plus
          // its fixed per-lane fragment coordinates.
          OperationState coordinateState(
              loc, "tessera_nvidia.block_coordinate");
          coordinateState.addTypes({i64, i64});
          coordinateState.addAttribute(
              "arch", kernelBuilder.getStringAttr("sm_120"));
          coordinateState.addAttribute(
              "tile_m", kernelBuilder.getI64IntegerAttr(16));
          coordinateState.addAttribute(
              "tile_n", kernelBuilder.getI64IntegerAttr(8));
          coordinateState.addAttribute(
              "grid_order",
              kernelBuilder.getStringAttr("column_major_xy"));
          Operation *coordinates = kernelBuilder.create(coordinateState);
          Value rowBase = coordinates->getResult(0);
          Value colBase = coordinates->getResult(1);
          auto materialize = [&](tile::TileComposedLayoutAttr layout,
                                 ValueRange values) {
            OperationState state(loc, "tile.materialize_composed_layout");
            state.addOperands(values); state.addTypes(i64);
            state.addAttribute("layout", layout);
            return kernelBuilder.create(state)->getResult(0);
          };
          auto view = [&](Value base, Value linear, Value row, Value col,
                          Value rowBound, Value colBound, Value leadingDim,
                          tile::TileLayoutAttr layout,
                          tile::TileMemoryLayoutAttr memory) {
            OperationState state(loc, "tile.view");
            state.addOperands({base, linear, row, col});
            if (dynamicSm120)
              state.addOperands({rowBound, colBound, leadingDim});
            state.addTypes(tileType);
            state.addAttribute("tile.layout", layout); state.addAttribute("tile.memory", memory);
            state.addAttribute("tile.linear_base", kernelBuilder.getUnitAttr());
            return kernelBuilder.create(state)->getResult(0);
          };
          auto aType = tile::FragmentType::get(&getContext(), 16, 8, 16, selected->storage, "f32", "a", "row_major", "mma_sync");
          auto bType = tile::FragmentType::get(&getContext(), 16, 8, 16, selected->storage, "f32", "b", "col_major", "mma_sync");
          auto cType = tile::FragmentType::get(&getContext(), 16, 8, 16, "f32", "f32", "acc", "row_major", "mma_sync");
          auto pack = [&](Value source, Type type, StringRef role) {
            OperationState state(loc, "tile.fragment_pack"); state.addOperands(source); state.addTypes(type);
            state.addAttribute("role", kernelBuilder.getStringAttr(role)); state.addAttribute("mma", typedMma);
            return kernelBuilder.create(state)->getResult(0);
          };
          OperationState zeroState(loc, "tile.fragment_zero"); zeroState.addTypes(cType);
          zeroState.addAttribute("role", kernelBuilder.getStringAttr("acc")); zeroState.addAttribute("mma", typedMma);
          Value c = kernelBuilder.create(zeroState)->getResult(0);
          unsigned dIndex = 2 + unsigned(selected->bias) +
                            unsigned(selected->residual);
          Value runtimeM = entry->getArgument(dIndex + 1);
          Value runtimeN = entry->getArgument(dIndex + 2);
          Value runtimeK = entry->getArgument(dIndex + 3);
          Value lda = dynamicSm120 ? entry->getArgument(dIndex + 4)
                                   : Value(arith::ConstantIntOp::create(kernelBuilder, loc, selected->k, 64));
          Value ldb = dynamicSm120 ? entry->getArgument(dIndex + 5)
                                   : Value(arith::ConstantIntOp::create(kernelBuilder, loc, selected->k, 64));
          Value ldd = dynamicSm120 ? entry->getArgument(dIndex + 6)
                                   : Value(arith::ConstantIntOp::create(kernelBuilder, loc, selected->n, 64));
          auto dynamicComposed = [&](bool columnMajor) {
            auto leaf = [&](int64_t value) -> Attribute {
              return kernelBuilder.getI64IntegerAttr(value);
            };
            auto singleton = [&](int64_t value) {
              return kernelBuilder.getArrayAttr({leaf(value)});
            };
            auto basis = kernelBuilder.getArrayAttr({
                kernelBuilder.getArrayAttr({singleton(-1), singleton(1)}),
                kernelBuilder.getArrayAttr({singleton(-1), singleton(1)}),
            });
            SmallVector<Attribute> outerStride = columnMajor
                ? SmallVector<Attribute>{singleton(1), singleton(-1)}
                : SmallVector<Attribute>{singleton(-1), singleton(1)};
            return tile::TileComposedLayoutAttr::get(
                &getContext(), kernelBuilder.getArrayAttr({singleton(-1), singleton(-1)}),
                kernelBuilder.getArrayAttr(outerStride),
                basis, {0, 0});
          };
          Value kLimit = dynamicSm120
              ? runtimeK
              : Value(arith::ConstantIntOp::create(kernelBuilder, loc, selected->k, 64));
          Value kStep = arith::ConstantIntOp::create(kernelBuilder, loc, 16, 64);
          auto kLoop = scf::ForOp::create(kernelBuilder, loc, zero, kLimit, kStep,
                                          ValueRange{c});
          {
            OpBuilder::InsertionGuard loopGuard(kernelBuilder);
            kernelBuilder.setInsertionPointToStart(kLoop.getBody());
            Value panel = kLoop.getInductionVar();
            Value carry = kLoop.getRegionIterArgs().front();
            auto aComposed = dynamicSm120
                ? dynamicComposed(false)
                : composed({selected->m, selected->k}, {selected->k, 1});
            auto bComposed = dynamicSm120
                ? dynamicComposed(true)
                : composed({selected->k, selected->n}, {1, selected->k});
            SmallVector<Value> aMaterialize{rowBase, panel};
            SmallVector<Value> bMaterialize{panel, colBase};
            if (dynamicSm120) {
              // Canonical dynamic-leaf order is outer shapes, outer strides,
              // then basis shapes.  The basis extents repeat the logical
              // bounds intentionally: mixed-radix identity is `coord % dim`,
              // never the old accidental `coord % 1` zero map.
              aMaterialize.append(
                  {runtimeM, runtimeK, lda, runtimeM, runtimeK});
              bMaterialize.append(
                  {runtimeK, runtimeN, ldb, runtimeK, runtimeN});
            }
            Value aTile = view(entry->getArgument(0),
                materialize(aComposed, aMaterialize), rowBase, panel,
                runtimeM, runtimeK, lda, aLayout, aMemory);
            Value bTile = view(entry->getArgument(1),
                materialize(bComposed, bMaterialize), panel, colBase,
                runtimeK, runtimeN, ldb, bLayout, bMemory);
            Value a = pack(aTile, aType, "a");
            Value b = pack(bTile, bType, "b");
            OperationState mmaState(loc, "tile.mma"); mmaState.addOperands({a, b, carry}); mmaState.addTypes(cType);
            mmaState.addAttribute("mma", typedMma); mmaState.addAttribute("tessera.schedule_hash", kernelBuilder.getStringAttr(scheduled.getArtifactHash()));
            mmaState.addAttribute("tessera.macro_tile_m", kernelBuilder.getI64IntegerAttr(selected->macroTileM));
            mmaState.addAttribute("tessera.macro_tile_n", kernelBuilder.getI64IntegerAttr(selected->macroTileN));
            Value next = kernelBuilder.create(mmaState)->getResult(0);
            scf::YieldOp::create(kernelBuilder, loc, next);
          }
          kernelBuilder.setInsertionPointAfter(kLoop);
          Value result = kLoop.getResult(0);
          OperationState unpackState(loc, "tile.fragment_unpack"); unpackState.addOperands(result); unpackState.addTypes(tileType);
          unpackState.addAttribute("tile.layout", bLayout); unpackState.addAttribute("mma", typedMma);
          Value outTile = kernelBuilder.create(unpackState)->getResult(0);
          OperationState storeState(loc, "tile.store");
          storeState.addOperands({outTile, entry->getArgument(dIndex), rowBase, colBase});
          if (dynamicSm120)
            storeState.addOperands({runtimeM, runtimeN, ldd});
          storeState.addAttribute("tile.layout", bLayout); storeState.addAttribute("tile.memory", dMemory);
          kernelBuilder.create(storeState);
        } else {
        OperationState kernelState(loc, "tile.matmul_kernel");
        kernelState.addOperands(entry->getArguments());
        kernelState.addAttribute("mma", mma);
        kernelState.addAttribute("epilogue", epilogue);
        kernelState.addAttribute(
            "warps", kernelBuilder.getI64IntegerAttr(
                         macroSm120Producer ? 4 : selected->warps));
        kernelState.addAttribute(
            "staging", kernelBuilder.getStringAttr(
                           macroSm120Producer ? "shared" : "global"));
        if (selected->residual) {
          kernelState.addAttribute("residual", kernelBuilder.getBoolAttr(true));
          kernelState.addAttribute(
              "epilogue_order",
              kernelBuilder.getStringAttr(
                  "matmul_bias_activation_residual"));
        }
        // A 16-byte cp.async vector may start at every row only when the
        // packed row stride is a multiple of eight 16-bit elements.  Ragged
        // K remains a valid macro-CTA case, but uses the masked scalar staging
        // path so alignment is a proof obligation rather than an assumption.
        if (macroSm120Producer && selected->k % 8 == 0) {
          kernelState.addAttribute("tessera.vectorized_shared_panel",
                                   kernelBuilder.getBoolAttr(true));
          kernelState.addAttribute("tessera.async_shared_panel",
                                   kernelBuilder.getBoolAttr(true));
        }
        kernelState.addAttribute(
            "numeric_policy",
            kernelBuilder.getDictionaryAttr({
                kernelBuilder.getNamedAttr("storage",
                                            kernelBuilder.getStringAttr(selected->storage)),
                kernelBuilder.getNamedAttr("accum",
                                            kernelBuilder.getStringAttr(selected->accum)),
            }));
        kernelState.addAttribute("tessera.canonical_k_loop",
                                 kernelBuilder.getBoolAttr(true));
        kernelState.addAttribute("tessera.tile_m",
                                 kernelBuilder.getI64IntegerAttr(selected->tileM));
        kernelState.addAttribute("tessera.tile_n",
                                 kernelBuilder.getI64IntegerAttr(selected->tileN));
        kernelState.addAttribute("tessera.tile_k",
                                 kernelBuilder.getI64IntegerAttr(selected->tileK));
        kernelState.addAttribute("tessera.macro_tile_m",
                                 kernelBuilder.getI64IntegerAttr(selected->macroTileM));
        kernelState.addAttribute("tessera.macro_tile_n",
                                 kernelBuilder.getI64IntegerAttr(selected->macroTileN));
        kernelState.addAttribute("tessera.pipeline_depth",
                                 kernelBuilder.getI64IntegerAttr(selected->pipelineDepth));
        kernelState.addAttribute("tessera.raster_order",
                                 kernelBuilder.getStringAttr(selected->rasterOrder));
        kernelState.addAttribute("tessera.raster_group",
                                 kernelBuilder.getI64IntegerAttr(selected->rasterGroup));
        kernelState.addAttribute("tessera.schedule_hash",
                                 kernelBuilder.getStringAttr(scheduled.getArtifactHash()));
        kernelBuilder.create(kernelState);
        }
        LLVM::ReturnOp::create(kernelBuilder, loc, ValueRange{});

        scheduled.getScheduled().replaceAllUsesWith(graph->getResult(0));
        scheduled.erase();
        for (schedule::ArtifactOp artifact : matchingArtifacts)
          artifact.erase();
        if (!llvm::is_contained(consumedNvidiaGraphFunctions, graphFunction))
          consumedNvidiaGraphFunctions.push_back(graphFunction);
        continue;
      }

      auto lhsType = cast<RankedTensorType>(graph->getOperand(0).getType());
      auto rhsType = cast<RankedTensorType>(graph->getOperand(1).getType());
      auto outType = cast<RankedTensorType>(graph->getResult(0).getType());
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
      Value runtimeMIndex, runtimeNIndex, runtimeKIndex;
      if (selected->dynamicM)
        runtimeMIndex = builder.create<tensor::DimOp>(loc, graph->getOperand(0), 0);
      if (selected->dynamicN)
        runtimeNIndex = builder.create<tensor::DimOp>(loc, graph->getOperand(1), 1);
      if (selected->dynamicK)
        runtimeKIndex = builder.create<tensor::DimOp>(loc, graph->getOperand(0), 1);
      SmallVector<Value> outputDynamicSizes;
      if (outType.isDynamicDim(0))
        outputDynamicSizes.push_back(runtimeMIndex);
      if (outType.isDynamicDim(1))
        outputDynamicSizes.push_back(runtimeNIndex);
      Value output = builder.create<memref::AllocOp>(
          loc, outputMemref, outputDynamicSizes);
      Value outputIndex =
          builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, output);
      Value outputInteger = builder.create<arith::IndexCastOp>(
          loc, builder.getI64Type(), outputIndex);
      Value d = builder.create<LLVM::IntToPtrOp>(loc, pointerType, outputInteger);
      auto extentI64 = [&](bool dynamic, Value runtimeIndex, int64_t bound) {
        if (dynamic)
          return Value(builder.create<arith::IndexCastOp>(
              loc, builder.getI64Type(), runtimeIndex));
        return Value(builder.create<arith::ConstantIntOp>(loc, bound, 64));
      };
      Value m = extentI64(selected->dynamicM, runtimeMIndex, selected->m);
      Value n = extentI64(selected->dynamicN, runtimeNIndex, selected->n);
      Value k = extentI64(selected->dynamicK, runtimeKIndex, selected->k);

      StringRef family = selected->target == "rocm" ? "wmma" : "auto";
      auto mma = tile::TileMmaDescAttr::get(
          &getContext(), family, selected->tileM, selected->tileN,
          selected->tileK, selected->storage,
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
    for (func::FuncOp graphFunction : consumedNvidiaGraphFunctions)
      graphFunction.erase();

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

    SmallVector<Operation *> scheduledSpectralBackwards;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "schedule.spectral_backward")
        scheduledSpectralBackwards.push_back(op);
    });
    for (Operation *scheduled : scheduledSpectralBackwards) {
      auto hash = scheduled->getAttrOfType<StringAttr>("artifact_hash");
      FailureOr<std::string> derived = spectralBackwardDigest(scheduled);
      if (failed(derived) || !hash || *derived != hash.getValue()) {
        scheduled->emitError(
            "scheduled spectral adjoint policy was altered after hashing");
        return signalPassFailure();
      }
      if (scheduled->getNumOperands() == 0) {
        scheduled->emitError("requires retained Graph spectral adjoint results");
        return signalPassFailure();
      }
      Operation *graph = scheduled->getOperand(0).getDefiningOp();
      if (!graph || graph->getName().getStringRef() !=
                        "tessera.spectral_backward") {
        scheduled->emitError("requires the retained Graph spectral adjoint");
        return signalPassFailure();
      }
      SmallVector<schedule::ArtifactOp> matchingArtifacts;
      mod.walk([&](schedule::ArtifactOp artifact) {
        if (artifact.getHash() == hash.getValue())
          matchingArtifacts.push_back(artifact);
      });
      if (matchingArtifacts.size() != 1) {
        scheduled->emitError("requires exactly one matching schedule.artifact");
        return signalPassFailure();
      }

      Location loc = scheduled->getLoc();
      builder.setInsertionPoint(scheduled);
      auto pointerType = LLVM::LLVMPointerType::get(&getContext());
      auto tensorPointer = [&](Value tensor) -> FailureOr<Value> {
        auto type = dyn_cast<RankedTensorType>(tensor.getType());
        if (!type || !type.hasStaticShape()) return failure();
        auto memref = MemRefType::get(type.getShape(), type.getElementType());
        Value buffer =
            builder.create<bufferization::ToBufferOp>(loc, memref, tensor);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), index);
        return builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer)
            .getResult();
      };
      SmallVector<Value> launchOperands;
      for (Value input : graph->getOperands()) {
        FailureOr<Value> pointer = tensorPointer(input);
        if (failed(pointer)) {
          scheduled->emitError("requires static ranked adjoint inputs");
          return signalPassFailure();
        }
        launchOperands.push_back(*pointer);
      }
      SmallVector<Value> outputBuffers;
      for (Type type : scheduled->getResultTypes()) {
        auto tensor = dyn_cast<RankedTensorType>(type);
        if (!tensor || !tensor.hasStaticShape()) {
          scheduled->emitError("requires static ranked gradient outputs");
          return signalPassFailure();
        }
        auto memref = MemRefType::get(tensor.getShape(), tensor.getElementType());
        Value buffer = builder.create<memref::AllocOp>(loc, memref);
        outputBuffers.push_back(buffer);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), index);
        launchOperands.push_back(
            builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer));
      }

      OperationState kernelState(loc, "tile.spectral_backward_kernel");
      kernelState.addOperands(launchOperands);
      kernelState.addAttribute(
          "input_count", builder.getI64IntegerAttr(graph->getNumOperands()));
      kernelState.addAttribute(
          "output_count", builder.getI64IntegerAttr(scheduled->getNumResults()));
      for (StringRef name : {"target", "arch", "kind", "axis",
                             "logical_length", "normalization",
                             "spectrum_layout", "hop", "center", "onesided",
                             "pad_mode", "output_length",
                             "input_signature", "output_signature",
                             "mutation_lineage"})
        if (Attribute attr = scheduled->getAttr(name))
          kernelState.addAttribute(name, attr);
      auto kindAttr = scheduled->getAttrOfType<StringAttr>("kind");
      auto axisAttr = scheduled->getAttrOfType<IntegerAttr>("axis");
      if (!kindAttr || !axisAttr) {
        scheduled->emitError(
            "native compound adjoints require explicit kind and axis");
        return signalPassFailure();
      }
      StringRef kind = kindAttr.getValue();
      auto ranked = [](Type type) { return dyn_cast<RankedTensorType>(type); };
      auto dyType = ranked(graph->getOperand(0).getType());
      auto inputType = ranked(graph->getOperand(1).getType());
      auto parameterType = ranked(graph->getOperand(2).getType());
      if (!dyType || !inputType || !parameterType ||
          dyType.getRank() != inputType.getRank() ||
          inputType.getRank() != parameterType.getRank()) {
        scheduled->emitError("native compound adjoints require equal-rank static operands");
        return signalPassFailure();
      }
      int64_t axis = axisAttr.getInt();
      if (axis < 0) axis += inputType.getRank();
      if (axis != inputType.getRank() - 1) {
        scheduled->emitError("initial native compound adjoints require the last axis");
        return signalPassFailure();
      }
      int64_t inputLength = inputType.getDimSize(axis);
      int64_t parameterLength = parameterType.getDimSize(axis);
      int64_t cotangentLength = dyType.getDimSize(axis);
      if (inputLength <= 0 || parameterLength <= 0 || cotangentLength <= 0) {
        scheduled->emitError(
            "initial native compound adjoints require positive static extents");
        return signalPassFailure();
      }
      int64_t batch = inputType.getNumElements() / inputLength;
      if (kind == "tessera.spectral_filter") {
        auto complex = dyn_cast<ComplexType>(inputType.getElementType());
        if (!complex || !complex.getElementType().isF32() ||
            inputType != parameterType || inputType != dyType) {
          scheduled->emitError("native spectral-filter adjoint requires identical complex<f32> tensors");
          return signalPassFailure();
        }
        kernelState.addAttribute(
            "elements", builder.getI64IntegerAttr(inputType.getNumElements()));
      } else if (kind == "tessera.spectral_conv") {
        if (!inputType.getElementType().isF32() ||
            !parameterType.getElementType().isF32() ||
            !dyType.getElementType().isF32() ||
            cotangentLength != inputLength + parameterLength - 1 ||
            parameterType.getNumElements() / parameterLength != batch ||
            dyType.getNumElements() / cotangentLength != batch) {
          scheduled->emitError("native spectral-conv adjoint requires unbroadcast batched f32 full convolution");
          return signalPassFailure();
        }
        auto logicalLengthAttr =
            scheduled->getAttrOfType<IntegerAttr>("logical_length");
        auto normalizationAttr =
            scheduled->getAttrOfType<StringAttr>("normalization");
        if (!logicalLengthAttr || logicalLengthAttr.getInt() <= 0 ||
            !normalizationAttr) {
          scheduled->emitError(
              "native spectral-conv adjoint requires logical_length and normalization");
          return signalPassFailure();
        }
        int64_t logicalLength = logicalLengthAttr.getInt();
        StringRef normalization = normalizationAttr.getValue();
        double scale = normalization == "forward"
                           ? 1.0 / logicalLength
                           : normalization == "ortho"
                                 ? 1.0 / std::sqrt(double(logicalLength))
                                 : 1.0;
        kernelState.addAttribute("batch", builder.getI64IntegerAttr(batch));
        kernelState.addAttribute(
            "cotangent_length", builder.getI64IntegerAttr(cotangentLength));
        kernelState.addAttribute(
            "input_length", builder.getI64IntegerAttr(inputLength));
        kernelState.addAttribute(
            "parameter_length", builder.getI64IntegerAttr(parameterLength));
        kernelState.addAttribute(
            "normalization_scale", builder.getF32FloatAttr(scale));
      }
      kernelState.addAttribute("tessera.schedule_hash", hash);
      builder.create(kernelState);

      for (auto [result, buffer, type] :
           llvm::zip(scheduled->getResults(), outputBuffers,
                     scheduled->getResultTypes())) {
        Value tensor = builder.create<bufferization::ToTensorOp>(
            loc, cast<RankedTensorType>(type), buffer);
        result.replaceAllUsesWith(tensor);
      }
      scheduled->erase();
      if (graph->use_empty()) graph->erase();
      for (schedule::ArtifactOp artifact : matchingArtifacts) artifact.erase();
    }

    SmallVector<Operation *> scheduledSpectralPrograms;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "schedule.spectral_program")
        scheduledSpectralPrograms.push_back(op);
    });
    for (Operation *scheduled : scheduledSpectralPrograms) {
      auto hash = scheduled->getAttrOfType<StringAttr>("artifact_hash");
      auto scheduleDigest =
          mod->getAttrOfType<StringAttr>("tessera.schedule_digest");
      if (!hash || !scheduleDigest || hash.getValue().size() != 64 ||
          hash.getValue() != scheduleDigest.getValue()) {
        scheduled->emitError(
            "scheduled spectral program requires the module Schedule Object digest");
        return signalPassFailure();
      }
      // Matching digest strings alone do NOT bind the attributes this pass
      // then consumes: a cached or hand-edited program could keep both
      // digests and still change workspace_bytes or native_entry, which the
      // native launch would use (PR #626 review). Re-verify every consumed
      // policy value against the semantic payload the producer carried.
      auto semantic =
          mod->getAttrOfType<StringAttr>("tessera.spectral_semantic");
      auto schedulePayload =
          mod->getAttrOfType<StringAttr>("tessera.schedule_payload");
      if (!semantic || semantic.getValue().empty() || !schedulePayload ||
          schedulePayload.getValue().empty()) {
        scheduled->emitError(
            "scheduled spectral program requires the module "
            "tessera.spectral_semantic and tessera.schedule_payload "
            "preimages to verify consumed policy");
        return signalPassFailure();
      }
      // Verify the identity CHAIN, so the attributes and the declaration
      // that vouches for them cannot be co-edited:
      //   sha256(schedule_payload) == tessera.schedule_digest, and that
      //   payload names object_id "spectral:<sha256(spectral_semantic)>".
      // Breaking any link changes a hash.
      auto hexDigest = [](StringRef text) {
        llvm::SHA256 hasher;
        hasher.update(text);
        return llvm::toHex(hasher.final(), /*LowerCase=*/true);
      };
      if (hexDigest(schedulePayload.getValue()) != scheduleDigest.getValue()) {
        scheduled->emitError(
            "tessera.schedule_payload does not hash to the module Schedule "
            "Object digest; the carried schedule is not the one addressed");
        return signalPassFailure();
      }
      std::string expectedObjectId =
          ("spectral:" + hexDigest(semantic.getValue()));
      if (!schedulePayload.getValue().contains(expectedObjectId)) {
        scheduled->emitError(
            "tessera.spectral_semantic does not hash to the object_id inside "
            "the digest-bound schedule payload; the declared policy is not "
            "the one the schedule identity covers");
        return signalPassFailure();
      }
      auto payloadField = [&](StringRef field) -> StringRef {
        StringRef payload = semantic.getValue();
        size_t at = payload.find((field + "=").str());
        while (at != StringRef::npos && at != 0 && payload[at - 1] != ';')
          at = payload.find((field + "=").str(), at + 1);
        if (at == StringRef::npos) return StringRef();
        StringRef rest = payload.drop_front(at + field.size() + 1);
        size_t end = rest.find(';');
        return end == StringRef::npos ? rest : rest.take_front(end);
      };
      auto requireMatches = [&](StringRef field, const Twine &actual) -> bool {
        StringRef declared = payloadField(field);
        if (declared.empty() || declared != actual.str()) {
          scheduled->emitError("scheduled spectral program attribute '")
              << field << "' is " << actual
              << ", which disagrees with the carried semantic payload ('"
              << declared << "'); the digest does not cover a changed policy";
          return false;
        }
        return true;
      };
      if (auto workspaceAttr =
              scheduled->getAttrOfType<IntegerAttr>("workspace_bytes")) {
        if (!requireMatches("workspace_bytes",
                            Twine(workspaceAttr.getInt())))
          return signalPassFailure();
      }
      if (auto entryAttr =
              scheduled->getAttrOfType<StringAttr>("native_entry")) {
        if (!requireMatches("native_entry", entryAttr.getValue()))
          return signalPassFailure();
      }
      if (auto normalizationAttr =
              scheduled->getAttrOfType<StringAttr>("normalization")) {
        if (!requireMatches("normalization", normalizationAttr.getValue()))
          return signalPassFailure();
      }
      SmallVector<schedule::ArtifactOp> matchingArtifacts;
      mod.walk([&](schedule::ArtifactOp artifact) {
        if (artifact.getHash() == hash.getValue())
          matchingArtifacts.push_back(artifact);
      });
      if (matchingArtifacts.size() != 1) {
        scheduled->emitError(
            "requires exactly one matching schedule.artifact");
        return signalPassFailure();
      }
      auto outputType = dyn_cast<RankedTensorType>(scheduled->getResult(0).getType());
      if (!outputType || !outputType.hasStaticShape()) {
        scheduled->emitError("requires a static ranked output tensor");
        return signalPassFailure();
      }

      Location loc = scheduled->getLoc();
      builder.setInsertionPoint(scheduled);
      auto pointerType = LLVM::LLVMPointerType::get(&getContext());
      SmallVector<Value> launchOperands;
      for (Value subject : scheduled->getOperands()) {
        auto inputType = dyn_cast<RankedTensorType>(subject.getType());
        if (!inputType || !inputType.hasStaticShape()) {
          scheduled->emitError("requires static ranked input tensors");
          return signalPassFailure();
        }
        auto memref = MemRefType::get(inputType.getShape(), inputType.getElementType());
        Value buffer = builder.create<bufferization::ToBufferOp>(loc, memref, subject);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer =
            builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), index);
        launchOperands.push_back(
            builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer));
      }
      auto outputMemref =
          MemRefType::get(outputType.getShape(), outputType.getElementType());
      Value outputBuffer = builder.create<memref::AllocOp>(loc, outputMemref);
      Value outputIndex =
          builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outputBuffer);
      Value outputInteger = builder.create<arith::IndexCastOp>(
          loc, builder.getI64Type(), outputIndex);
      launchOperands.push_back(
          builder.create<LLVM::IntToPtrOp>(loc, pointerType, outputInteger));
      auto workspace = scheduled->getAttrOfType<IntegerAttr>("workspace_bytes");
      launchOperands.push_back(builder.create<arith::ConstantIntOp>(
          loc, workspace.getInt(), 64));

      OperationState kernelState(loc, "tile.spectral_program_kernel");
      kernelState.addOperands(launchOperands);
      kernelState.addAttribute(
          "input_count",
          builder.getI64IntegerAttr(scheduled->getNumOperands()));
      for (StringRef name :
           {"target", "arch", "kind", "input_shapes", "input_signature",
            "shape_bounds", "template_digest", "shape_policy",
            "storage", "abi_storage", "storage_conversion", "axis_packing",
            "normalization",
            "complex_layout", "accumulation", "workspace_policy", "fusion_topology",
            "mutation_lineage", "native_entry", "child_fft_digests"})
        kernelState.addAttribute(name, scheduled->getAttr(name));
      for (StringRef name : {"output_shape", "axis", "dct_type", "padding", "crop",
                             "window_length", "hop", "frames",
                             "workspace_bytes"})
        kernelState.addAttribute(name, scheduled->getAttr(name));
      kernelState.addAttribute(
          "tessera.workgroup_size",
          scheduled->getAttr("workgroup_size"));
      kernelState.addAttribute("tessera.schedule_hash", hash);
      builder.create(kernelState);

      Value result = builder.create<bufferization::ToTensorOp>(
          loc, outputType, outputBuffer);
      scheduled->getResult(0).replaceAllUsesWith(result);
      scheduled->erase();
      for (schedule::ArtifactOp artifact : matchingArtifacts) artifact.erase();
    }

    SmallVector<Operation *> scheduledFFTs;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "schedule.fft")
        scheduledFFTs.push_back(op);
    });
    for (Operation *scheduled : scheduledFFTs) {
      Operation *graph = scheduled->getOperand(0).getDefiningOp();
      if (!graph) {
        scheduled->emitError("requires the retained Graph FFT result");
        return signalPassFailure();
      }
      FailureOr<FFTSchedule> selected = getFFTSchedule(graph);
      auto hash = scheduled->getAttrOfType<StringAttr>("artifact_hash");
      if (failed(selected) || !hash ||
          fftScheduleDigest(*selected) != hash.getValue()) {
        scheduled->emitError(
            "scheduled FFT decision does not match the retained Graph contract");
        return signalPassFailure();
      }
      auto stringAttr = [&](StringRef name) -> StringRef {
        auto attr = scheduled->getAttrOfType<StringAttr>(name);
        return attr ? attr.getValue() : StringRef();
      };
      auto intAttr = [&](StringRef name) -> std::optional<int64_t> {
        auto attr = scheduled->getAttrOfType<IntegerAttr>(name);
        return attr ? std::optional<int64_t>(attr.getInt()) : std::nullopt;
      };
      auto boolAttr = [&](StringRef name) -> std::optional<bool> {
        auto attr = scheduled->getAttrOfType<BoolAttr>(name);
        return attr ? std::optional<bool>(attr.getValue()) : std::nullopt;
      };
      auto sequence =
          scheduled->getAttrOfType<DenseI64ArrayAttr>("radix_sequence");
      bool altered =
          stringAttr("arch") != selected->arch ||
          stringAttr("mode") != selected->mode ||
          intAttr("axis") != selected->axis ||
          intAttr("length") != selected->length ||
          intAttr("physical_length") != selected->physicalLength ||
          intAttr("batch") != selected->batch ||
          stringAttr("real_transform_policy") !=
              selected->realTransformPolicy ||
          stringAttr("hermitian_layout") != selected->hermitianLayout ||
          stringAttr("hermitian_weight") != selected->hermitianWeight ||
          boolAttr("inverse") != selected->inverse ||
          stringAttr("normalization") != selected->normalization ||
          stringAttr("storage") != "complex64_interleaved_f32" ||
          stringAttr("accum") != "f32" ||
          stringAttr("radix_policy") != selected->radixPolicy ||
          stringAttr("strategy") != selected->strategy || !sequence ||
          stringAttr("algorithm") != selected->algorithm ||
          ArrayRef<int64_t>(sequence.asArrayRef()) !=
              ArrayRef<int64_t>(selected->radixSequence) ||
          intAttr("bluestein_m") != selected->bluesteinM ||
          intAttr("workspace_elems") != selected->workspaceElems ||
          stringAttr("workspace_policy") != selected->workspacePolicy ||
          stringAttr("residency") != selected->residency ||
          stringAttr("twiddle_layout") != "interleaved_f32" ||
          stringAttr("twiddle_policy") != selected->twiddlePolicy ||
          boolAttr("deterministic") != true ||
          stringAttr("kernel_family") != selected->kernelFamily ||
          intAttr("workgroup_size") != selected->workgroupSize;
      if (altered) {
        scheduled->emitError("scheduled FFT policy was altered after hashing");
        return signalPassFailure();
      }
      auto graphDigest =
          graph->getAttrOfType<StringAttr>("schedule.artifact_hash");
      SmallVector<schedule::ArtifactOp> matchingArtifacts;
      mod.walk([&](schedule::ArtifactOp artifact) {
        if (artifact.getHash() == hash.getValue())
          matchingArtifacts.push_back(artifact);
      });
      if (!graphDigest || graphDigest.getValue() != hash.getValue() ||
          matchingArtifacts.size() != 1) {
        scheduled->emitError(
            "requires exactly one matching Graph hash and schedule.artifact");
        return signalPassFailure();
      }

      Location loc = scheduled->getLoc();
      builder.setInsertionPoint(scheduled);
      auto pointerType = LLVM::LLVMPointerType::get(&getContext());
      auto inputType = cast<RankedTensorType>(graph->getOperand(0).getType());
      auto inputMemref =
          MemRefType::get(inputType.getShape(), inputType.getElementType());
      Value inputBuffer = builder.create<bufferization::ToBufferOp>(
          loc, inputMemref, graph->getOperand(0));
      Value inputIndex = builder.create<memref::ExtractAlignedPointerAsIndexOp>(
          loc, inputBuffer);
      Value inputInteger =
          builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), inputIndex);
      Value inputPointer =
          builder.create<LLVM::IntToPtrOp>(loc, pointerType, inputInteger);
      auto outputType = cast<RankedTensorType>(graph->getResult(0).getType());
      auto outputMemref =
          MemRefType::get(outputType.getShape(), outputType.getElementType());
      Value outputBuffer = builder.create<memref::AllocOp>(loc, outputMemref);
      Value outputIndex = builder.create<memref::ExtractAlignedPointerAsIndexOp>(
          loc, outputBuffer);
      Value outputInteger = builder.create<arith::IndexCastOp>(
          loc, builder.getI64Type(), outputIndex);
      Value outputPointer =
          builder.create<LLVM::IntToPtrOp>(loc, pointerType, outputInteger);
      Value batch =
          builder.create<arith::ConstantIntOp>(loc, selected->batch, 64);
      Value length =
          builder.create<arith::ConstantIntOp>(loc, selected->length, 64);

      OperationState kernelState(loc, "tile.fft_kernel");
      kernelState.addOperands({inputPointer, outputPointer, batch, length});
      kernelState.addAttribute("mode", builder.getStringAttr(selected->mode));
      kernelState.addAttribute("axis", builder.getI64IntegerAttr(selected->axis));
      kernelState.addAttribute("length",
                               builder.getI64IntegerAttr(selected->length));
      kernelState.addAttribute("physical_length",
                               builder.getI64IntegerAttr(selected->physicalLength));
      kernelState.addAttribute("batch",
                               builder.getI64IntegerAttr(selected->batch));
      kernelState.addAttribute(
          "real_transform_policy",
          builder.getStringAttr(selected->realTransformPolicy));
      kernelState.addAttribute("hermitian_layout",
                               builder.getStringAttr(selected->hermitianLayout));
      kernelState.addAttribute("hermitian_weight",
                               builder.getStringAttr(selected->hermitianWeight));
      kernelState.addAttribute("inverse", builder.getBoolAttr(selected->inverse));
      kernelState.addAttribute("normalization",
                               builder.getStringAttr(selected->normalization));
      kernelState.addAttribute("scale", builder.getF32FloatAttr(selected->scale));
      kernelState.addAttribute("storage",
                               builder.getStringAttr("complex64_interleaved_f32"));
      kernelState.addAttribute("accum", builder.getStringAttr("f32"));
      kernelState.addAttribute("radix_policy",
                               builder.getStringAttr(selected->radixPolicy));
      kernelState.addAttribute("strategy",
                               builder.getStringAttr(selected->strategy));
      kernelState.addAttribute("algorithm",
                               builder.getStringAttr(selected->algorithm));
      kernelState.addAttribute("radix_sequence",
                               builder.getDenseI64ArrayAttr(selected->radixSequence));
      kernelState.addAttribute("bluestein_m",
                               builder.getI64IntegerAttr(selected->bluesteinM));
      kernelState.addAttribute("workspace_elems",
                               builder.getI64IntegerAttr(selected->workspaceElems));
      kernelState.addAttribute("workspace_policy",
                               builder.getStringAttr(selected->workspacePolicy));
      kernelState.addAttribute("residency",
                               builder.getStringAttr(selected->residency));
      kernelState.addAttribute("twiddle_layout",
                               builder.getStringAttr("interleaved_f32"));
      kernelState.addAttribute("twiddle_policy",
                               builder.getStringAttr(selected->twiddlePolicy));
      kernelState.addAttribute("deterministic", builder.getBoolAttr(true));
      kernelState.addAttribute("kernel_family",
                               builder.getStringAttr(selected->kernelFamily));
      kernelState.addAttribute(
          "tessera.workgroup_size",
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

    SmallVector<schedule::DepthAttentionOp> scheduledDepthAttentions;
    mod.walk([&](schedule::DepthAttentionOp op) {
      scheduledDepthAttentions.push_back(op);
    });
    for (schedule::DepthAttentionOp scheduled : scheduledDepthAttentions) {
      Operation *graph = scheduled.getSubject().getDefiningOp();
      if (!graph || graph->getName().getStringRef() != "tessera.depth_attn") {
        scheduled.emitError(
            "BLOCK-ATTNRES-1 requires the retained Graph depth-attention result");
        return signalPassFailure();
      }
      FailureOr<DepthAttentionSchedule> selected =
          getDepthAttentionSchedule(graph);
      if (failed(selected) || depthAttentionScheduleDigest(*selected) !=
                                  scheduled.getArtifactHash()) {
        scheduled.emitError(
            "scheduled decision does not match the retained Graph depth-attention contract");
        return signalPassFailure();
      }
      bool altered =
          scheduled.getArch() != selected->arch ||
          scheduled.getStorage() != selected->storage ||
          scheduled.getSoftmax() != selected->softmax ||
          scheduled.getAccum() != selected->accum ||
          scheduled.getEps().convertToDouble() != selected->eps ||
          static_cast<int64_t>(scheduled.getSourceCount()) !=
              selected->sourceCount ||
          static_cast<int64_t>(scheduled.getRows()) != selected->rows ||
          static_cast<int64_t>(scheduled.getWidth()) != selected->width ||
          static_cast<int64_t>(scheduled.getSourceTile()) !=
              selected->sourceTile ||
          static_cast<int64_t>(scheduled.getWorkgroupSize()) !=
              selected->workgroupSize ||
          scheduled.getStatisticsRecurrence() !=
              selected->statisticsRecurrence ||
          scheduled.getMergeRecurrence() != selected->mergeRecurrence ||
          scheduled.getReassociable() != selected->reassociable;
      if (altered) {
        scheduled.emitError(
            "scheduled depth-attention policy was altered after hashing");
        return signalPassFailure();
      }
      auto graphDigest =
          graph->getAttrOfType<StringAttr>("schedule.artifact_hash");
      SmallVector<schedule::ArtifactOp> matchingArtifacts;
      mod.walk([&](schedule::ArtifactOp artifact) {
        if (artifact.getHash() == scheduled.getArtifactHash())
          matchingArtifacts.push_back(artifact);
      });
      if (!graphDigest ||
          graphDigest.getValue() != scheduled.getArtifactHash() ||
          matchingArtifacts.size() != 1) {
        scheduled.emitError(
            "requires exactly one matching Graph hash and schedule.artifact");
        return signalPassFailure();
      }

      Location loc = scheduled.getLoc();
      builder.setInsertionPoint(scheduled);
      auto pointerType = LLVM::LLVMPointerType::get(&getContext());
      auto toPointer = [&](Value tensor) -> Value {
        auto type = cast<RankedTensorType>(tensor.getType());
        auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
        Value buffer = builder.create<bufferization::ToBufferOp>(
            loc, memrefType, tensor);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), index);
        return builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer);
      };
      SmallVector<Value> operands = {toPointer(graph->getOperand(0)),
                                     toPointer(graph->getOperand(1))};
      auto outputType = cast<RankedTensorType>(graph->getResult(0).getType());
      auto outputMemref =
          MemRefType::get(outputType.getShape(), outputType.getElementType());
      Value outputBuffer = builder.create<memref::AllocOp>(loc, outputMemref);
      Value outputIndex =
          builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outputBuffer);
      Value outputInteger = builder.create<arith::IndexCastOp>(
          loc, builder.getI64Type(), outputIndex);
      operands.push_back(
          builder.create<LLVM::IntToPtrOp>(loc, pointerType, outputInteger));
      for (int64_t extent :
           {selected->sourceCount, selected->rows, selected->width})
        operands.push_back(
            builder.create<arith::ConstantIntOp>(loc, extent, 64));

      OperationState kernelState(loc, "tile.depth_attention_kernel");
      kernelState.addOperands(operands);
      kernelState.addAttribute("storage",
                               builder.getStringAttr(selected->storage));
      kernelState.addAttribute("arch", builder.getStringAttr(selected->arch));
      kernelState.addAttribute("softmax",
                               builder.getStringAttr(selected->softmax));
      kernelState.addAttribute("accum",
                               builder.getStringAttr(selected->accum));
      kernelState.addAttribute("eps",
                               builder.getF64FloatAttr(selected->eps));
      kernelState.addAttribute(
          "source_tile", builder.getI64IntegerAttr(selected->sourceTile));
      kernelState.addAttribute(
          "tessera.workgroup_size",
          builder.getI64IntegerAttr(selected->workgroupSize));
      kernelState.addAttribute(
          "statistics_recurrence",
          builder.getStringAttr(selected->statisticsRecurrence));
      kernelState.addAttribute(
          "merge_recurrence",
          builder.getStringAttr(selected->mergeRecurrence));
      kernelState.addAttribute("reassociable",
                               builder.getBoolAttr(selected->reassociable));
      kernelState.addAttribute(
          "tessera.schedule_hash",
          builder.getStringAttr(scheduled.getArtifactHash()));
      builder.create(kernelState);

      Value result = builder.create<bufferization::ToTensorOp>(
          loc, outputType, outputBuffer);
      scheduled.getScheduled().replaceAllUsesWith(result);
      scheduled.erase();
      if (graph->use_empty())
        graph->erase();
      for (schedule::ArtifactOp artifact : matchingArtifacts)
        artifact.erase();
    }

    SmallVector<schedule::AttentionOp> scheduledAttentions;
    mod.walk([&](schedule::AttentionOp op) {
      scheduledAttentions.push_back(op);
    });
    for (schedule::AttentionOp scheduled : scheduledAttentions) {
      Operation *graph = scheduled.getSubject().getDefiningOp();
      if (!graph || graph->getName().getStringRef() != "tessera.flash_attn") {
        scheduled.emitError(
            "E2E-REAL-5A requires the retained Graph attention result");
        return signalPassFailure();
      }
      FailureOr<AttentionSchedule> selected = getAttentionSchedule(graph);
      if (failed(selected) ||
          attentionScheduleDigest(*selected) != scheduled.getArtifactHash()) {
        scheduled.emitError(
            "scheduled decision does not match the retained Graph attention contract");
        return signalPassFailure();
      }
      bool altered =
          scheduled.getArch() != selected->arch ||
          scheduled.getStorage() != selected->storage ||
          scheduled.getAccum() != selected->accum ||
          scheduled.getScale().convertToDouble() != selected->scale ||
          scheduled.getCausal() != selected->causal ||
          scheduled.getBias() != selected->bias ||
          static_cast<int64_t>(scheduled.getWindowLeft()) !=
              selected->windowLeft ||
          static_cast<int64_t>(scheduled.getWindowRight()) !=
              selected->windowRight ||
          scheduled.getSoftcap().convertToDouble() != selected->softcap ||
          scheduled.getDropoutP().convertToDouble() != selected->dropoutP ||
          static_cast<int64_t>(scheduled.getDropoutSeed()) !=
              selected->dropoutSeed ||
          static_cast<int64_t>(scheduled.getTileQ()) != selected->tileQ ||
          static_cast<int64_t>(scheduled.getTileKv()) != selected->tileKV ||
          static_cast<int64_t>(scheduled.getWorkgroupSize()) !=
              selected->workgroupSize ||
          scheduled.getRecurrence() != selected->recurrence ||
          scheduled.getBackwardLsePolicy() != selected->backwardLsePolicy ||
          scheduled.getBackwardLseSelection() !=
              selected->backwardLseSelection;
      if (altered) {
        scheduled.emitError(
            "scheduled attention policy was altered after hashing");
        return signalPassFailure();
      }
      auto graphDigest =
          graph->getAttrOfType<StringAttr>("schedule.artifact_hash");
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

      Location loc = scheduled.getLoc();
      builder.setInsertionPoint(scheduled);
      auto pointerType = LLVM::LLVMPointerType::get(&getContext());
      auto toPointer = [&](Value tensor) -> Value {
        auto type = cast<RankedTensorType>(tensor.getType());
        auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
        Value buffer = builder.create<bufferization::ToBufferOp>(
            loc, memrefType, tensor);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer =
            builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), index);
        return builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer);
      };
      SmallVector<Value> operands;
      operands.push_back(toPointer(graph->getOperand(0)));
      operands.push_back(toPointer(graph->getOperand(1)));
      operands.push_back(toPointer(graph->getOperand(2)));
      if (selected->bias)
        operands.push_back(toPointer(graph->getOperand(3)));

      auto outputType = cast<RankedTensorType>(graph->getResult(0).getType());
      auto outputMemref =
          MemRefType::get(outputType.getShape(), outputType.getElementType());
      Value outputBuffer = builder.create<memref::AllocOp>(loc, outputMemref);
      Value outputIndex =
          builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outputBuffer);
      Value outputInteger = builder.create<arith::IndexCastOp>(
          loc, builder.getI64Type(), outputIndex);
      operands.push_back(
          builder.create<LLVM::IntToPtrOp>(loc, pointerType, outputInteger));
      for (int64_t dimension :
           {selected->batch, selected->queryHeads, selected->kvHeads,
            selected->queryRows, selected->keyRows, selected->headDim,
            selected->valueDim})
        operands.push_back(
            builder.create<arith::ConstantIntOp>(loc, dimension, 64));

      OperationState kernelState(loc, "tile.attention_kernel");
      kernelState.addOperands(operands);
      kernelState.addAttribute("storage",
                               builder.getStringAttr(selected->storage));
      kernelState.addAttribute("accum", builder.getStringAttr(selected->accum));
      kernelState.addAttribute("scale", builder.getF32FloatAttr(selected->scale));
      kernelState.addAttribute("causal", builder.getBoolAttr(selected->causal));
      kernelState.addAttribute("bias", builder.getBoolAttr(selected->bias));
      kernelState.addAttribute("window_left",
                               builder.getI64IntegerAttr(selected->windowLeft));
      kernelState.addAttribute("window_right",
                               builder.getI64IntegerAttr(selected->windowRight));
      kernelState.addAttribute("softcap",
                               builder.getF32FloatAttr(selected->softcap));
      kernelState.addAttribute("dropout_p",
                               builder.getF32FloatAttr(selected->dropoutP));
      kernelState.addAttribute("dropout_seed",
                               builder.getI64IntegerAttr(selected->dropoutSeed));
      kernelState.addAttribute("head_dim",
                               builder.getI64IntegerAttr(selected->headDim));
      kernelState.addAttribute("value_dim",
                               builder.getI64IntegerAttr(selected->valueDim));
      kernelState.addAttribute(
          "gqa", builder.getBoolAttr(selected->queryHeads != selected->kvHeads));
      kernelState.addAttribute("tessera.tile_q",
                               builder.getI64IntegerAttr(selected->tileQ));
      kernelState.addAttribute("tessera.tile_kv",
                               builder.getI64IntegerAttr(selected->tileKV));
      kernelState.addAttribute(
          "tessera.workgroup_size",
          builder.getI64IntegerAttr(selected->workgroupSize));
      kernelState.addAttribute("tessera.attention_recurrence",
                               builder.getStringAttr(selected->recurrence));
      kernelState.addAttribute(
          "tessera.backward_lse_policy",
          builder.getStringAttr(selected->backwardLsePolicy));
      kernelState.addAttribute(
          "tessera.backward_lse_selection",
          builder.getStringAttr(selected->backwardLseSelection));
      kernelState.addAttribute("tessera.schedule_hash",
                               builder.getStringAttr(scheduled.getArtifactHash()));
      builder.create(kernelState);

      Value result = builder.create<bufferization::ToTensorOp>(
          loc, outputType, outputBuffer);
      scheduled.getScheduled().replaceAllUsesWith(result);
      scheduled.erase();
      if (graph->use_empty())
        graph->erase();
      for (schedule::ArtifactOp artifact : matchingArtifacts)
        artifact.erase();
    }

    SmallVector<schedule::AttentionBackwardOp> scheduledBackwards;
    mod.walk([&](schedule::AttentionBackwardOp op) {
      scheduledBackwards.push_back(op);
    });
    for (schedule::AttentionBackwardOp scheduled : scheduledBackwards) {
      Operation *scheduledOperation = scheduled.getOperation();
      Operation *graph = scheduledOperation->getOperand(0).getDefiningOp();
      if (!graph || graph->getName().getStringRef() !=
                        "tessera_attn.backward" ||
          graph->getNumOperands() != 5 || graph->getNumResults() != 3 ||
          scheduledOperation->getOperand(1).getDefiningOp() != graph ||
          scheduledOperation->getOperand(2).getDefiningOp() != graph) {
        scheduled.emitError(
            "E2E-REAL-5B requires all dQ/dK/dV subjects from one retained "
            "canonical Graph attention backward op");
        return signalPassFailure();
      }
      FailureOr<AttentionBackwardSchedule> selected =
          getAttentionBackwardSchedule(graph);
      auto artifactHash =
          scheduledOperation->getAttrOfType<StringAttr>("artifact_hash");
      if (failed(selected) || !artifactHash ||
          attentionBackwardScheduleDigest(*selected) !=
              artifactHash.getValue()) {
        scheduled.emitError(
            "scheduled decision does not match the retained attention "
            "backward contract");
        return signalPassFailure();
      }
      auto stringAttr = [&](StringRef name) -> StringRef {
        auto attr = scheduledOperation->getAttrOfType<StringAttr>(name);
        return attr ? attr.getValue() : StringRef();
      };
      auto integerAttr = [&](StringRef name) -> std::optional<int64_t> {
        auto attr = scheduledOperation->getAttrOfType<IntegerAttr>(name);
        if (!attr)
          return std::nullopt;
        return attr.getInt();
      };
      auto floatAttr = [&](StringRef name) -> std::optional<double> {
        auto attr = scheduledOperation->getAttrOfType<FloatAttr>(name);
        if (!attr)
          return std::nullopt;
        return attr.getValueAsDouble();
      };
      auto boolAttr = [&](StringRef name) -> std::optional<bool> {
        auto attr = scheduledOperation->getAttrOfType<BoolAttr>(name);
        if (!attr)
          return std::nullopt;
        return attr.getValue();
      };
      auto reductionOrder = scheduledOperation->getAttrOfType<DenseI64ArrayAttr>(
          "reduction_order");
      bool altered =
          stringAttr("arch") != selected->arch ||
          stringAttr("storage") != selected->storage ||
          stringAttr("accum") != selected->accum ||
          floatAttr("scale") != selected->scale ||
          boolAttr("causal") != selected->causal ||
          boolAttr("bias") != selected->bias ||
          integerAttr("window_left") != selected->windowLeft ||
          integerAttr("window_right") != selected->windowRight ||
          floatAttr("softcap") != selected->softcap ||
          floatAttr("dropout_p") != selected->dropoutP ||
          integerAttr("dropout_seed") != selected->dropoutSeed ||
          integerAttr("query_block") != selected->queryBlock ||
          integerAttr("key_block") != selected->keyBlock ||
          integerAttr("split_count") != selected->splitCount ||
          !reductionOrder || reductionOrder.size() != 2 ||
          reductionOrder[0] != 0 || reductionOrder[1] != 1 ||
          integerAttr("workspace_bytes") != selected->workspaceBytes ||
          integerAttr("workgroup_size") != selected->workgroupSize ||
          stringAttr("recurrence") != selected->recurrence ||
          stringAttr("lse_checkpoint_policy") !=
              selected->lseCheckpointPolicy ||
          stringAttr("lse_checkpoint_selection") !=
              selected->lseCheckpointSelection;
      if (altered) {
        scheduled.emitError(
            "scheduled attention backward policy was altered after hashing");
        return signalPassFailure();
      }
      auto graphDigest =
          graph->getAttrOfType<StringAttr>("schedule.artifact_hash");
      SmallVector<schedule::ArtifactOp> matchingArtifacts;
      mod.walk([&](schedule::ArtifactOp artifact) {
        if (artifact.getHash() == artifactHash.getValue())
          matchingArtifacts.push_back(artifact);
      });
      if (!graphDigest || graphDigest.getValue() != artifactHash.getValue() ||
          matchingArtifacts.size() != 1) {
        scheduled.emitError(
            "requires exactly one matching Graph hash and schedule.artifact");
        return signalPassFailure();
      }

      Location loc = scheduled.getLoc();
      builder.setInsertionPoint(scheduled);
      auto pointerType = LLVM::LLVMPointerType::get(&getContext());
      auto toPointer = [&](Value tensor) -> Value {
        auto type = cast<RankedTensorType>(tensor.getType());
        auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
        Value buffer = builder.create<bufferization::ToBufferOp>(
            loc, memrefType, tensor);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer =
            builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), index);
        return builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer);
      };
      auto allocatePointer = [&](RankedTensorType type) {
        auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
        Value buffer = builder.create<memref::AllocOp>(loc, memrefType);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer =
            builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), index);
        Value pointer =
            builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer);
        return std::make_pair(buffer, pointer);
      };
      SmallVector<Value> operands;
      for (Value input : graph->getOperands().take_front(4))
        operands.push_back(toPointer(input));
      if (selected->bias)
        operands.push_back(toPointer(graph->getOperand(4)));
      if (selected->lseCheckpointSelection == "saved") {
        auto lseType = RankedTensorType::get(
            {selected->batch, selected->queryHeads, selected->queryRows},
            builder.getF32Type());
        operands.push_back(allocatePointer(lseType).second);
      }

      SmallVector<Value> outputBuffers;
      for (Type resultType : graph->getResultTypes()) {
        auto [buffer, pointer] =
            allocatePointer(cast<RankedTensorType>(resultType));
        outputBuffers.push_back(buffer);
        operands.push_back(pointer);
      }
      for (int64_t dimension :
           {selected->batch, selected->queryHeads, selected->kvHeads,
            selected->queryRows, selected->keyRows, selected->headDim,
            selected->valueDim})
        operands.push_back(
            builder.create<arith::ConstantIntOp>(loc, dimension, 64));

      OperationState kernelState(loc, "tile.attention_backward_kernel");
      kernelState.addOperands(operands);
      kernelState.addAttribute("storage",
                               builder.getStringAttr(selected->storage));
      kernelState.addAttribute("accum", builder.getStringAttr("f32"));
      kernelState.addAttribute("scale", builder.getF32FloatAttr(selected->scale));
      kernelState.addAttribute("causal", builder.getBoolAttr(selected->causal));
      kernelState.addAttribute("bias", builder.getBoolAttr(selected->bias));
      kernelState.addAttribute("window_left",
                               builder.getI64IntegerAttr(selected->windowLeft));
      kernelState.addAttribute("window_right",
                               builder.getI64IntegerAttr(selected->windowRight));
      kernelState.addAttribute("softcap",
                               builder.getF32FloatAttr(selected->softcap));
      kernelState.addAttribute("dropout_p",
                               builder.getF32FloatAttr(selected->dropoutP));
      kernelState.addAttribute("dropout_seed",
                               builder.getI64IntegerAttr(selected->dropoutSeed));
      kernelState.addAttribute("head_dim",
                               builder.getI64IntegerAttr(selected->headDim));
      kernelState.addAttribute("value_dim",
                               builder.getI64IntegerAttr(selected->valueDim));
      kernelState.addAttribute(
          "gqa", builder.getBoolAttr(selected->queryHeads != selected->kvHeads));
      kernelState.addAttribute("lse_checkpoint",
                               builder.getStringAttr(
                                   selected->lseCheckpointSelection));
      kernelState.addAttribute("route",
                               builder.getStringAttr(
                                   "deterministic_split_reduced"));
      kernelState.addAttribute("deterministic", builder.getBoolAttr(true));
      kernelState.addAttribute(
          "workspace_bytes",
          builder.getI64IntegerAttr(selected->workspaceBytes));
      kernelState.addAttribute("workspace_owner",
                               builder.getStringAttr("program_launch"));
      kernelState.addAttribute("split_count",
                               builder.getI64IntegerAttr(selected->splitCount));
      kernelState.addAttribute("reduction_order",
                               builder.getDenseI64ArrayAttr({0, 1}));
      kernelState.addAttribute("query_block",
                               builder.getI64IntegerAttr(selected->queryBlock));
      kernelState.addAttribute("key_block",
                               builder.getI64IntegerAttr(selected->keyBlock));
      kernelState.addAttribute(
          "loop_order",
          builder.getStrArrayAttr({"forward", "pre", "dkdv_split",
                                   "dkdv_reduce", "dq"}));
      kernelState.addAttribute(
          "tessera.attention_backward_recurrence",
          builder.getStringAttr(selected->recurrence));
      kernelState.addAttribute(
          "tessera.lse_checkpoint_policy",
          builder.getStringAttr(selected->lseCheckpointPolicy));
      kernelState.addAttribute(
          "tessera.workgroup_size",
          builder.getI64IntegerAttr(selected->workgroupSize));
      kernelState.addAttribute("tessera.schedule_hash", artifactHash);
      builder.create(kernelState);

      for (auto [result, buffer, resultType] :
           llvm::zip_equal(scheduledOperation->getResults(), outputBuffers,
                           graph->getResultTypes())) {
        Value tensor = builder.create<bufferization::ToTensorOp>(
            loc, cast<RankedTensorType>(resultType), buffer);
        result.replaceAllUsesWith(tensor);
      }
      scheduled.erase();
      if (graph->use_empty())
        graph->erase();
      for (schedule::ArtifactOp artifact : matchingArtifacts)
        artifact.erase();
    }

    SmallVector<schedule::ESLowRankCorrectionOp> scheduledES;
    mod.walk([&](schedule::ESLowRankCorrectionOp op) {
      scheduledES.push_back(op);
    });
    for (schedule::ESLowRankCorrectionOp scheduled : scheduledES) {
      std::string payloadHash = llvm::toHex(
          llvm::SHA256::hash(
              llvm::arrayRefFromStringRef(scheduled.getLineagePayload())),
          /*LowerCase=*/true);
      if (payloadHash != scheduled.getArtifactHash()) {
        scheduled.emitError("ES lineage payload does not match artifact_hash");
        return signalPassFailure();
      }
      Operation *graph = scheduled.getSubject().getDefiningOp();
      if (!graph || graph->getName().getStringRef() !=
                        "tessera.es_low_rank_correction" ||
          graph->getNumOperands() != 3) {
        scheduled.emitError("requires the retained Graph ES correction");
        return signalPassFailure();
      }
      SmallVector<schedule::ArtifactOp> matchingArtifacts;
      mod.walk([&](schedule::ArtifactOp artifact) {
        if (artifact.getHash() == scheduled.getArtifactHash())
          matchingArtifacts.push_back(artifact);
      });
      if (matchingArtifacts.size() != 1) {
        scheduled.emitError("requires exactly one matching schedule.artifact");
        return signalPassFailure();
      }
      auto outputType =
          dyn_cast<RankedTensorType>(scheduled.getScheduled().getType());
      if (!outputType || !outputType.hasStaticShape() ||
          !outputType.getElementType().isF32()) {
        scheduled.emitError("initial ES lowering requires a static f32 result");
        return signalPassFailure();
      }

      Location loc = scheduled.getLoc();
      builder.setInsertionPoint(scheduled);
      auto pointerType = LLVM::LLVMPointerType::get(&getContext());
      auto toPointer = [&](Value tensor) -> FailureOr<Value> {
        auto type = dyn_cast<RankedTensorType>(tensor.getType());
        if (!type || !type.hasStaticShape()) return failure();
        auto memref = MemRefType::get(type.getShape(), type.getElementType());
        Value buffer = builder.create<bufferization::ToBufferOp>(loc, memref, tensor);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), index);
        return builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer)
            .getResult();
      };
      SmallVector<Value> operands;
      for (Value input : graph->getOperands()) {
        FailureOr<Value> pointer = toPointer(input);
        if (failed(pointer)) {
          scheduled.emitError("requires static ranked x/member/key inputs");
          return signalPassFailure();
        }
        operands.push_back(*pointer);
      }
      auto outputMemref =
          MemRefType::get(outputType.getShape(), outputType.getElementType());
      Value outputBuffer = builder.create<memref::AllocOp>(loc, outputMemref);
      Value outputIndex = builder.create<memref::ExtractAlignedPointerAsIndexOp>(
          loc, outputBuffer);
      Value outputInteger = builder.create<arith::IndexCastOp>(
          loc, builder.getI64Type(), outputIndex);
      operands.push_back(
          builder.create<LLVM::IntToPtrOp>(loc, pointerType, outputInteger));
      for (int64_t dimension :
           {scheduled.getPopulation(), scheduled.getRowsPerMember(),
            scheduled.getInDim(), scheduled.getOutDim()})
        operands.push_back(
            builder.create<arith::ConstantIntOp>(loc, dimension, 64));

      OperationState kernelState(loc, "tile.es_low_rank_correction_kernel");
      kernelState.addOperands(operands);
      kernelState.addAttribute("population", scheduled.getPopulationAttr());
      kernelState.addAttribute("rows_per_member",
                               scheduled.getRowsPerMemberAttr());
      kernelState.addAttribute("in_dim", scheduled.getInDimAttr());
      kernelState.addAttribute("out_dim", scheduled.getOutDimAttr());
      for (StringRef name : {"arch", "rank", "epoch", "sigma", "antithetic",
                             "score", "rng_algorithm", "rng_version", "storage",
                             "accum"})
        kernelState.addAttribute(name, scheduled->getAttr(name));
      kernelState.addAttribute("tessera.workgroup_size",
                               scheduled.getWorkgroupSizeAttr());
      kernelState.addAttribute("tessera.schedule_hash",
                               scheduled.getArtifactHashAttr());
      builder.create(kernelState);

      Value result = builder.create<bufferization::ToTensorOp>(
          loc, outputType, outputBuffer);
      scheduled.getScheduled().replaceAllUsesWith(result);
      scheduled.erase();
      if (graph->use_empty()) graph->erase();
      for (schedule::ArtifactOp artifact : matchingArtifacts) artifact.erase();
    }

    SmallVector<schedule::SolverIFTOp> scheduledIFTs;
    mod.walk([&](schedule::SolverIFTOp op) { scheduledIFTs.push_back(op); });
    for (schedule::SolverIFTOp scheduled : scheduledIFTs) {
      std::string payloadHash = llvm::toHex(
          llvm::SHA256::hash(
              llvm::arrayRefFromStringRef(scheduled.getLineagePayload())),
          /*LowerCase=*/true);
      if (payloadHash != scheduled.getArtifactHash()) {
        scheduled.emitError("solver IFT lineage payload does not match artifact_hash");
        return signalPassFailure();
      }
      SmallVector<schedule::ArtifactOp> matchingArtifacts;
      mod.walk([&](schedule::ArtifactOp artifact) {
        if (artifact.getHash() == scheduled.getArtifactHash())
          matchingArtifacts.push_back(artifact);
      });
      if (matchingArtifacts.size() != 1) {
        scheduled.emitError("requires exactly one matching schedule.artifact");
        return signalPassFailure();
      }
      auto tensorType =
          dyn_cast<RankedTensorType>(scheduled.getParameter().getType());
      if (!tensorType || !tensorType.hasStaticShape() ||
          !tensorType.getElementType().isF32()) {
        scheduled.emitError("initial solver IFT lowering requires static f32 tensors");
        return signalPassFailure();
      }
      if (scheduled.getSolution().getType() != tensorType ||
          scheduled.getCotangent().getType() != tensorType) {
        scheduled.emitError("solver IFT operands must have one static f32 type");
        return signalPassFailure();
      }

      Location loc = scheduled.getLoc();
      builder.setInsertionPoint(scheduled);
      auto pointerType = LLVM::LLVMPointerType::get(&getContext());
      auto toPointer = [&](Value tensor) -> Value {
        auto memrefType = MemRefType::get(
            tensorType.getShape(), tensorType.getElementType());
        Value buffer = builder.create<bufferization::ToBufferOp>(
            loc, memrefType, tensor);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), index);
        return builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer);
      };
      auto allocatePointer = [&]() {
        auto memrefType = MemRefType::get(
            tensorType.getShape(), tensorType.getElementType());
        Value buffer = builder.create<memref::AllocOp>(loc, memrefType);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), index);
        Value pointer =
            builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer);
        return std::make_pair(buffer, pointer);
      };

      SmallVector<Value> operands;
      for (Value input : {scheduled.getParameter(), scheduled.getSolution(),
                          scheduled.getCotangent()})
        operands.push_back(toPointer(input));
      SmallVector<Value> outputBuffers;
      for (int i = 0; i < 3; ++i) {
        auto [buffer, pointer] = allocatePointer();
        outputBuffers.push_back(buffer);
        operands.push_back(pointer);
      }
      operands.push_back(builder.create<arith::ConstantIntOp>(
          loc, tensorType.getNumElements(), 64));

      OperationState kernelState(loc, "tile.solver_ift_kernel");
      kernelState.addOperands(operands);
      kernelState.addAttribute("arch", scheduled.getArchAttr());
      kernelState.addAttribute("residual_model",
                               scheduled.getResidualModelAttr());
      kernelState.addAttribute("residual_digest",
                               scheduled.getResidualDigestAttr());
      kernelState.addAttribute("linear_solver",
                               scheduled.getLinearSolverAttr());
      kernelState.addAttribute("product_mode",
                               scheduled.getProductModeAttr());
      kernelState.addAttribute("transpose", scheduled.getTransposeAttr());
      kernelState.addAttribute("wrt", scheduled.getWrtAttr());
      kernelState.addAttribute("adjoint_scale",
                               scheduled.getAdjointScaleAttr());
      kernelState.addAttribute("storage", scheduled.getStorageAttr());
      kernelState.addAttribute("accum", scheduled.getAccumAttr());
      kernelState.addAttribute("tessera.workgroup_size",
                               scheduled.getWorkgroupSizeAttr());
      kernelState.addAttribute("tessera.schedule_hash",
                               scheduled.getArtifactHashAttr());
      builder.create(kernelState);

      for (auto [result, buffer] :
           llvm::zip_equal(scheduled.getResults(), outputBuffers)) {
        Value tensor = builder.create<bufferization::ToTensorOp>(
            loc, tensorType, buffer);
        result.replaceAllUsesWith(tensor);
      }
      scheduled.erase();
      for (schedule::ArtifactOp artifact : matchingArtifacts)
        artifact.erase();
    }

    SmallVector<schedule::LionVJPOp> scheduledLions;
    mod.walk([&](schedule::LionVJPOp op) { scheduledLions.push_back(op); });
    for (schedule::LionVJPOp scheduled : scheduledLions) {
      std::string payloadHash = llvm::toHex(
          llvm::SHA256::hash(
              llvm::arrayRefFromStringRef(scheduled.getLineagePayload())),
          /*LowerCase=*/true);
      if (payloadHash != scheduled.getArtifactHash()) {
        scheduled.emitError(
            "Lion VJP lineage payload does not match artifact_hash");
        return signalPassFailure();
      }
      SmallVector<schedule::ArtifactOp> matchingArtifacts;
      mod.walk([&](schedule::ArtifactOp artifact) {
        if (artifact.getHash() == scheduled.getArtifactHash())
          matchingArtifacts.push_back(artifact);
      });
      if (matchingArtifacts.size() != 1) {
        scheduled.emitError(
            "requires exactly one matching schedule.artifact");
        return signalPassFailure();
      }
      auto tensorType = dyn_cast<RankedTensorType>(scheduled.getParameter().getType());
      if (!tensorType || !tensorType.hasStaticShape() ||
          !tensorType.getElementType().isF32()) {
        scheduled.emitError("initial Lion VJP lowering requires static f32 tensors");
        return signalPassFailure();
      }
      for (Value input : {scheduled.getGradient(), scheduled.getMoment(),
                          scheduled.getDparameter(), scheduled.getDmoment()}) {
        if (input.getType() != tensorType) {
          scheduled.emitError("Lion VJP inputs must have one static f32 type");
          return signalPassFailure();
        }
      }

      Location loc = scheduled.getLoc();
      builder.setInsertionPoint(scheduled);
      auto pointerType = LLVM::LLVMPointerType::get(&getContext());
      auto toPointer = [&](Value tensor) -> Value {
        auto memrefType = MemRefType::get(
            tensorType.getShape(), tensorType.getElementType());
        Value buffer = builder.create<bufferization::ToBufferOp>(
            loc, memrefType, tensor);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), index);
        return builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer);
      };
      auto allocatePointer = [&]() {
        auto memrefType = MemRefType::get(
            tensorType.getShape(), tensorType.getElementType());
        Value buffer = builder.create<memref::AllocOp>(loc, memrefType);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), index);
        Value pointer =
            builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer);
        return std::make_pair(buffer, pointer);
      };

      SmallVector<Value> operands;
      for (Value input : {scheduled.getParameter(), scheduled.getGradient(),
                          scheduled.getMoment(), scheduled.getDparameter(),
                          scheduled.getDmoment()})
        operands.push_back(toPointer(input));
      SmallVector<Value> outputBuffers;
      for (int i = 0; i < 3; ++i) {
        auto [buffer, pointer] = allocatePointer();
        outputBuffers.push_back(buffer);
        operands.push_back(pointer);
      }
      int64_t elements = tensorType.getNumElements();
      operands.push_back(
          builder.create<arith::ConstantIntOp>(loc, elements, 64));

      OperationState kernelState(loc, "tile.training_kernel");
      kernelState.addOperands(operands);
      kernelState.addAttribute("family", builder.getStringAttr("lion_vjp"));
      kernelState.addAttribute("storage", builder.getStringAttr("f32"));
      kernelState.addAttribute("arch", scheduled.getArchAttr());
      kernelState.addAttribute("learning_rate", scheduled.getLearningRateAttr());
      kernelState.addAttribute("beta2", scheduled.getBeta2Attr());
      kernelState.addAttribute("weight_decay", scheduled.getWeightDecayAttr());
      kernelState.addAttribute("derivative_policy",
                               scheduled.getDerivativePolicyAttr());
      kernelState.addAttribute("mutation_mode", scheduled.getMutationModeAttr());
      kernelState.addAttribute("alias_policy", scheduled.getAliasPolicyAttr());
      kernelState.addAttribute("state_transition",
                               scheduled.getStateTransitionAttr());
      kernelState.addAttribute("ordered_writes", scheduled.getOrderedWritesAttr());
      kernelState.addAttribute("tessera.workgroup_size",
                               scheduled.getWorkgroupSizeAttr());
      kernelState.addAttribute("tessera.schedule_hash",
                               scheduled.getArtifactHashAttr());
      builder.create(kernelState);

      for (auto [result, buffer] :
           llvm::zip_equal(scheduled.getResults(), outputBuffers)) {
        Value tensor = builder.create<bufferization::ToTensorOp>(
            loc, tensorType, buffer);
        result.replaceAllUsesWith(tensor);
      }
      scheduled.erase();
      for (schedule::ArtifactOp artifact : matchingArtifacts)
        artifact.erase();
    }

    SmallVector<schedule::OptimizerVJPOp> scheduledOptimizers;
    mod.walk([&](schedule::OptimizerVJPOp op) {
      scheduledOptimizers.push_back(op);
    });
    for (schedule::OptimizerVJPOp scheduled : scheduledOptimizers) {
      std::string payloadHash = llvm::toHex(
          llvm::SHA256::hash(
              llvm::arrayRefFromStringRef(scheduled.getLineagePayload())),
          /*LowerCase=*/true);
      if (payloadHash != scheduled.getArtifactHash()) {
        scheduled.emitError(
            "optimizer VJP lineage payload does not match artifact_hash");
        return signalPassFailure();
      }
      SmallVector<schedule::ArtifactOp> matchingArtifacts;
      mod.walk([&](schedule::ArtifactOp artifact) {
        if (artifact.getHash() == scheduled.getArtifactHash())
          matchingArtifacts.push_back(artifact);
      });
      if (matchingArtifacts.size() != 1) {
        scheduled.emitError("requires exactly one matching schedule.artifact");
        return signalPassFailure();
      }
      auto tensorType =
          dyn_cast<RankedTensorType>(scheduled.getInputs().front().getType());
      if (!tensorType || !tensorType.hasStaticShape() ||
          !tensorType.getElementType().isF32()) {
        scheduled.emitError(
            "initial optimizer VJP lowering requires static f32 tensors");
        return signalPassFailure();
      }

      Location loc = scheduled.getLoc();
      builder.setInsertionPoint(scheduled);
      auto pointerType = LLVM::LLVMPointerType::get(&getContext());
      auto toPointer = [&](Value tensor) -> Value {
        auto memrefType =
            MemRefType::get(tensorType.getShape(), tensorType.getElementType());
        Value buffer = builder.create<bufferization::ToBufferOp>(
            loc, memrefType, tensor);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), index);
        return builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer);
      };
      auto allocatePointer = [&]() {
        auto memrefType =
            MemRefType::get(tensorType.getShape(), tensorType.getElementType());
        Value buffer = builder.create<memref::AllocOp>(loc, memrefType);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), index);
        Value pointer =
            builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer);
        return std::make_pair(buffer, pointer);
      };

      SmallVector<Value> operands;
      for (Value input : scheduled.getInputs())
        operands.push_back(toPointer(input));
      SmallVector<Value> outputBuffers;
      for (Type ignored : scheduled.getResultTypes()) {
        (void)ignored;
        auto [buffer, pointer] = allocatePointer();
        outputBuffers.push_back(buffer);
        operands.push_back(pointer);
      }
      operands.push_back(builder.create<arith::ConstantIntOp>(
          loc, tensorType.getNumElements(), 64));

      OperationState kernelState(loc, "tile.training_kernel");
      kernelState.addOperands(operands);
      kernelState.addAttribute("family", builder.getStringAttr("optimizer_vjp"));
      kernelState.addAttribute("optimizer", scheduled.getOptimizerAttr());
      kernelState.addAttribute("storage", builder.getStringAttr("f32"));
      kernelState.addAttribute("arch", scheduled.getArchAttr());
      kernelState.addAttribute("learning_rate", scheduled.getLearningRateAttr());
      kernelState.addAttribute("beta1", scheduled.getBeta1Attr());
      kernelState.addAttribute("beta2", scheduled.getBeta2Attr());
      kernelState.addAttribute("epsilon", scheduled.getEpsilonAttr());
      kernelState.addAttribute("momentum", scheduled.getMomentumAttr());
      kernelState.addAttribute("weight_decay", scheduled.getWeightDecayAttr());
      kernelState.addAttribute("step", scheduled.getStepAttr());
      kernelState.addAttribute("nesterov", scheduled.getNesterovAttr());
      kernelState.addAttribute("mutation_mode", scheduled.getMutationModeAttr());
      kernelState.addAttribute("alias_policy", scheduled.getAliasPolicyAttr());
      kernelState.addAttribute("state_transition",
                               scheduled.getStateTransitionAttr());
      kernelState.addAttribute("ordered_writes",
                               scheduled.getOrderedWritesAttr());
      kernelState.addAttribute("tessera.workgroup_size",
                               scheduled.getWorkgroupSizeAttr());
      kernelState.addAttribute("tessera.schedule_hash",
                               scheduled.getArtifactHashAttr());
      builder.create(kernelState);

      for (auto [result, buffer] :
           llvm::zip_equal(scheduled.getResults(), outputBuffers)) {
        Value tensor = builder.create<bufferization::ToTensorOp>(
            loc, tensorType, buffer);
        result.replaceAllUsesWith(tensor);
      }
      scheduled.erase();
      for (schedule::ArtifactOp artifact : matchingArtifacts)
        artifact.erase();
    }

    SmallVector<schedule::AdafactorVJPOp> scheduledAdafactors;
    mod.walk([&](schedule::AdafactorVJPOp op) {
      scheduledAdafactors.push_back(op);
    });
    for (schedule::AdafactorVJPOp scheduled : scheduledAdafactors) {
      std::string payloadHash = llvm::toHex(
          llvm::SHA256::hash(
              llvm::arrayRefFromStringRef(scheduled.getLineagePayload())),
          /*LowerCase=*/true);
      if (payloadHash != scheduled.getArtifactHash()) {
        scheduled.emitError(
            "Adafactor VJP lineage payload does not match artifact_hash");
        return signalPassFailure();
      }
      SmallVector<schedule::ArtifactOp> matchingArtifacts;
      mod.walk([&](schedule::ArtifactOp artifact) {
        if (artifact.getHash() == scheduled.getArtifactHash())
          matchingArtifacts.push_back(artifact);
      });
      if (matchingArtifacts.size() != 1) {
        scheduled.emitError(
            "requires exactly one matching schedule.artifact");
        return signalPassFailure();
      }
      auto parameterType =
          dyn_cast<RankedTensorType>(scheduled.getInputs()[0].getType());
      if (!parameterType || !parameterType.hasStaticShape() ||
          !parameterType.getElementType().isF32()) {
        scheduled.emitError(
            "initial Adafactor VJP lowering requires static f32 tensors");
        return signalPassFailure();
      }

      Location loc = scheduled.getLoc();
      builder.setInsertionPoint(scheduled);
      auto pointerType = LLVM::LLVMPointerType::get(&getContext());
      auto toPointer = [&](Value tensor) -> Value {
        auto type = cast<RankedTensorType>(tensor.getType());
        auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
        Value buffer = builder.create<bufferization::ToBufferOp>(
            loc, memrefType, tensor);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), index);
        return builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer);
      };
      auto allocatePointer = [&](RankedTensorType type) {
        auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
        Value buffer = builder.create<memref::AllocOp>(loc, memrefType);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), index);
        Value pointer =
            builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer);
        return std::make_pair(buffer, pointer);
      };

      SmallVector<Value> operands;
      for (Value input : scheduled.getInputs())
        operands.push_back(toPointer(input));
      SmallVector<Value> outputBuffers;
      for (Type resultType : scheduled.getResultTypes()) {
        auto [buffer, pointer] =
            allocatePointer(cast<RankedTensorType>(resultType));
        outputBuffers.push_back(buffer);
        operands.push_back(pointer);
      }
      if (scheduled.getTopology() == "factored") {
        int64_t columns = parameterType.getShape().back();
        int64_t rows = parameterType.getNumElements() / columns;
        operands.push_back(
            builder.create<arith::ConstantIntOp>(loc, rows, 64));
        operands.push_back(
            builder.create<arith::ConstantIntOp>(loc, columns, 64));
      } else {
        operands.push_back(builder.create<arith::ConstantIntOp>(
            loc, parameterType.getNumElements(), 64));
      }

      OperationState kernelState(loc, "tile.training_kernel");
      kernelState.addOperands(operands);
      kernelState.addAttribute("family",
                               builder.getStringAttr("adafactor_vjp"));
      kernelState.addAttribute("topology", scheduled.getTopologyAttr());
      kernelState.addAttribute("storage", builder.getStringAttr("f32"));
      kernelState.addAttribute("arch", scheduled.getArchAttr());
      kernelState.addAttribute("learning_rate", scheduled.getLearningRateAttr());
      kernelState.addAttribute("beta2", scheduled.getBeta2Attr());
      kernelState.addAttribute("epsilon", scheduled.getEpsilonAttr());
      kernelState.addAttribute("mutation_mode", scheduled.getMutationModeAttr());
      kernelState.addAttribute("alias_policy", scheduled.getAliasPolicyAttr());
      kernelState.addAttribute("state_transition",
                               scheduled.getStateTransitionAttr());
      kernelState.addAttribute("ordered_writes", scheduled.getOrderedWritesAttr());
      kernelState.addAttribute("tessera.workgroup_size",
                               scheduled.getWorkgroupSizeAttr());
      kernelState.addAttribute("tessera.schedule_hash",
                               scheduled.getArtifactHashAttr());
      builder.create(kernelState);

      for (auto [result, buffer, type] : llvm::zip_equal(
               scheduled.getResults(), outputBuffers,
               scheduled.getResultTypes())) {
        Value tensor = builder.create<bufferization::ToTensorOp>(
            loc, cast<RankedTensorType>(type), buffer);
        result.replaceAllUsesWith(tensor);
      }
      scheduled.erase();
      for (schedule::ArtifactOp artifact : matchingArtifacts)
        artifact.erase();
    }

    SmallVector<schedule::SequenceMixerBackwardOp> scheduledMixers;
    mod.walk([&](schedule::SequenceMixerBackwardOp op) {
      scheduledMixers.push_back(op);
    });
    for (schedule::SequenceMixerBackwardOp scheduled : scheduledMixers) {
      std::string payloadHash = llvm::toHex(
          llvm::SHA256::hash(
              llvm::arrayRefFromStringRef(scheduled.getLineagePayload())),
          /*LowerCase=*/true);
      if (payloadHash != scheduled.getArtifactHash()) {
        scheduled.emitError(
            "sequence-mixer lineage payload does not match artifact_hash");
        return signalPassFailure();
      }
      SmallVector<schedule::ArtifactOp> matchingArtifacts;
      mod.walk([&](schedule::ArtifactOp artifact) {
        if (artifact.getHash() == scheduled.getArtifactHash())
          matchingArtifacts.push_back(artifact);
      });
      if (matchingArtifacts.size() != 1) {
        scheduled.emitError("requires exactly one matching schedule.artifact");
        return signalPassFailure();
      }
      auto qType = dyn_cast<RankedTensorType>(scheduled.getQ().getType());
      auto vType = dyn_cast<RankedTensorType>(scheduled.getV().getType());
      if (!qType || !vType || !qType.hasStaticShape() ||
          !vType.hasStaticShape() || !qType.getElementType().isF32() ||
          !vType.getElementType().isF32()) {
        scheduled.emitError(
            "initial sequence-mixer lowering requires static f32 tensors");
        return signalPassFailure();
      }

      Location loc = scheduled.getLoc();
      builder.setInsertionPoint(scheduled);
      auto pointerType = LLVM::LLVMPointerType::get(&getContext());
      auto toPointer = [&](Value tensor) -> Value {
        auto type = cast<RankedTensorType>(tensor.getType());
        auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
        Value buffer = builder.create<bufferization::ToBufferOp>(
            loc, memrefType, tensor);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), index);
        return builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer);
      };
      auto allocatePointer = [&](RankedTensorType type) {
        auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
        Value buffer = builder.create<memref::AllocOp>(loc, memrefType);
        Value index =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
        Value integer = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), index);
        Value pointer =
            builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer);
        return std::make_pair(buffer, pointer);
      };

      SmallVector<Value> operands;
      for (Value input : {scheduled.getQ(), scheduled.getK(), scheduled.getV(),
                          scheduled.getGate(), scheduled.getBeta(),
                          scheduled.getDecay(), scheduled.getDy()})
        operands.push_back(toPointer(input));
      SmallVector<Value> outputBuffers;
      for (Type resultType : scheduled.getResultTypes()) {
        auto [buffer, pointer] =
            allocatePointer(cast<RankedTensorType>(resultType));
        outputBuffers.push_back(buffer);
        operands.push_back(pointer);
      }
      for (int64_t dimension : {qType.getShape()[0], qType.getShape()[1],
                                qType.getShape()[2], qType.getShape()[3],
                                vType.getShape()[3]})
        operands.push_back(
            builder.create<arith::ConstantIntOp>(loc, dimension, 64));

      OperationState kernelState(loc, "tile.training_kernel");
      kernelState.addOperands(operands);
      kernelState.addAttribute(
          "family", builder.getStringAttr("sequence_mixer_backward"));
      kernelState.addAttribute("mixer_family", scheduled.getFamilyAttr());
      kernelState.addAttribute("storage", builder.getStringAttr("f32"));
      kernelState.addAttribute("arch", scheduled.getArchAttr());
      kernelState.addAttribute("erase", scheduled.getEraseAttr());
      kernelState.addAttribute("chunk_size", scheduled.getChunkSizeAttr());
      kernelState.addAttribute("parallel_chunks",
                               scheduled.getParallelChunksAttr());
      kernelState.addAttribute("mutation_mode", scheduled.getMutationModeAttr());
      kernelState.addAttribute("alias_policy", scheduled.getAliasPolicyAttr());
      kernelState.addAttribute("workspace_owner",
                               scheduled.getWorkspaceOwnerAttr());
      kernelState.addAttribute("phase_order", scheduled.getPhaseOrderAttr());
      kernelState.addAttribute("tessera.workgroup_size",
                               scheduled.getWorkgroupSizeAttr());
      kernelState.addAttribute("tessera.schedule_hash",
                               scheduled.getArtifactHashAttr());
      builder.create(kernelState);

      for (auto [result, buffer, type] : llvm::zip_equal(
               scheduled.getResults(), outputBuffers,
               scheduled.getResultTypes())) {
        Value tensor = builder.create<bufferization::ToTensorOp>(
            loc, cast<RankedTensorType>(type), buffer);
        result.replaceAllUsesWith(tensor);
      }
      scheduled.erase();
      for (schedule::ArtifactOp artifact : matchingArtifacts)
        artifact.erase();
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
