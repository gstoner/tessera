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

#include <algorithm>
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

struct FFTSchedule {
  StringRef target;
  StringRef arch;
  StringRef mode;
  StringRef radixPolicy;
  StringRef strategy;
  StringRef algorithm;
  StringRef kernelFamily;
  StringRef workspacePolicy;
  StringRef residency;
  StringRef twiddlePolicy;
  SmallVector<int64_t> inputShape;
  SmallVector<int64_t> outputShape;
  SmallVector<int64_t> radixSequence;
  int64_t axis = -1;
  int64_t length = 0;
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
      rocm ? "gfx1151_stockham_bluestein_v3" : "zen5_avx512_fft_v3";
  schedule.workgroupSize = rocm ? 256 : 1;
  schedule.inputShape.assign(input.getShape().begin(), input.getShape().end());
  schedule.outputShape.assign(output.getShape().begin(), output.getShape().end());

  auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
  schedule.axis = axisAttr ? axisAttr.getInt() : -1;
  if (schedule.axis < 0) schedule.axis += input.getRank();
  if (schedule.axis < 0 || schedule.axis >= input.getRank()) return failure();
  auto norm = op->getAttrOfType<StringAttr>("norm");
  if (norm && norm.getValue() != "backward") return failure();

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
    auto lengthAttr = op->getAttrOfType<IntegerAttr>("n");
    schedule.length =
        lengthAttr ? lengthAttr.getInt() : output.getDimSize(schedule.axis);
    if (output.getDimSize(schedule.axis) != schedule.length ||
        input.getDimSize(schedule.axis) != schedule.length / 2 + 1)
      return failure();
  } else {
    return failure();
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
  schedule.scale = schedule.inverse ? 1.0 / schedule.length : 1.0;

  if (rocm) {
    auto stages = mixedRadixSequence(schedule.length);
    if (stages) {
      schedule.strategy = "mixed_radix";
      schedule.radixSequence = *stages;
      schedule.workspaceElems = schedule.length;
      schedule.workspacePolicy = "persistent_plan_n";
    } else {
      schedule.strategy = "bluestein";
      schedule.bluesteinM = nextPowerOfTwo(2 * schedule.length - 1);
      schedule.workspaceElems = 4 * schedule.bluesteinM;
      schedule.workspacePolicy = "persistent_plan_4m";
      schedule.twiddlePolicy = "persistent_device_chirp_fft";
    }
  } else {
    auto stages = mixedRadixSequence(schedule.length);
    if (stages && preferX86MixedRadix(schedule.length, *stages)) {
      schedule.radixPolicy = "mixed_radix";
      schedule.strategy = "mixed_radix";
      schedule.algorithm = "stockham_autosort";
      schedule.radixSequence = *stages;
      schedule.workspaceElems = 2 * schedule.length;
      schedule.workspacePolicy = "thread_local_2n";
      schedule.residency = "host_thread_local_ping_pong";
    } else if (isPowerOfTwo(schedule.length)) {
      schedule.strategy = "radix2";
      for (int64_t rest = schedule.length; rest > 1; rest /= 2)
        schedule.radixSequence.push_back(2);
      schedule.workspacePolicy = "inplace_no_scratch";
    } else if (schedule.length <= 8) {
      schedule.strategy = "dft";
      schedule.workspacePolicy = "inplace_no_scratch";
    } else {
      schedule.strategy = "bluestein";
      schedule.bluesteinM = nextPowerOfTwo(2 * schedule.length - 1);
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
       Twine(schedule.batch) + ";inverse=" +
       Twine(schedule.inverse ? 1 : 0) + ";normalization=backward;scale=" +
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
  if (!x86 && !rocm)
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
                schedule.headDim % 16 != 0)))
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
  schedule.backwardLsePolicy = x86 ? "save_lse" : "gfx1151_auto_128";
  schedule.backwardLseSelection =
      x86 || schedule.queryRows >= 128 ? "saved" : "recompute";
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
  if (!x86 && !rocm)
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
                schedule.headDim % 16 != 0)))
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
  schedule.lseCheckpointPolicy = x86 ? "save_lse" : "gfx1151_auto_128";
  schedule.lseCheckpointSelection =
      x86 || std::max(schedule.queryRows, schedule.keyRows) >= 128
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
            "Zen 5 or gfx1151 with backward normalization");
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
      state.addAttribute("batch", builder.getI64IntegerAttr(selected->batch));
      state.addAttribute("inverse", builder.getBoolAttr(selected->inverse));
      state.addAttribute("normalization", builder.getStringAttr("backward"));
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
               selected->strategy + ";backward")
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
          intAttr("batch") != selected->batch ||
          boolAttr("inverse") != selected->inverse ||
          stringAttr("normalization") != "backward" ||
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
      kernelState.addAttribute("batch",
                               builder.getI64IntegerAttr(selected->batch));
      kernelState.addAttribute("inverse", builder.getBoolAttr(selected->inverse));
      kernelState.addAttribute("normalization",
                               builder.getStringAttr("backward"));
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
