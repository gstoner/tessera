// ThreadgroupPipelineToApple.cpp — APPLE-PIPE-1 (2026-07-26).
//
// Apple consumption of the shared Tile-IR physical-allocation and staged
// pipeline contract. This is the Metal-native sibling of the AMD wave/LDS
// planner (`ROCMWaveLdsPipeline.cpp`) and the NVIDIA warp-specialized path:
// allocation identity and pipeline ownership are canonical SSA (`!tile.buffer`
// and `!tile.pipeline_state`), and this pass attaches the architecture-owned
// physical decisions Metal actually needs.
//
// What is Apple-owned here, and only here:
//   * threadgroup-memory placement — a 16-byte-aligned offset per allocation
//     inside one per-function arena, bounded by the device's threadgroup
//     memory capacity (a uniform 32 KiB across Apple4-Apple10 per the Metal
//     Feature Set Tables "GPU implementation limits by family");
//   * staged buffering mode — Metal has no asynchronous copy engine, so a
//     staged producer/consumer ring is the ping-pong threadgroup staging the
//     TILE-1 simdgroup materializer emits (`msl_gemm_emit.py`,
//     `double_buffer=True`). Depth > 2 has no Metal realization and is
//     rejected rather than silently narrowed.
//
// What is deliberately NOT consumed: TMA descriptors, mbarriers, and tensor
// memory are NVIDIA physical vocabulary with no Metal equivalent. Per
// Architecture Decision #21 they are rejected with a stable diagnostic naming
// the operation and the target, never silently dropped.
//
// Scope of this rung: the SSA vocabulary only. Migrating the legacy
// name-based `#tile.buffer_ref` / annotation-only `#tile.pipeline_state`
// metadata into SSA allocations is an AMD-owned planner behavior; Apple
// requires SSA identity and says so with its own diagnostic.

#include "Tessera/Target/Apple/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

using namespace mlir;

namespace tessera::apple {
namespace {

// Uniform Apple threadgroup-memory ceiling (Apple4-Apple10). Mirrors
// `_ArchDefaults.threadgroup_memory_bytes` in python/tessera/compiler/
// apple_target.py; the host runtime may report a higher SKU-specific limit,
// so this is the compile-time conservative floor.
constexpr int64_t kAppleThreadgroupCapacityBytes = 32 * 1024;

// Metal stages through threadgroup memory with a ping-pong pair; there is no
// deeper hardware-managed ring.
constexpr int64_t kAppleMaxStageDepth = 2;

constexpr int64_t kAppleThreadgroupAlignment = 16;

// Storage dtypes Apple's cooperative-matrix route actually accepts. Mirrors
// `select_apple_simdgroup_fragment` in python/tessera/compiler/apple_fragment.py:
// Apple7+ `simdgroup_matrix` takes fp16/bf16 operands and accumulates in fp32.
// The FP8/FP4/MX `MTLTensor` dtypes are macOS-27.0 SDK-gated (toolchain, not
// hardware), so no NVFP4/FP8/FP6 cooperative-matrix route exists here today.
bool isAppleMmaStorage(StringRef storage) {
  return storage == "fp16" || storage == "f16" || storage == "bf16";
}

// Read the A / B / accumulator types a `#tile.mma_desc` declares.
//
// The Apple backend deliberately does not link the Tile dialect (every other
// lowering here matches ops generically by name), so `TileMmaDescAttr`'s typed
// accessors are unavailable and the printed form is inspected instead. Narrowly
// scoped on purpose: it reads only the three `key = "value"` fields it needs,
// and returns empty when it cannot find them, which callers must treat as
// "cannot prove a route exists" rather than as permission.
SmallVector<std::string> mmaDescriptorTypes(Attribute mma) {
  std::string printed;
  llvm::raw_string_ostream stream(printed);
  mma.print(stream);
  StringRef text = stream.str();

  SmallVector<std::string> found;
  for (StringRef key : {"a = \"", "b = \"", "acc = \""}) {
    size_t at = text.find(key);
    if (at == StringRef::npos)
      return {};
    StringRef rest = text.drop_front(at + key.size());
    size_t end = rest.find('"');
    if (end == StringRef::npos)
      return {};
    found.push_back(rest.take_front(end).str());
  }
  return found;
}

bool isNvidiaOnlyTileOp(StringRef name) {
  return name.starts_with("tile.tma.") || name.starts_with("tile.tma_") ||
         name.starts_with("tile.mbarrier.") ||
         name.starts_with("tile.mbarrier_") ||
         name.starts_with("tile.tmem.") || name == "tile.tcgen05.mma";
}

int64_t alignUpTo(int64_t value, int64_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

struct AppleThreadgroupPipelinePass
    : public PassWrapper<AppleThreadgroupPipelinePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AppleThreadgroupPipelinePass)

  AppleThreadgroupPipelinePass() = default;
  AppleThreadgroupPipelinePass(const AppleThreadgroupPipelinePass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "tessera-apple-threadgroup-pipeline";
  }
  StringRef getDescription() const override {
    return "Apple consumption of shared Tile !tile.buffer / "
           "!tile.pipeline_state — Metal threadgroup allocation and staged "
           "buffering mode.";
  }

  Option<int64_t> capacityBytes{
      *this, "threadgroup-capacity-bytes",
      llvm::cl::desc("Device threadgroup-memory capacity in bytes (Apple "
                     "compile-time floor is 32768)"),
      llvm::cl::init(kAppleThreadgroupCapacityBytes)};

  Option<int64_t> maxStageDepth{
      *this, "max-stage-depth",
      llvm::cl::desc("Deepest staged producer/consumer ring Metal can "
                     "realize (ping-pong = 2)"),
      llvm::cl::init(kAppleMaxStageDepth)};

  // Reject the physical vocabulary that has no Metal realization. Returns
  // false on the first rejection so the caller stops before placing memory.
  bool rejectUnsupportedVocabulary(FunctionOpInterface func) {
    bool ok = true;
    func.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (isNvidiaOnlyTileOp(name)) {
        op->emitOpError("APPLE_TILE_UNSUPPORTED_VOCABULARY: '")
            << name
            << "' is NVIDIA physical vocabulary with no Metal equivalent; "
               "apple_gpu has no TMA engine, transaction barrier, or tensor "
               "memory";
        ok = false;
        return;
      }
      if (op->hasAttr("tile.buffer_ref")) {
        op->emitOpError(
            "APPLE_THREADGROUP_LEGACY_METADATA: name-based #tile.buffer_ref "
            "is not an Apple physical allocation; thread !tile.buffer from "
            "tile.alloc so placement follows SSA identity");
        ok = false;
      }
      // A cooperative-matrix descriptor is a capability claim. Apple answers it
      // from its own fragment contract rather than inheriting whichever dtype
      // the shared descriptor happens to permit.
      //
      // The descriptor is authoritative, not `numeric_policy`. Only `mma` is
      // required by the op verifier, so a policy can be absent entirely, or can
      // disagree with the descriptor it accompanies — an int4 or FP8
      // `#tile.mma_desc` carrying an fp16 policy would otherwise launder itself
      // straight past this gate. Both the declared A/B/accumulator types and
      // the policy must name storage Apple can actually route.
      if (Attribute mma = op->getAttr("mma")) {
        SmallVector<std::string> claimed = mmaDescriptorTypes(mma);
        if (claimed.empty()) {
          op->emitOpError("APPLE_MMA_STORAGE_UNSUPPORTED: apple_gpu cannot "
                          "read the operand types of this cooperative-matrix "
                          "descriptor, so it cannot prove a simdgroup_matrix "
                          "route exists");
          ok = false;
          return;
        }
        for (const std::string &type : claimed) {
          // The accumulator is fp32 by the simdgroup_matrix contract; operands
          // must be fp16/bf16.
          if (isAppleMmaStorage(type) || type == "f32" || type == "fp32")
            continue;
          op->emitOpError("APPLE_MMA_STORAGE_UNSUPPORTED: apple_gpu has no "
                          "cooperative-matrix route for '")
              << type
              << "'; simdgroup_matrix accepts fp16/bf16 operands with fp32 "
                 "accumulation, and the FP8/FP4/MX MTLTensor dtypes are "
                 "macOS-27 SDK-gated";
          ok = false;
          return;
        }
        auto policy = op->getAttrOfType<DictionaryAttr>("numeric_policy");
        auto storage =
            policy ? policy.getAs<StringAttr>("storage") : StringAttr();
        if (storage && !isAppleMmaStorage(storage.getValue())) {
          op->emitOpError("APPLE_MMA_STORAGE_UNSUPPORTED: apple_gpu has no "
                          "cooperative-matrix route for '")
              << storage.getValue()
              << "' storage; simdgroup_matrix accepts fp16/bf16 with fp32 "
                 "accumulation, and the FP8/FP4/MX MTLTensor dtypes are "
                 "macOS-27 SDK-gated";
          ok = false;
        }
      }
    });
    return ok;
  }

  // Place one function's threadgroup allocations into a single arena.
  bool placeAllocations(FunctionOpInterface func) {
    int64_t arenaBytes = 0;
    bool ok = true;
    func.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "tile.alloc")
        return;
      auto space = op->getAttrOfType<StringAttr>("space");
      auto bytes = op->getAttrOfType<IntegerAttr>("bytes");
      if (!space || !bytes) {
        op->emitOpError("APPLE_THREADGROUP_MALFORMED_ALLOC: tile.alloc "
                        "requires 'space' and 'bytes'");
        ok = false;
        return;
      }
      // `smem` is the portable Tile spelling for the shared/threadgroup
      // address space. `tmem` and `gmem` are not threadgroup allocations and
      // this rung owns neither.
      if (space.getValue() != "smem") {
        op->emitOpError("APPLE_THREADGROUP_SPACE_UNSUPPORTED: apple_gpu "
                        "places only 'smem' allocations in threadgroup "
                        "memory; got '")
            << space.getValue() << "'";
        ok = false;
        return;
      }
      int64_t size = bytes.getInt();
      int64_t offset = alignUpTo(arenaBytes, kAppleThreadgroupAlignment);
      int64_t next = offset + size;
      if (next > capacityBytes) {
        op->emitOpError("APPLE_THREADGROUP_MEMORY_EXCEEDED: ")
            << next << " bytes exceeds the " << capacityBytes.getValue()
            << "-byte threadgroup capacity of this Apple target";
        ok = false;
        return;
      }
      OpBuilder b(op);
      op->setAttr("tessera_apple.address_space", b.getStringAttr("threadgroup"));
      op->setAttr("tessera_apple.threadgroup_offset", b.getI64IntegerAttr(offset));
      op->setAttr("tessera_apple.threadgroup_bytes", b.getI64IntegerAttr(size));
      op->setAttr("tessera_apple.threadgroup_capacity_bytes",
                  b.getI64IntegerAttr(capacityBytes));
      arenaBytes = next;
    });
    if (ok && arenaBytes > 0) {
      OpBuilder b(func);
      func->setAttr("tessera_apple.threadgroup_arena_bytes",
                    b.getI64IntegerAttr(arenaBytes));
    }
    return ok;
  }

  // Claim the staged producer/consumer ring as Metal ping-pong staging, and
  // require every advance to carry real SSA pipeline state.
  // Resolve a value to the staged state it ultimately came from, following
  // loop-carried threading.
  //
  // A ring carried by `scf.for` reaches its advance as a *region iter_arg*, not
  // as the `tile.pipeline_init` result: the canonical GEMM and streaming
  // attention both emit `tile.pipeline_advance %arg8` where `%arg8` is the K
  // loop's block argument. Membership-testing the raw operand would report
  // APPLE_STAGE_UNROOTED_ADVANCE for state that is legitimately threaded — and
  // would reject the very shared contracts this pass exists to consume.
  //
  // So a block argument is mapped back to the matching init operand of its
  // owning loop, following chains through nested loops.
  static Value resolveThroughIterArgs(Value value) {
    unsigned guard = 0;
    while (guard++ < 16) {
      auto arg = llvm::dyn_cast<BlockArgument>(value);
      if (!arg)
        return value;
      Block *block = arg.getOwner();
      Operation *parent = block->getParentOp();
      if (!parent || parent->getNumRegions() == 0)
        return value;
      // Region-carried loops (scf.for and friends) put induction variables
      // first, then one init operand per carried value. Line the argument up
      // with the operand that seeded it.
      unsigned carried = block->getNumArguments();
      unsigned operands = parent->getNumOperands();
      if (carried > operands)
        return value;
      unsigned lead = operands - carried; // lb/ub/step ahead of the iter_args
      unsigned index = arg.getArgNumber() + lead;
      if (index >= operands)
        return value;
      value = parent->getOperand(index);
    }
    return value;
  }

  bool claimStagedPipelines(FunctionOpInterface func) {
    bool ok = true;
    // Value -> buffering mode of the init that roots it, so an advance reports
    // the schedule its own ring actually has.
    llvm::DenseMap<Value, StringAttr> stageStates;

    func.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tile.pipeline_init") {
        auto depth = op->getAttrOfType<IntegerAttr>("depth");
        auto role = op->getAttrOfType<StringAttr>("role");
        if (!depth || !role) {
          op->emitOpError("APPLE_STAGE_MALFORMED_INIT: tile.pipeline_init "
                          "requires 'depth' and 'role'");
          ok = false;
          return;
        }
        if (depth.getInt() > maxStageDepth) {
          op->emitOpError("APPLE_STAGE_DEPTH_UNSUPPORTED: depth ")
              << depth.getInt()
              << " has no Metal realization; threadgroup staging is "
                 "ping-pong (max depth "
              << maxStageDepth.getValue() << ")";
          ok = false;
          return;
        }
        OpBuilder b(op);
        StringAttr mode =
            b.getStringAttr(depth.getInt() == 2 ? "ping_pong" : "single");
        op->setAttr("tessera_apple.stage_role", role);
        op->setAttr("tessera_apple.stage_depth", depth);
        op->setAttr("tessera_apple.stage_buffering", mode);
        for (Value result : op->getResults())
          stageStates[result] = mode;
        return;
      }
      if (name == "tile.pipeline_advance") {
        // Carry the rooted init's mode rather than assuming ping-pong: a
        // depth-1 ring whose advances claimed ping_pong would hand the emitter
        // two contradictory physical schedules for one pipeline.
        StringAttr mode;
        for (Value operand : op->getOperands()) {
          auto it = stageStates.find(resolveThroughIterArgs(operand));
          if (it != stageStates.end()) {
            mode = it->second;
            break;
          }
        }
        if (!mode) {
          op->emitOpError("APPLE_STAGE_UNROOTED_ADVANCE: tile.pipeline_advance "
                          "has no !tile.pipeline_state operand traced to an "
                          "Apple-claimed tile.pipeline_init");
          ok = false;
          return;
        }
        op->setAttr("tessera_apple.stage_buffering", mode);
        for (Value result : op->getResults())
          stageStates[result] = mode;
      }
    });
    return ok;
  }

  void runOnOperation() override {
    if (capacityBytes <= 0 || maxStageDepth <= 0) {
      getOperation().emitError(
          "APPLE_THREADGROUP_INVALID_OPTION: threadgroup-capacity-bytes and "
          "max-stage-depth must be positive");
      signalPassFailure();
      return;
    }
    bool ok = true;
    getOperation().walk([&](FunctionOpInterface func) {
      if (!rejectUnsupportedVocabulary(func)) {
        ok = false;
        return;
      }
      if (!placeAllocations(func))
        ok = false;
      if (!claimStagedPipelines(func))
        ok = false;
    });
    if (!ok)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createAppleThreadgroupPipelinePass() {
  return std::make_unique<AppleThreadgroupPipelinePass>();
}

} // namespace tessera::apple
