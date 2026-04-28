//===- RNGStreamAssign.cpp — finalize per-rank RNG stream IDs -----------*- C++ -*-===//
//
// Reads the module-level tessera.num_ranks attr (or --num-ranks option) and
// finalizes each tessera_rng.* op's stream ID to:
//
//   stream_id = global_seed * num_ranks + rank_offset
//
// where rank_offset is the value of rng.stream_id set by RNGLegalizePass
// (i.e., the program-order index within the local rank).  This guarantees
// that streams are:
//   - Independent across ranks (different multiplicative offsets)
//   - Deterministic given the same global seed
//
// The pass also emits a rng.num_streams module attr for downstream use.
//
//===----------------------------------------------------------------------===//

#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct RNGStreamAssignPass
    : PassWrapper<RNGStreamAssignPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RNGStreamAssignPass)

  Option<int> numRanks{
      *this, "num-ranks",
      llvm::cl::desc("Total number of parallel ranks"),
      llvm::cl::init(1)};

  Option<int64_t> globalSeed{
      *this, "global-seed",
      llvm::cl::desc("Global RNG seed"),
      llvm::cl::init(0)};

  StringRef getArgument() const final { return "tessera-rng-stream-assign"; }
  StringRef getDescription() const final {
    return "Finalize tessera_rng.* stream IDs: stream_id = "
           "global_seed * num_ranks + rank_offset";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    // Allow module attr to override CLI.
    int64_t nRanks = numRanks;
    if (auto attr = mod->getAttrOfType<IntegerAttr>("tessera.num_ranks"))
      nRanks = attr.getInt();
    if (nRanks < 1) nRanks = 1;

    int64_t seed = globalSeed;
    if (auto attr = mod->getAttrOfType<IntegerAttr>("tessera.global_seed"))
      seed = attr.getInt();

    int64_t maxStream = 0;

    mod.walk([&](Operation *op) {
      if (!op->getName().getStringRef().startswith("tessera_rng."))
        return;
      // Must have been legalized first (stream_id carries the rank-local offset).
      auto sidAttr = op->getAttrOfType<IntegerAttr>("rng.stream_id");
      if (!sidAttr)
        return;

      int64_t rankOffset = sidAttr.getInt();
      int64_t finalId = seed * nRanks + rankOffset;
      op->setAttr("rng.stream_id",
                  IntegerAttr::get(IntegerType::get(ctx, 64), finalId));
      if (finalId > maxStream)
        maxStream = finalId;
    });

    // Record total stream count on the module for the runtime.
    mod->setAttr("tessera_rng.num_streams",
                 IntegerAttr::get(IntegerType::get(ctx, 64), maxStream + 1));
  }
};

} // namespace

namespace tessera {
namespace passes {
std::unique_ptr<Pass> createRNGStreamAssignPass() {
  return std::make_unique<RNGStreamAssignPass>();
}
} // namespace passes
} // namespace tessera
