//===- RNGLegalize.cpp — legalize tessera_rng.* ops ----------------------*- C++ -*-===//
//
// Walk tessera_rng.* operations and:
//   1. Validate that a "seed" attr is present; if not, assign the default (0).
//   2. Assign a monotonically increasing rng.stream_id (integer) to each op
//      in program order, so that every RNG call has a unique stream slot.
//   3. Attach rng.backend = "philox" | "threefry" based on the module-level
//      tessera.solver_config attr (default: philox).
//
//===----------------------------------------------------------------------===//

#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct RNGLegalizePass
    : PassWrapper<RNGLegalizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RNGLegalizePass)

  Option<std::string> rngBackend{
      *this, "rng-backend",
      llvm::cl::desc("RNG backend: philox | threefry | xoshiro"),
      llvm::cl::init(std::string("philox"))};

  Option<int> defaultSeed{
      *this, "rng-seed",
      llvm::cl::desc("Default global seed for all RNG streams"),
      llvm::cl::init(0)};

  StringRef getArgument() const final { return "tessera-rng-legalize"; }
  StringRef getDescription() const final {
    return "Legalize tessera_rng.* ops: assign stream IDs and validate seeds";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    // Determine backend from module attr (overrides CLI option if present).
    StringRef backend = rngBackend;
    if (auto attr = mod->getAttrOfType<StringAttr>(
            "tessera.solver_config.rng_backend"))
      backend = attr.getValue();

    int64_t streamCounter = 0;

    mod.walk([&](Operation *op) {
      StringRef opName = op->getName().getStringRef();
      if (!opName.startswith("tessera_rng."))
        return;

      // 1. Ensure seed attr exists.
      if (!op->hasAttr("seed")) {
        op->setAttr("seed",
                    IntegerAttr::get(IntegerType::get(ctx, 64), defaultSeed));
      }

      // 2. Assign stream ID (program-order counter).
      op->setAttr("rng.stream_id",
                  IntegerAttr::get(IntegerType::get(ctx, 64), streamCounter++));

      // 3. Attach backend.
      op->setAttr("rng.backend", StringAttr::get(ctx, backend));

      // Mark as legalized.
      op->setAttr("rng.legalized", UnitAttr::get(ctx));
    });
  }
};

} // namespace

namespace tessera {
namespace passes {
std::unique_ptr<Pass> createRNGLegalizePass() {
  return std::make_unique<RNGLegalizePass>();
}
} // namespace passes
} // namespace tessera
