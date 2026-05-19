//===- InitTPP.cpp - TPP dialect / pass / pipeline registration -----------===//
//
// Concrete bodies for the entry points declared in
// `include/tpp/InitTPP.h`.  Used by `tessera-opt` (when built with
// `TESSERA_HAVE_TPP`) and by the standalone `tessera-tpp-opt` driver.
//
// Each TPP pass class lives in lib/Passes/<Name>.cpp and exposes a free
// `create*Pass()` factory; this file calls `PassRegistration<>` against
// each pass class via the factory.  The pipeline alias is registered
// here too so it shows up under `--help` next to the individual passes.
//
//===----------------------------------------------------------------------===//

#include "tpp/InitTPP.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "TPPDialect.h.inc"   // declares ::tessera::tpp::TPPDialect

// Pass factories (defined in lib/Passes/*.cpp).
namespace {
std::unique_ptr<mlir::Pass> tessera_tpp_createLegalizeSpaceTimePass();
std::unique_ptr<mlir::Pass> tessera_tpp_createHaloInferPass();
std::unique_ptr<mlir::Pass> tessera_tpp_createFuseStencilTimePass();
std::unique_ptr<mlir::Pass> tessera_tpp_createAsyncPrefetchPass();
std::unique_ptr<mlir::Pass> tessera_tpp_createVectorizeTPPPass();
std::unique_ptr<mlir::Pass> tessera_tpp_createDistributeHaloPass();
std::unique_ptr<mlir::Pass> tessera_tpp_createLowerTPPToTargetIRPass();
} // namespace

// The TPP pass library declares these without a namespace (see
// lib/Passes/*.cpp).  Forward declarations to bridge into our `inline`
// trampolines above.
std::unique_ptr<mlir::Pass> createLegalizeSpaceTimePass();
std::unique_ptr<mlir::Pass> createHaloInferPass();
std::unique_ptr<mlir::Pass> createFuseStencilTimePass();
std::unique_ptr<mlir::Pass> createAsyncPrefetchPass();
std::unique_ptr<mlir::Pass> createVectorizeTPPPass();
std::unique_ptr<mlir::Pass> createDistributeHaloPass();
std::unique_ptr<mlir::Pass> createLowerTPPToTargetIRPass();

namespace {
std::unique_ptr<mlir::Pass> tessera_tpp_createLegalizeSpaceTimePass() {
  return createLegalizeSpaceTimePass();
}
std::unique_ptr<mlir::Pass> tessera_tpp_createHaloInferPass() {
  return createHaloInferPass();
}
std::unique_ptr<mlir::Pass> tessera_tpp_createFuseStencilTimePass() {
  return createFuseStencilTimePass();
}
std::unique_ptr<mlir::Pass> tessera_tpp_createAsyncPrefetchPass() {
  return createAsyncPrefetchPass();
}
std::unique_ptr<mlir::Pass> tessera_tpp_createVectorizeTPPPass() {
  return createVectorizeTPPPass();
}
std::unique_ptr<mlir::Pass> tessera_tpp_createDistributeHaloPass() {
  return createDistributeHaloPass();
}
std::unique_ptr<mlir::Pass> tessera_tpp_createLowerTPPToTargetIRPass() {
  return createLowerTPPToTargetIRPass();
}
} // namespace

namespace tessera {
namespace tpp {

void registerTPPDialect(::mlir::DialectRegistry &registry) {
  registry.insert<::tessera::tpp::TPPDialect>();
}

void registerTPPPasses() {
  // The pass classes live in anonymous namespaces inside lib/Passes/*.cpp,
  // so we can't use the type-parameterized `PassRegistration<Pass>` form
  // here.  `mlir::registerPass(...)` takes a bare allocator function and
  // reads getArgument()/getDescription() off the constructed pass for
  // command-line registration.
  static bool once = []() {
    ::mlir::registerPass(
        []() { return tessera_tpp_createLegalizeSpaceTimePass(); });
    ::mlir::registerPass(
        []() { return tessera_tpp_createHaloInferPass(); });
    ::mlir::registerPass(
        []() { return tessera_tpp_createFuseStencilTimePass(); });
    ::mlir::registerPass(
        []() { return tessera_tpp_createAsyncPrefetchPass(); });
    ::mlir::registerPass(
        []() { return tessera_tpp_createVectorizeTPPPass(); });
    ::mlir::registerPass(
        []() { return tessera_tpp_createDistributeHaloPass(); });
    ::mlir::registerPass(
        []() { return tessera_tpp_createLowerTPPToTargetIRPass(); });
    return true;
  }();
  (void)once;
}

void registerTPPPipelines() {
  static mlir::PassPipelineRegistration<> tppSpaceTime(
      "tpp-space-time",
      "Canonical TPP space-time stencil lowering pipeline.  Chains "
      "legalize-space-time → halo-infer → fuse-stencil-time → "
      "async-prefetch → vectorize → distribute-halo → lower-tpp-to-target-ir.",
      [](mlir::OpPassManager &pm) {
        pm.addPass(tessera_tpp_createLegalizeSpaceTimePass());
        pm.addPass(tessera_tpp_createHaloInferPass());
        pm.addPass(tessera_tpp_createFuseStencilTimePass());
        pm.addPass(tessera_tpp_createAsyncPrefetchPass());
        pm.addPass(tessera_tpp_createVectorizeTPPPass());
        pm.addPass(tessera_tpp_createDistributeHaloPass());
        pm.addPass(tessera_tpp_createLowerTPPToTargetIRPass());
      });
}

} // namespace tpp
} // namespace tessera
