#include "tessera/targets/cerebras/Passes.h"
#include <cstdio>

#if HAVE_MLIR
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "CerebrasDialect.h.inc"
#include "TTargetDialect.h.inc"
#endif

namespace tessera { namespace cerebras {

void registerCerebrasLoweringPasses() {
#if HAVE_MLIR
  registerTesseraCerebrasBackendPasses();
#else
  std::puts("[tessera-cerebras] registerCerebrasLoweringPasses() (stub)");
#endif
}

void createLowerTileToCerebrasPass() {
  std::puts("[tessera-cerebras] createLowerTileToCerebrasPass() (stub)");
}

void createBankAwareVectorizePass() {
  std::puts("[tessera-cerebras] createBankAwareVectorizePass() (stub)");
}

#if HAVE_MLIR
void buildTesseraCerebrasBackendPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createLowerTTargetToCerebrasPass());
  pm.addPass(createCerebrasCanonicalizePass());
  pm.addPass(createCerebrasCSLEmitPass());
}

void registerTesseraCerebrasBackendPasses() {
  mlir::PassPipelineRegistration<> pipeline(
      "tessera-cerebras-backend",
      "Lower Tessera target IR to Cerebras CSL artifacts",
      [](mlir::OpPassManager &pm) { buildTesseraCerebrasBackendPipeline(pm); });
}

void registerTesseraCerebrasBackendDialects(mlir::DialectRegistry &registry) {
  registry.insert<tessera::cerebras::CerebrasDialectImpl>();
  registry.insert<tessera::ttarget::TTarget_Dialect>();
}
#else
void registerTesseraCerebrasBackendPasses() {}
void registerTesseraCerebrasBackendDialects(mlir::DialectRegistry &) {}
#endif

}} // namespace
