#pragma once
#include <memory>

namespace mlir {
class DialectRegistry;
class OpPassManager;
class Pass;
}

namespace tessera {
namespace cerebras {

void registerCerebrasLoweringPasses();
void buildTesseraCerebrasBackendPipeline(mlir::OpPassManager &pm);
void registerTesseraCerebrasBackendPasses();
void registerTesseraCerebrasBackendDialects(mlir::DialectRegistry &registry);

// Factories (valid when MLIR is available)
std::unique_ptr<mlir::Pass> createLowerTTargetToCerebrasPass();
std::unique_ptr<mlir::Pass> createCerebrasCanonicalizePass();
std::unique_ptr<mlir::Pass> createCerebrasCSLEmitPass();

} // namespace cerebras
} // namespace tessera
