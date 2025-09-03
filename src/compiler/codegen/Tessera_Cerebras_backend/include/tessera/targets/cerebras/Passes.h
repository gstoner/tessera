#pragma once
#include <memory>

namespace mlir { class Pass; }

namespace tessera {
namespace cerebras {

void registerCerebrasLoweringPasses();

// Factories (valid when MLIR is available)
std::unique_ptr<mlir::Pass> createLowerTTargetToCerebrasPass();
std::unique_ptr<mlir::Pass> createCerebrasCanonicalizePass();
std::unique_ptr<mlir::Pass> createCerebrasCSLEmitPass();

} // namespace cerebras
} // namespace tessera
