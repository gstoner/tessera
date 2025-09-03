#include "tessera/targets/cerebras/Passes.h"
#include <cstdio>

namespace tessera { namespace cerebras {

void registerCerebrasLoweringPasses() {
  // In a real build, you'd call MLIR registerPass or PassRegistration<T>.
  std::puts("[tessera-cerebras] registerCerebrasLoweringPasses() (stub)");
}

void createLowerTileToCerebrasPass() {
  std::puts("[tessera-cerebras] createLowerTileToCerebrasPass() (stub)");
}

void createBankAwareVectorizePass() {
  std::puts("[tessera-cerebras] createBankAwareVectorizePass() (stub)");
}

}} // namespace
