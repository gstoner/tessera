// Simple CLI that calls the Python-like codegen stubs via our runtime-less scaffolding.
// For now, just prints where the emitted CSL/layout would go.
#include "tessera/targets/cerebras/Passes.h"
#include <iostream>

int main(int argc, char** argv) {
  std::cout << "tessera-cerebras-codegen (stub): would emit CSL and layout.json here.\n";
  tessera::cerebras::registerCerebrasLoweringPasses();
  return 0;
}
