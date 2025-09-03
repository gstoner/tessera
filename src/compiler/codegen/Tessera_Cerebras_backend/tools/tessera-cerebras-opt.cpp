#include "tessera/targets/cerebras/Passes.h"
#include <iostream>

#if HAVE_MLIR
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#include "CerebrasDialect.h.inc"
#include "CerebrasOps.h.inc"
#include "TTargetDialect.h.inc"
#include "TTargetOps.h.inc"

using namespace mlir;
#endif

int main(int argc, char** argv) {
#if HAVE_MLIR
  DialectRegistry registry;
  registry.insert<tessera::cerebras::CerebrasDialectImpl>();
  registry.insert<tessera::ttarget::TTarget_Dialect>();
  MLIRContext context(registry);
  context.disableMultithreading();

  auto module = parseSourceFile<ModuleOp>("-", &context);
  if (!module) {
    std::cerr << "Failed to parse MLIR from stdin.\n";
    return 1;
  }

  PassManager pm(&context);
  pm.addPass(tessera::cerebras::createLowerTTargetToCerebrasPass());
  pm.addPass(tessera::cerebras::createCerebrasCanonicalizePass());
  pm.addPass(tessera::cerebras::createCerebrasCSLEmitPass());
  if (failed(pm.run(*module))) {
    std::cerr << "Pass pipeline failed.\n";
    return 1;
  }
  module->print(llvm::outs());
  return 0;
#else
  std::cout << "tessera-cerebras-opt (stub)\n";
  return 0;
#endif
}
