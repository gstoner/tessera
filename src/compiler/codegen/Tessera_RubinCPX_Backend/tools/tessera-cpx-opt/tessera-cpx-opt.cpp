
//===- tessera-cpx-opt.cpp - Tessera CPX optimizer driver -----------------===//
//
// Standalone mlir-opt-style driver for the Tessera Rubin CPX backend.
// Registers:
//   • All standard MLIR dialects (for analysis passes to run on them)
//   • The `tessera.target.cpx` dialect (NVRubinCPXDialect)
//   • The CPX pass pipelines:
//       tessera-cpx-pipeline        (full: video-ingest + partition + vectorize + lower)
//       tessera-cpx-context-pipeline (partial: no video-ingest)
//   • Individual CPX passes:
//       --tessera-fuse-video-ingest
//       --tessera-partition-longcontext
//       --tessera-vectorize-nvfp4
//       --tessera-lower-kv-transport
//
// Usage (standalone build):
//   tessera-cpx-opt --pass-pipeline='tessera-cpx-pipeline' input.mlir
//   tessera-cpx-opt --tessera-partition-longcontext input.mlir
//
//===-----------------------------------------------------------------------===//

#include "tessera/Target/NVRubinCPX/NVRubinCPX.h"
#include "tessera/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;

  // Register all standard MLIR dialects (func, arith, linalg, memref, …)
  registerAllDialects(registry);

  // Register the Tessera Rubin CPX dialect
  registry.insert<tessera::target::NVRubinCPXDialect>();

  // Register all standard MLIR passes
  registerAllPasses();

  // Register CPX-specific pass pipelines and individual passes
  tessera::registerCPXPassPipelines();

  return failed(
      MlirOptMain(argc, argv, "tessera-cpx-opt — Rubin CPX backend driver\n",
                  registry));
}
