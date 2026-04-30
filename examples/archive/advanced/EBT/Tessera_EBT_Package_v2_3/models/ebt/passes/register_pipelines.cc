#include "register_pipelines.h"

// NOTE: This file is a scaffold. In-tree, include MLIR & Tessera headers, e.g.:
// #include "mlir/Pass/Pass.h"
// #include "mlir/Pass/PassManager.h"
// #include "mlir/InitAllPasses.h"
// using namespace mlir;

namespace tessera { namespace ebt {

static void addCanonicalizePipeline(/*mlir::OpPassManager& pm*/) {
  // pm.addPass(createCSEPass());
  // pm.addPass(createCanonicalizerPass());
  // pm.addPass(createEBTNormalizeSelfVerifyPass());
  // pm.addPass(createTesseraHoistInvariantsPass());
  // pm.addPass(createTesseraInlineDecodeInitPass());
  // pm.addPass(createTesseraShapeSimplifyPass());
  // pm.addPass(createCanonicalizerPass());
  // pm.addPass(createCSEPass());
}

static void addLowerPipeline(/*mlir::OpPassManager& pm,*/ const EBTPipelineOptions& opts) {
  // pm.addPass(createEBTMaterializeLoopsPass(opts.K, opts.T));
  // pm.addPass(createLoopFusionPass(/*aggressive=*/true));
  // pm.addPass(createTesseraTilingPipelinePass(/*targets=*/{"attention","mlp","ebt.energy"}));
  // if (!opts.useJVP) pm.addPass(createTesseraAutodiffPass(/*mode=*/"vjp", /*targets=*/{"ebt.energy"}));
  // pm.addPass(createEBTSelectGradPathPass(/*preferJVP=*/opts.useJVP));
  // pm.addPass(createTesseraVectorizePass());
  // pm.addPass(createConvertVectorToLLVMPass());
  // pm.addPass(createTesseraTargetSelectPass(/*auto*/));
  // pm.addPass(createTesseraTargetLowerPass());
  // pm.addPass(createCanonicalizerPass());
  // pm.addPass(createCSEPass());
}

void registerEBTPipelines() {
  // In-tree code would use PassPipelineRegistration like:
  // PassPipelineRegistration<>("tessera-ebt-canonicalize", "...",
  //   [](OpPassManager& pm){ addCanonicalizePipeline(pm); });
  // PassPipelineRegistration<>("tessera-ebt-lower", "...",
  //   [opts=params...](OpPassManager& pm){ addLowerPipeline(pm, opts); });
}

}} // ns
