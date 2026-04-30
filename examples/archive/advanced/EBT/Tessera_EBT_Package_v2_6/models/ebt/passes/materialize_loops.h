#pragma once
#include "mlir/Pass/Pass.h"
namespace tessera { namespace ebt {
struct MaterializeLoopsOptions {
  int K=4, T=4;
};
std::unique_ptr<mlir::Pass> createMaterializeLoopsPass(MaterializeLoopsOptions);
void registerMaterializeLoopsPipeline();
}} // ns
