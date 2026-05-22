#pragma once
#include "mlir/Pass/Pass.h"
namespace tessera { namespace ebt {
std::unique_ptr<mlir::Pass> createSelectGradPathPass(bool preferJVP);
void registerSelectGradPathPipeline();
}} // ns
