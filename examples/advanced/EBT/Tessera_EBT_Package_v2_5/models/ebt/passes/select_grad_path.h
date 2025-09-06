#pragma once
namespace mlir { class Pass; }
namespace tessera { namespace ebt {
mlir::Pass* createSelectGradPathPass(bool preferJVP);
void registerSelectGradPathPipeline();
}} // ns
