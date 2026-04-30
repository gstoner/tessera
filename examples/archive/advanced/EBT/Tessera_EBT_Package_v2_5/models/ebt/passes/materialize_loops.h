#pragma once
// Scaffold close to real MLIR code.
#include <cstdint>
namespace mlir { class Pass; class OpPassManager; }
namespace tessera { namespace ebt {
struct MaterializeLoopsOptions { int32_t K=4; int32_t T=4; };
mlir::Pass* createMaterializeLoopsPass(const MaterializeLoopsOptions&);
void registerMaterializeLoopsPipeline(); // PassPipelineRegistration wrapper
}} // ns
