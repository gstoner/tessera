#pragma once
namespace tessera { namespace neighbors {
void registerHaloInferPass();
void registerStencilLowerPass();
void registerBoundaryConditionLowerPass();
void registerPipelineOverlapPass();
void registerDynamicTopologyPass();
}} // namespace
