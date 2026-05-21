#pragma once
namespace tessera { namespace neighbors {
void registerHaloInferPass();
void registerStencilLowerPass();
void registerBoundaryConditionLowerPass();
void registerStencilLoopMaterializePass();
void registerHaloMeshIntegrationPass();
void registerPipelineOverlapPass();
void registerDynamicTopologyPass();
}} // namespace
