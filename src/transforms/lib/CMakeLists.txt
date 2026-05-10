
set(LLVM_LINK_COMPONENTS Support)

add_library(TesseraPasses
  # Phase 1
  CanonicalizeTesseraIR.cpp
  VerifyTesseraIR.cpp
  MigrateTesseraIR.cpp
  Passes.cpp

  # Phase 2 lowering chain
  DistributionLoweringPass.cpp
  EffectAnnotationPass.cpp
  TilingPass.cpp
  TileToX86Pass.cpp

  # Phase 3 GPU / FA-4 Tile IR lowering
  TileIRLoweringPass.cpp

  # Phase 4 distributed training
  GPUCollectiveInsertionPass.cpp
  PipelineStageInsertionPass.cpp

  # Phase F4 autodiff (reverse-mode via AdjointInterface).
  # ODS scaffold lives at src/compiler/ir/include/Tessera/AdjointInterface.td.
  AutodiffPass.cpp

  # Phase F5 — adjoint collective insertion (runs after AutodiffPass).
  AdjointCollectiveInsertionPass.cpp

  # Phase 8.4.8 — SwiGLU fusion (Stage 2b of SwiGLU Performance Plan).
  SwigluFusionPass.cpp

  # attention_variants_plan, MLA-1 — DeepSeek MLA decode fusion.
  MLAFusionPass.cpp

  # attention_variants_plan, NSA-4 — DeepSeek NSA fusion.
  NativeSparseAttnFusionPass.cpp

  # Phase 6 diagnostics
  ${CMAKE_SOURCE_DIR}/src/compiler/diagnostics/ShapeInferencePass.cpp
  ${CMAKE_SOURCE_DIR}/src/compiler/diagnostics/ErrorReporter.cpp

  ${CMAKE_SOURCE_DIR}/src/compiler/tile_opt_fa4/lib/WarpSpecializationPass.cpp
  ${CMAKE_SOURCE_DIR}/src/compiler/tile_opt_fa4/lib/AsyncCopyLoweringPass.cpp
  ${CMAKE_SOURCE_DIR}/src/compiler/codegen/tessera_gpu_backend_NVIDIA/NVWGMMALoweringPass.cpp
  ${CMAKE_SOURCE_DIR}/src/compiler/codegen/tessera_gpu_backend_NVIDIA/NVTMADescriptorPass.cpp
  ${CMAKE_SOURCE_DIR}/src/compiler/codegen/tessera_gpu_backend_NVIDIA/NVFlashAttnKernelEmitter.cpp
)

target_include_directories(TesseraPasses
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../compiler/ir/include
    ${CMAKE_CURRENT_BINARY_DIR}/../../compiler/ir
    ${CMAKE_CURRENT_SOURCE_DIR}/../../compiler/tile_opt_fa4/include
)

# Phase F4 — pass body needs the AdjointInterface generated header.
add_dependencies(TesseraPasses TesseraAdjointInterfaceTableGen)

target_link_libraries(TesseraPasses
  PUBLIC
    TesseraIR
    MLIRIR
    MLIRPass
    MLIRParser
    MLIRSupport
    MLIRTransforms
    # Phase 2 dialect dependencies
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRMemRefDialect
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    # Phase 3 dialect dependencies
    TesseraAttnDialect
    TesseraQueueDialect
    MLIRNVVMDialect
    MLIRGPUDialect
)

mlir_check_all_link_libraries(TesseraPasses)
