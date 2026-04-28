
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
