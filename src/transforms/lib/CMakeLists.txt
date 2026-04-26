
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
)

target_include_directories(TesseraPasses
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ir
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
)

mlir_check_all_link_libraries(TesseraPasses)
