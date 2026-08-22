add_mlir_library(TesseraNVIDIAConversion
  NVIDIALowering.cpp
  NVIDIADynamicShared.cpp
  GenerateNVIDIAPhiloxKernel.cpp

  LINK_LIBS PUBLIC
    TesseraIR
    TesseraNVIDIAIR
    TesseraPasses
    MLIRArithDialect
    MLIRAnalysis
    MLIRFuncDialect
    MLIRGPUDialect
    MLIRIR
    MLIRLLVMDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRNVVMDialect
    MLIRPass
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRSupport
    MLIRTransforms
)

target_include_directories(TesseraNVIDIAConversion
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_BINARY_DIR}/../IR
)
