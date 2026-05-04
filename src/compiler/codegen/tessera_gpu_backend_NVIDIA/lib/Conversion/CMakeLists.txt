add_mlir_library(TesseraNVIDIAConversion
  NVIDIALowering.cpp

  LINK_LIBS PUBLIC
    TesseraNVIDIAIR
    MLIRArithDialect
    MLIRFuncDialect
    MLIRGPUDialect
    MLIRIR
    MLIRLLVMDialect
    MLIRNVVMDialect
    MLIRPass
    MLIRSupport
    MLIRTransforms
)

target_include_directories(TesseraNVIDIAConversion
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_BINARY_DIR}/../IR
)
