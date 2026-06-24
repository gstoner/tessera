add_mlir_library(TesseraROCMConversion
  TileToROCM.cpp
  ROCMWaveLdsPipeline.cpp
  TesseraTargetToROCDL.cpp
  KernelABI.cpp
  GenerateWMMAGemmKernel.cpp
  GenerateWMMAFlashAttnKernel.cpp
  GenerateWMMAFlashAttnBwdKernel.cpp
  Passes.cpp
  LINK_LIBS PUBLIC
    MLIRIR MLIRLLVMDialect MLIRROCDLDialect MLIRFuncDialect MLIRTransformUtils
    MLIRGPUDialect MLIRSCFDialect MLIRVectorDialect MLIRArithDialect
    MLIRMemRefDialect MLIRMathDialect
    TesseraIR
    TesseraROCMIR)
target_include_directories(TesseraROCMConversion PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../ir/include
  ${CMAKE_CURRENT_BINARY_DIR}/../IR)
