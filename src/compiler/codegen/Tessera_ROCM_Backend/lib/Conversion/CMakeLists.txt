add_mlir_library(TesseraROCMConversion
  TileToROCM.cpp
  TesseraTargetToROCDL.cpp
  KernelABI.cpp
  GenerateWMMAGemmKernel.cpp
  Passes.cpp
  LINK_LIBS PUBLIC
    MLIRIR MLIRLLVMDialect MLIRROCDLDialect MLIRFuncDialect MLIRTransformUtils
    MLIRGPUDialect MLIRVectorDialect MLIRArithDialect MLIRMemRefDialect
    TesseraROCMIR)
target_include_directories(TesseraROCMConversion PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_BINARY_DIR}/../IR)
