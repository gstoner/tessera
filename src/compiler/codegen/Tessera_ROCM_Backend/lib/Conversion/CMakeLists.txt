add_mlir_library(TesseraROCMConversion
  TesseraTargetToROCDL.cpp
  KernelABI.cpp
  Passes.cpp
  LINK_LIBS PUBLIC
    MLIRIR MLIRLLVMDialect MLIRROCDLDialect MLIRFuncDialect MLIRTransformUtils
    TesseraROCMIR)
target_include_directories(TesseraROCMConversion PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
