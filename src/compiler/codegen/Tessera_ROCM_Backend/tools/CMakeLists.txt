add_executable(tessera-rocm-opt tessera_rocm_opt.cpp)
target_link_libraries(tessera-rocm-opt PRIVATE
  MLIRSupport MLIRIR MLIRParser MLIRPass MLIRArithDialect MLIRFuncDialect MLIRTransforms MLIRTranslateLib
  MLIRSCFDialect
  MLIRLLVMDialect MLIRROCDLDialect MLIROptLib TesseraROCMIR TesseraROCMConversion)

add_executable(tessera-rocm-emit tessera_rocm_emit.cpp)
target_link_libraries(tessera-rocm-emit PRIVATE MLIRSupport MLIRIR MLIRParser MLIRPass)
