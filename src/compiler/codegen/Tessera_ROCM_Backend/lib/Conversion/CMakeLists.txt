add_mlir_library(TesseraROCMConversion
  TileToROCM.cpp
  ROCMWaveLdsPipeline.cpp
  TesseraTargetToROCDL.cpp
  KernelABI.cpp
  GenerateWMMAGemmKernel.cpp
  GenerateWMMAFlashAttnKernel.cpp
  GenerateWMMAFlashAttnBwdKernel.cpp
  GenerateWMMALinearAttnKernel.cpp
  GenerateROCMActivationKernel.cpp
  GenerateROCMSiluMulKernel.cpp
  GenerateROCMPointwiseLossKernel.cpp
  GenerateROCMBinaryLossKernel.cpp
  GenerateROCMPolicyLossKernel.cpp
  GenerateROCMFpQuantKernel.cpp
  GenerateROCMDftKernel.cpp
  GenerateROCMSpmmKernel.cpp
  GenerateROCMSddmmKernel.cpp
  GenerateROCMSelectiveSsmKernel.cpp
  GenerateROCMCholeskyKernel.cpp
  GenerateROCMTriSolveKernel.cpp
  GenerateROCMLuKernel.cpp
  GenerateROCMQrKernel.cpp
  GenerateROCMSvdKernel.cpp
  GenerateROCMOptimizerKernel.cpp
  GenerateROCMPredicateKernel.cpp
  GenerateROCMMoeKernel.cpp
  GenerateROCMAlibiKernel.cpp
  GenerateROCMDeltaNetKernel.cpp
  GenerateROCMRopeKernel.cpp
  GenerateROCMSoftmaxKernel.cpp
  GenerateROCMNormKernel.cpp
  GenerateROCMReduceKernel.cpp
  GenerateROCMArgReduceKernel.cpp
  GenerateROCMScanKernel.cpp
  GenerateROCMUnaryKernel.cpp
  GenerateROCMBinaryKernel.cpp
  GenerateROCMCompareKernel.cpp
  GenerateROCMLogicalKernel.cpp
  GenerateROCMBitwiseKernel.cpp
  GenerateROCMWhereKernel.cpp
  LowerROCMAsyncCopyToLoop.cpp
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
