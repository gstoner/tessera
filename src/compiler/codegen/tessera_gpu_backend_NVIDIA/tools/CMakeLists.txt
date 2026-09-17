add_executable(tessera-nvidia-opt tessera_nvidia_opt.cpp)

target_include_directories(tessera-nvidia-opt PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}/../lib/IR
)

target_link_libraries(tessera-nvidia-opt PRIVATE
  TesseraIR
  TesseraNVIDIAIR
  TesseraNVIDIAConversion
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  MLIRGPUToNVVMTransforms       # convert-gpu-to-nvvm (single-invocation stamp fixture)
  MLIRSCFToControlFlow          # convert-scf-to-cf
  MLIRReconcileUnrealizedCasts  # reconcile-unrealized-casts
  MLIRArithToLLVM               # ConvertToLLVMPatternInterface extensions the
  MLIRControlFlowToLLVM         # NVVM lowering resolves per loaded dialect
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRUBToLLVM
  MLIROptLib
)
