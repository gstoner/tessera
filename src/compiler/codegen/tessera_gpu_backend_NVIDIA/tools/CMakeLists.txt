add_executable(tessera-nvidia-opt tessera_nvidia_opt.cpp)

target_include_directories(tessera-nvidia-opt PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}/../lib/IR
)

target_link_libraries(tessera-nvidia-opt PRIVATE
  TesseraNVIDIAIR
  TesseraNVIDIAConversion
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIROptLib
)
