# Target-IR-only parser/verifier.  Unlike tessera-opt, this executable omits
# Graph and Tile dialects so a Target IR fixture cannot pass by relying on a
# permissive all-dialects registry.
add_executable(tessera-target-opt tessera-target-opt.cpp)
target_link_libraries(tessera-target-opt PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIROptLib
)

if(TARGET TesseraNVIDIAConversion)
  target_compile_definitions(tessera-target-opt PRIVATE TESSERA_HAVE_NVIDIA_TARGET_IR)
  target_include_directories(tessera-target-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/compiler/codegen/tessera_gpu_backend_NVIDIA/include
    ${CMAKE_BINARY_DIR}/src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/IR
  )
  target_link_libraries(tessera-target-opt PRIVATE
    TesseraNVIDIAIR
    TesseraNVIDIAConversion
  )
endif()
