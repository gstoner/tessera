if(NOT LLVM_EXTERNAL_LIT)
  find_program(LLVM_EXTERNAL_LIT llvm-lit lit.py lit)
endif()

set(TESSERA_NVIDIA_OPT_EXECUTABLE
    ${CMAKE_BINARY_DIR}/src/compiler/codegen/tessera_gpu_backend_NVIDIA/tools/tessera-nvidia-opt)
configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

if(LLVM_EXTERNAL_LIT)
  add_custom_target(check-tessera-nvidia
    COMMAND ${LLVM_EXTERNAL_LIT} -sv ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS tessera-nvidia-opt)
else()
  add_custom_target(check-tessera-nvidia
    COMMAND ${CMAKE_COMMAND} -E echo "llvm-lit not found - skipping Tessera NVIDIA lit tests"
    DEPENDS tessera-nvidia-opt)
endif()
