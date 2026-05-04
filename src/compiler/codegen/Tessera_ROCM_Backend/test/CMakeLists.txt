set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
if(NOT LLVM_EXTERNAL_LIT)
  find_program(LLVM_EXTERNAL_LIT llvm-lit lit.py lit)
endif()
set(TESSERA_ROCM_OPT_EXECUTABLE
    ${CMAKE_BINARY_DIR}/src/compiler/codegen/Tessera_ROCM_Backend/tools/tessera-rocm-opt)
configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)
if(LLVM_EXTERNAL_LIT)
  add_custom_target(check-tessera-rocm
    COMMAND ${LLVM_EXTERNAL_LIT} -sv ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS tessera-rocm-opt)
else()
  add_custom_target(check-tessera-rocm
    COMMAND ${CMAKE_COMMAND} -E echo "llvm-lit not found - skipping Tessera ROCm lit tests"
    DEPENDS tessera-rocm-opt)
endif()
