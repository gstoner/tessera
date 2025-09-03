set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)
add_custom_target(check-tessera-rocm
  COMMAND ${LLVM_EXTERNAL_LIT} -sv ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS tessera-rocm-opt)
