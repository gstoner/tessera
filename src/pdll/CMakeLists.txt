
# Optional: generate native rewriters if mlir-pdll is available.
find_program(MLIR_PDLL_EXECUTABLE NAMES mlir-pdll mlir-pdll.exe)
if(MLIR_PDLL_EXECUTABLE)
  set(PDLL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/canon_cleanup.pdll)
  set(PDLL_GEN ${CMAKE_CURRENT_BINARY_DIR}/PDLLRewriters.inc)
  add_custom_command(
    OUTPUT ${PDLL_GEN}
    COMMAND ${MLIR_PDLL_EXECUTABLE} --gen-rewriters ${PDLL_SRC} -o ${PDLL_GEN}
    DEPENDS ${PDLL_SRC}
    VERBATIM
  )
  add_custom_target(TesseraPDLLGen DEPENDS ${PDLL_GEN})
  # Example: you could include ${PDLL_GEN} into a small library here.
endif()
