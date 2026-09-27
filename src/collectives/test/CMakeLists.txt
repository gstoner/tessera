# Shared validated lit runner (cmake/TesseraLit.cmake). add_lit_testsuite reads
# LLVM_EXTERNAL_LIT, so hand it the validated runner (directory scope only).
include(${CMAKE_CURRENT_LIST_DIR}/../../../cmake/TesseraLit.cmake)
tessera_lit_command(_tessera_collectives_lit)
if(_tessera_collectives_lit)
  set(LLVM_EXTERNAL_LIT "${_tessera_collectives_lit}")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py")
  add_lit_testsuite(check-tessera-collective "Run Tessera collective tests"
    ${CMAKE_CURRENT_SOURCE_DIR})
else()
  add_custom_target(check-tessera-collective
    COMMAND ${CMAKE_COMMAND} -E echo
            "No Tessera collective lit suite is defined - runtime smoke is the v1.4 validation lane")
endif()
